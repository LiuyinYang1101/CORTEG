"""
DESCRIPTION:
    Prepare the ECoG and trajectory data in a task format, i.e., pair prior 1-s ECoG epoch with the next trjectory point.
    Ideally this procedure should be done within each fold, but we opt for first extracting features
    for the whole dataset and then do chronological cross-validation. This way, we avoided repeated feature extraction,
    which will drastically increase time.

"""

from scipy.io import loadmat
import pickle
import numpy as np
import mne
from mne.filter import resample
from scipy.signal import hilbert

# metadata for different datasets
datasets = {
    "Stanford": {
        "subjects": 9,
        "subject_name": ['bp', 'cc', 'ht', 'jc', 'jp', 'mv', 'wc', 'wm', 'zt'],
        "fs_ecog": 1000,
        "fs_dg": 25,
        # file path to the preprocessed data from MATLAB
        "path": './preprocessed_data/Stanford/'
    },
}

def HGALFS_feature_extractor(data, traj, fs, window_size=1000, step_size = 50, T = 200, delay = None):
    """
    INPUT:
        data: [time, channel] ECoG data, 1 kHz
        traj: [time, 5] trajectory of 5 fingers, 1 kHz
        window_size: window size of ECoG in ms
        step_size: step size of sliding windows in ms
        T: number of time points after downsampling
        delay: optional time delay between ECoG and trajectory

    OUTPUT:
        X: [nEpoch, nChannel, T, nBand]
        Y: [nEpoch, 5]
        X0: [nEpoch, nChannel, window_size/fs * 128 Hz]
    """
    assert data.shape[0] == traj.shape[0], "ECoG and Trajectory must be same shape"
    data = data.astype(np.float64)
    traj = traj.astype(np.float64)

    if delay is not None:
        data, traj = data[delay:, :], traj[:-delay, :]

    # Downsample trajectory to match step size
    traj_ds = resample(traj.T, down=step_size, npad='auto').T

    X, Y  = [], []
    X0 = []
    n_samples = len(traj_ds)
    for k in range(n_samples):
        end = k * step_size
        start = end - window_size
        if start < 0:
            continue
        data_win = data[start:end].T
        traj_point = traj_ds[k]

        n_channels, n_times = data_win.shape
        subwin_len = n_times // T  # for downsampling

        # extract HGA (n_channels, T)
        iir_params = dict(order=4, ftype='butter')
        data_filtered = mne.filter.filter_data(data_win, sfreq=fs,
                                          l_freq=70 , h_freq=200,
                                          method='iir', iir_params=iir_params,
                                          verbose=False)
        analytic = hilbert(data_filtered, axis=1)
        envelope = np.abs(analytic)  # (n_channels, n_times)
        envelope = envelope[..., :subwin_len * T]

        HGA = envelope.reshape(n_channels, T, subwin_len).mean(axis=2)

        # extract LFS (n_channels, T)
        LFS = resample(data_win, down=5, npad='auto')

        # -> (n_channels, T, n_bands)
        features = np.stack([HGA, LFS], axis=-1)

        X.append(features)
        Y.append(traj_point)

        data_win_ds = resample(data_win, up=16, down=125, npad='auto')
        X0.append(data_win_ds)

    return np.array(X), np.array(Y), np.array(X0)

def main():
    # select one dataset for analysis
    selected = 'Stanford'
    fileLoc = datasets[selected]['path']
    fs_ecog = datasets[selected]['fs_ecog']
    fs_dg = datasets[selected]['fs_dg']

    for iS in range(datasets[selected]['subjects']):  # datasets[selected]['subjects']
        fileName = fileLoc + datasets[selected]['subject_name'][iS] + '.mat'
        data1 = loadmat(fileName)  # data [channel, time], flex [time, 5]

        # get data
        ECoG_1k = data1['data'].T
        trajectory_1k = data1['flex']

        # split train and test dataset
        if ECoG_1k.shape[0] // fs_ecog >= 600:
            ECoG_1k_train, trajectory_1k_train = ECoG_1k[0:400 * fs_ecog, :], trajectory_1k[0:400 * fs_ecog, :]
            ECoG_1k_test, trajectory_1k_test = ECoG_1k[400 * fs_ecog:, :], trajectory_1k[400 * fs_ecog:, :]
        else:
            len = ECoG_1k.shape[0] // 3 * 2
            ECoG_1k_train, trajectory_1k_train = ECoG_1k[0:len, :], trajectory_1k[0:len, :]
            ECoG_1k_test, trajectory_1k_test = ECoG_1k[len:, :], trajectory_1k[len:, :]

        ECoG_train, trajectory_train, ECoG_train_128 = HGALFS_feature_extractor(ECoG_1k_train, trajectory_1k_train,
                                                                                fs_ecog,
                                                                                window_size=1 * fs_ecog,
                                                                                step_size=fs_ecog // fs_dg, T=200,
                                                                                delay=40)
        ECoG_test, trajectory_test, ECoG_test_128 = HGALFS_feature_extractor(ECoG_1k_test, trajectory_1k_test, fs_ecog,
                                                                             window_size=1 * fs_ecog,
                                                                             step_size=fs_ecog // fs_dg, T=200,
                                                                             delay=40)

        filename = fileLoc + f"{datasets[selected]['subject_name'][iS]}_features.pkl"

        with open(filename, "wb") as f:
            pickle.dump([ECoG_train, trajectory_train, ECoG_test, trajectory_test, ECoG_train_128, ECoG_test_128], f)

        print(f"Finished subject: {iS}\n", flush=True)

if __name__ == "__main__":
    main()


