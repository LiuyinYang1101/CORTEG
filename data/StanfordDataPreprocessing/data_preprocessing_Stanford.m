%
% DESCRIPTIONS:
%   Preprocessing code for Stanford datasets. The datasets include 9 subjects (3 subjects were included in BCI4),
%   ECoG sampled at 1kHz, band pass filtered between 0.15 to 200 Hz. The
%   data glove was sampled at 25 Hz, presented in 1 kHz (the same length as the ECoG data).
%
% DEPENDENCE:
%   FieldTrip toolbox: fieldtrip-20230926
%   
% REFERENCE:
%   [1] Miller, Kai J. "A library of human electrocorticographic data and analyses." Nature human behaviour 3.11 (2019): 1225-1235.
%   [2] Miller, Kai J., et al. "Human motor cortical activity is selectively phase-entrained on underlying rhythms." (2012): e1002655.
%


close all
clear

fs = 1000;
dataLoc = '..\data_downloaded\Stanford\'; % locate the 'data' folder when u downloaded the dataset
subCode = {'bp', 'cc', 'ht','jc','jp','mv','wc','wm','zt'}; % subject code. Each subject has a folder under their code name.

% set configurations for data visualization
cfg = [];
cfg.viewmode = 'vertical';  
cfg.blocksize = 10;   
cfg.ylim = [-150, 150];  
cfg.preproc.demean = 'no';
cfg.preproc.lpfilter = 'yes';
cfg.preproc.lpfreq = 100;
cfg.preproc.hpfilter = 'yes';
cfg.preproc.hpfreq = 1;

for iS = 1:numel(subCode)
    load([dataLoc, subCode{iS},'\', subCode{iS}, '_fingerflex.mat']);
    
    % change the unit
    data = data'*0.0298;
    
    % band stop filtering
    data = ft_preproc_bandstopfilter(data, fs,[59.5 60.5], 3, 'but','onepass');
    data = ft_preproc_bandstopfilter(data, fs,[119.5, 120.5], 3, 'but','onepass');
    data = ft_preproc_bandstopfilter(data, fs,[179.5, 180.5], 3, 'but','onepass');

    % find and exclude bad channels (interactive)
    nCh = size(data, 1);
    channelName = arrayfun(@(x) sprintf('ch%d', x), 1:nCh, 'UniformOutput', false)';
    datav.trial = data;
    datav.label = channelName;
    datav.fsample = fs;
    datav.time = (1:1:size(datav.trial, 2))/datav.fsample;

    figure;ft_databrowser(cfg, datav);
    visual_badchannel = input('\nEnter faulty channel names: '); % e.g., {'ch1', 'ch2', ...}
    % Below is the removed channels
    % 'mv': {'ch40'}
    % 'jp': {'ch31','ch32','ch37'}
    
    idx = find(ismember(datav.label, visual_badchannel));
    data(idx, :) = [];     
    channelName(idx) = [];

    % CAR
    data = ft_preproc_rereference(data, 'all', 'avg');

    % save data
    fprintf('\n subject%d: done!\n', iS)
    save(['./preprocessed_data/Stanford/', subCode{iS},'.mat'], 'data', 'flex','fs', 'channelName')
end

