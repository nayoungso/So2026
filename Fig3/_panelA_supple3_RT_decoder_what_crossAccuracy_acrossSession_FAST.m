function RT_decoder_what_crossAccuracy_acrossSession_FAST(monkID)
% RT_decoder_what_crossAccuracy_acrossSession_FAST
%
% Averages the per-session cross-temporal decoding accuracy
% (accuracy_dxd, accuracy_dxs, accuracy_sxd, accuracy_sxs, saved by
% RT_decoder_what_FAST.m) across all sessions found for the given
% monkey, using a numerically stable streaming (incremental) mean.
% Produces a 4-panel heatmap figure: train-dots/test-dots,
% train-dots/test-saccade, train-saccade/test-dots, and
% train-saccade/test-saccade generalization accuracy.
%
% INPUTS
%   monkID : 'H' or 'N' -- selects which sessions' saved WhatCD files
%            (CDPath's 'H*_WhatCD*_FAST.mat' or 'N*_WhatCD*_FAST.mat')
%            to average across. Files missing the accuracy_dxd field
%            are skipped with a warning.
%
% OUTPUT
%   None returned (figure only).
%
% EXAMPLE
%   RT_decoder_what_crossAccuracy_acrossSession_FAST('N')

CDPath = ['~/So2026/NeuralData/output/WhatCD/'];

%% -------------------------------
% file list
%% -------------------------------
if monkID=='H'
    fileList = dir(fullfile(CDPath,'H*_WhatCD*_FAST.mat'));
elseif monkID=='N'
    fileList = dir(fullfile(CDPath,'N*_WhatCD*_FAST.mat'));
end

WhatCDList = {fileList.name};
nFiles = length(WhatCDList);

fprintf('Found %d sessions\n', nFiles);

%% -------------------------------
% initialize accumulators
%% -------------------------------
mean_accuracy_dxd = [];
mean_accuracy_dxs = [];
mean_accuracy_sxd = [];
mean_accuracy_sxs = [];

valid_count = 0;

%% -------------------------------
% loop (streaming average)
%% -------------------------------
for ii = 1:nFiles

    fname = WhatCDList{ii};
    data = load([CDPath, fname]);

    % --- check required fields ---
    if ~isfield(data,'accuracy_dxd')
        warning('Skipping %s (missing variables)', fname);
        continue
    end

    valid_count = valid_count + 1;

    if valid_count == 1
        % initialize
        mean_accuracy_dxd = data.accuracy_dxd;
        mean_accuracy_dxs = data.accuracy_dxs;
        mean_accuracy_sxd = data.accuracy_sxd;
        mean_accuracy_sxs = data.accuracy_sxs;

        % also grab time axes
        searchT1 = data.searchT1;
        searchT2 = data.searchT2;

    else
        % incremental mean (numerically stable)
        alpha = 1 / valid_count;

        mean_accuracy_dxd = (1-alpha)*mean_accuracy_dxd + alpha*data.accuracy_dxd;
        mean_accuracy_dxs = (1-alpha)*mean_accuracy_dxs + alpha*data.accuracy_dxs;
        mean_accuracy_sxd = (1-alpha)*mean_accuracy_sxd + alpha*data.accuracy_sxd;
        mean_accuracy_sxs = (1-alpha)*mean_accuracy_sxs + alpha*data.accuracy_sxs;
    end

end

fprintf('Used %d valid sessions\n', valid_count);

%% -------------------------------
% plot
%% -------------------------------
figure(2); clf;

subplot(221)
imagesc(searchT1, searchT1, mean_accuracy_dxd);
xlabel('Test (dots)');
ylabel('Train (dots)');
clim([0.5 1]);
axis xy; colorbar;
%xlabel('Tested on time window, from motion on (ms)');
%ylabel('Trained on time window, from motion on (ms)');

subplot(222)
imagesc(searchT2, searchT1, mean_accuracy_dxs);
xlabel('Test (sacc)');
ylabel('Train (dots)');
clim([0.5 1]);
%xlabel('Tested on time window, from sacc (ms)');
axis xy; colorbar;

subplot(223)
imagesc(searchT1, searchT2, mean_accuracy_sxd);
xlabel('Test (dots)');
ylabel('Train (sacc)');
clim([0.5 1]);
axis xy; colorbar;
%xlabel('Tested on time window, from motion on (ms)');
%ylabel('Trained on time window, from sacc (ms)');

subplot(224)
imagesc(searchT2, searchT2, mean_accuracy_sxs);
xlabel('Test (sacc)');
ylabel('Train (sacc)');
clim([0.5 1]);
axis xy; colorbar;
%xlabel('Tested on time window, from sacc (ms)');

sgtitle(['What decoder cross-temporal accuracy (n = ', num2str(valid_count), ')']);

%% -------------------------------
% save
%% -------------------------------
%{
save([CDPath,monkID,'_WhatDecoder_crossAccuracy_grand_FAST.mat'], ...
    'mean_accuracy_dxd','mean_accuracy_dxs','mean_accuracy_sxd','mean_accuracy_sxs', ...
    'searchT1','searchT2','valid_count');

saveas(gcf,[CDPath,'figs/',monkID,'_WhatDecoder_crossAccuracy_grand_FAST.fig']);
%}

end