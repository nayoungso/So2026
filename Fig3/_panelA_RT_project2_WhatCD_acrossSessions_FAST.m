function RT_project2_WhatCD_acrossSessions_FAST(filelist,doSmooth, smoothWin)
% RT_project2_WhatCD_acrossSessions_FAST
%
% Pools the per-session "What projection" (S_d_dec/S_s_dec, from the
% dots-trained CD; S_d_choice/S_s_choice, from the saccade-trained CD;
% all dots- and saccade-aligned) saved by RT_project2_WhatCD_FAST.m
% across all sessions in filelist, by concatenating trials (not
% averaging per session first). Trials from all sessions are pooled
% within each signed-coherence group (all trials, and correct trials
% only) and averaged, with an optional Gaussian smoothing and a minimum-
% trial-count mask (time bins backed by fewer than min_trials trials are
% set to NaN). An additional "attrition" rule NaNs out dots-aligned time
% bins that fall within `cutoff` ms of a given trial's own RT, since
% those bins have already lost that trial to a saccade. Produces two
% figures (2x2 panels: all-trials / correct-only rows, dots-aligned /
% saccade-aligned columns) -- one for the dots-trained CD, one for the
% saccade-trained CD.
%
% INPUTS
%   filelist  : cell array of saved WhatProjection filenames (as saved
%               by RT_project2_WhatCD_FAST.m) to pool across.
%   doSmooth  : 0/1, smooth each coherence group's mean trace with a
%               Gaussian window before display.
%   smoothWin : smoothdata window size (samples), used only if doSmooth.
%
% OUTPUT
%   None returned (figures only).
%
% REQUIRES ON PATH: fig_setting.m (included alongside this file).
%
% EXAMPLE
%   RT_project2_WhatCD_acrossSessions_FAST(filelist, 0, [])

%% path
path = ['~/So2026/NeuralData/output/WhatProjections/'];

use_attrition = true;
cutoff = 200;
min_trials = 50;   % cut the time bins if the contributing trials are less than this number
%% -------------------------------
% initialize pooled variables
%% -------------------------------
S_d_dec_all = [];
S_s_dec_all = [];
S_d_choice_all = [];
S_s_choice_all = [];

coh_all_all = [];
correct_all = [];
rt_all = [];

%% -------------------------------
% load + concatenate
%% -------------------------------
for ii = 1:length(filelist)

    clear S_d_dec S_s_dec S_d_choice S_s_choice coh_all correct rt

    load([path,filelist{ii}], ...
        'S_d_dec','S_s_dec','S_d_choice','S_s_choice', ...
        'tvec_d','tvec_s','rt','coh_all','correct');

    % concatenate trials (columns)
    S_d_dec_all     = [S_d_dec_all, S_d_dec];
    S_s_dec_all     = [S_s_dec_all, S_s_dec];
    S_d_choice_all  = [S_d_choice_all, S_d_choice];
    S_s_choice_all  = [S_s_choice_all, S_s_choice];

    coh_all_all = [coh_all_all; coh_all(:)];
    correct_all = [correct_all; correct(:)];
    rt_all = [rt_all; rt(:)];
end

% attrition rule
if use_attrition
    valid_end = rt_all' - cutoff;   % 1 × trials
    mask = tvec_d(:) <= valid_end;     % implicit expansion

    %S_d_dec_all_masked = S_d_dec_all;
    S_d_dec_all(~mask) = nan;
    S_d_choice_all(~mask) = nan;
    %n_tr_d = sum(~isnan(S_d_dec_all),2);

end

%% -------------------------------
% coherence definition
%% -------------------------------
coh_set = [-400 -200 -100 -50 -25 -1 1 25 50 100 200 400];
nC = length(coh_set);

coh_idx = nan(size(coh_all_all));

for k = 1:nC
    coh_idx(coh_all_all == coh_set(k)) = k;
end

%% -------------------------------
% compute means (trial-weighted)
%% -------------------------------
nT_d = size(S_d_dec_all,1);
nT_s = size(S_s_dec_all,1);

mean_d_dec    = nan(nT_d,nC,2);
mean_s_dec    = nan(nT_s,nC,2);
mean_d_choice = nan(nT_d,nC,2);
mean_s_choice = nan(nT_s,nC,2);

for k = 1:nC

    idx_all  = (coh_idx == k);
    idx_corr = idx_all & (correct_all == 1);

    if any(idx_all)
        mean_d_dec(:,k,1)    = nanmean(S_d_dec_all(:,idx_all),2);
        mean_s_dec(:,k,1)    = nanmean(S_s_dec_all(:,idx_all),2);
        mean_d_choice(:,k,1) = nanmean(S_d_choice_all(:,idx_all),2);
        mean_s_choice(:,k,1) = nanmean(S_s_choice_all(:,idx_all),2);
        
        if doSmooth
           mean_d_dec(:,k,1) = smoothdata(mean_d_dec(:,k,1), 'gaussian', smoothWin);
           mean_s_dec(:,k,1) = smoothdata(mean_s_dec(:,k,1), 'gaussian', smoothWin);
           mean_d_choice(:,k,1) = smoothdata(mean_d_choice(:,k,1), 'gaussian', smoothWin);
           mean_s_choice(:,k,1) = smoothdata(mean_s_choice(:,k,1), 'gaussian', smoothWin);
        end

        % time bins with less than certain trial no will be nan'ed
        n_tr_d = sum(~isnan(S_d_dec_all(:,idx_all)),2);
        mean_d_dec(n_tr_d<min_trials,k,1) = nan;
        mean_d_choice(n_tr_d<min_trials,k,1) = nan;

    end

    if any(idx_corr)
        mean_d_dec(:,k,2)    = nanmean(S_d_dec_all(:,idx_corr),2);
        mean_s_dec(:,k,2)    = nanmean(S_s_dec_all(:,idx_corr),2);
        mean_d_choice(:,k,2) = nanmean(S_d_choice_all(:,idx_corr),2);
        mean_s_choice(:,k,2) = nanmean(S_s_choice_all(:,idx_corr),2);
        
        if doSmooth
           mean_d_dec(:,k,2) = smoothdata(mean_d_dec(:,k,2), 'gaussian', smoothWin);
           mean_s_dec(:,k,2) = smoothdata(mean_s_dec(:,k,2), 'gaussian', smoothWin);
           mean_d_choice(:,k,2) = smoothdata(mean_d_choice(:,k,2), 'gaussian', smoothWin);
           mean_s_choice(:,k,2) = smoothdata(mean_s_choice(:,k,2), 'gaussian', smoothWin);
        end

        % time bins with less than certain trial no will be nan'ed
        n_tr_d = sum(~isnan(S_d_dec_all(:,idx_corr)),2);
        mean_d_dec(n_tr_d<min_trials,k,2) = nan;
        mean_d_choice(n_tr_d<min_trials,k,2) = nan;

    end
end

%% optional smoothing



%% -------------------------------
% color
%% -------------------------------
ColorMapCoh = [0 0.5 0; 0 0.7 0.3; 0.4 0.8 0.6; 0.4 0.8 0.6; ...
               0.6 0.9 0.8; 0.8 1 0.8; ...
               1 1 0.8; 1 1 0.6; 1 0.9 0.5; ...
               1 0.8 0.4; 1 0.6 0; 1 0 0];

%% -------------------------------
% time vectors (from first session assumption)
%% -------------------------------
% safer: infer from size
binwidth = 25;
%tvec_d = -100:binwidth:(-100 + (nT_d-1)*binwidth);
%tvec_s = -550:binwidth:(-550 + (nT_s-1)*binwidth);

%% ===============================
% FIGURE 1 (dots-trained)
%% ===============================
figure(1); clf;

ymin = min([mean_d_dec(:); mean_s_dec(:)], [], 'omitnan') - 0.1;
ymax = max([mean_d_dec(:); mean_s_dec(:)], [], 'omitnan') + 0.1;

for row = 1:2

    subplot(2,2,(row-1)*2+1); hold on;
    line([0 0],[ymin ymax],'Color','k');

    for k = 1:nC
        plot(tvec_d, mean_d_dec(:,k,row), 'Color',ColorMapCoh(k,:), 'LineWidth',1.5);
    end
    ylim([ymin ymax]); box off; fig_setting();

    subplot(2,2,(row-1)*2+2); hold on;
    line([0 0],[ymin ymax],'Color','k');
    for k = 1:nC
        plot(tvec_s, mean_s_dec(:,k,row), 'Color',ColorMapCoh(k,:), 'LineWidth',1.5);
    end
    ylim([ymin ymax]); box off; fig_setting();

end

sgtitle('Session-averaged What (dots-trained)');

%% ===============================
% FIGURE 2 (sacc-trained)
%% ===============================
figure(2); clf;

ymin = min([mean_d_choice(:); mean_s_choice(:)], [], 'omitnan') - 0.1;
ymax = max([mean_d_choice(:); mean_s_choice(:)], [], 'omitnan') + 0.1;

for row = 1:2

    subplot(2,2,(row-1)*2+1); hold on;
    line([0 0],[ymin ymax],'Color','k');

    for k = 1:nC
        plot(tvec_d, mean_d_choice(:,k,row), ...
            'Color',ColorMapCoh(k,:), 'LineWidth',1.5);
    end

    ylim([ymin ymax]); box off; fig_setting();

    subplot(2,2,(row-1)*2+2); hold on;
    line([0 0],[ymin ymax],'Color','k');

    for k = 1:nC
        plot(tvec_s, mean_s_choice(:,k,row), ...
            'Color',ColorMapCoh(k,:), 'LineWidth',1.5);
    end

    ylim([ymin ymax]); box off; fig_setting();

end

sgtitle('Session-averaged What (sacc-trained)');

end

