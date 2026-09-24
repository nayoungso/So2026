function RT_project2_WhenCD_acrossSessions_FAST(monkID, filelist,doSmooth, smoothWin)
% RT_project2_WhenCD_acrossSessions_FAST
%
% Pools the per-session "When projection" (S_d dots-aligned, S_s
% saccade-aligned) saved by RT_project2_WhenCD_FAST.m across all
% sessions in filelist, onto a common dots-aligned time axis
% (t_min:binwidth:t_max = -100:25:800 ms; saccade-aligned axis is
% assumed identical across sessions and taken from the first file).
% Trials from all sessions are pooled within each RT group and averaged
% (trial-weighted grand mean +/- group), then plotted dots- and
% saccade-aligned (2-panel figure).
%
% INPUTS
%   monkID    : 'H', 'N', or 'A' (both monkeys) -- selects the plot
%               title and the saved filenames only; does not filter
%               filelist itself (filelist should already contain only
%               that monkey's sessions, except for 'A').
%   filelist  : cell array of saved WhenProjection filenames (as saved
%               by RT_project2_WhenCD_FAST.m) to pool across.
%   doSmooth  : 0/1, smooth each RT group's mean trace with a Gaussian
%               window before display.
%   smoothWin : smoothdata window size (samples), used only if doSmooth.
%
% OUTPUT
%   None returned; saves the grand-average traces to [path
%   'WhenProjections/' datfilename] and the summary figure to [path
%   'WhenProjections/figs/' figfilename] (filenames depend on monkID).
%
% REQUIRES ON PATH: fig_setting.m (included alongside this file).
%
% EXAMPLE
%   RT_project2_WhenCD_acrossSessions_FAST('N', filelist, 0, [])

path = ['~/So2026/NeuralData/output/'];

%% -------------------------------
% Titles
%% -------------------------------
if monkID=='H'
    %titlestring = 'Average S^{when} across sessions (Harry); baseline adjusted';
    titlestring = 'Average S^{when} across sessions (Harry)';
    figfilename = 'H_Grand_Swhen_RTgroup_FAST.fig';
    %figfilename = 'H_Grand_Swhen_RTgroup_FAST_baselineAdjusted.fig';
    datfilename = 'H_Grand_Swhen_RTgroup_FAST.mat';

elseif monkID=='N'
    %titlestring = 'Average S^{when} across sessions (Neptune); baseline adjusted';
    titlestring = 'Average S^{when} across sessions (Neptune)';
    figfilename = 'N_Grand_Swhen_RTgroup_FAST.fig';
    %figfilename = 'N_Grand_Swhen_RTgroup_FAST_baselineAdjusted.fig';
    datfilename = 'N_Grand_Swhen_RTgroup_FAST.mat';

elseif monkID=='A'
    titlestring = 'Average S^{when} across sessions (Both monkeys)';
    figfilename = 'All_Grand_Swhen_RTgroup_FAST.fig';
    %figfilename = 'All_Grand_Swhen_RTgroup_FAST_baselineAdjusted.fig';
    datfilename = 'All_Grand_Swhen_RTgroup_FAST.mat';
end

%% -------------------------------
% Define COMMON time axis (IMPORTANT)
%% -------------------------------
binwidth = 25;
t_min = -100;
t_max = 800;   % <-- adjust if needed

common_tvec_d = t_min:binwidth:t_max;
nT_common = length(common_tvec_d);

%% -------------------------------
% Initialize
%% -------------------------------
S_d_all = [];
S_s_all = [];
rt_group_all = [];

%% -------------------------------
% Concatenate across sessions
%% -------------------------------
for ii = 1:length(filelist)

    load([path,'WhenProjections/',filelist{ii}], ...
        'S_d','S_s','rt_group','tvec_d','tvec_s');

    %% -------- DOTS alignment (FIXED) --------
    S_d_trim = nan(nT_common, size(S_d,2));

    % align by matching time points
    [~, idx_common, idx_local] = intersect(common_tvec_d, tvec_d);

    S_d_trim(idx_common,:) = S_d(idx_local,:);

    %% -------- SACCADE (already consistent) --------
    if ii == 1
        common_tvec_s = tvec_s;  % assume identical across sessions
    end

    S_s_trim = S_s;

    %% -------- concatenate safely --------
    if isempty(S_d_all)
        S_d_all = S_d_trim;
        S_s_all = S_s_trim;
    else
        S_d_all = [S_d_all, S_d_trim];
        S_s_all = [S_s_all, S_s_trim];
    end

    rt_group_all = [rt_group_all; rt_group(:)];
end

rt_groupno = max(rt_group_all);

%% -------------------------------
% Compute means (trial-weighted)
%% -------------------------------
for k = 1:rt_groupno
    idx = (rt_group_all == k);

    mean_S_d(:,k) = nanmean(S_d_all(:,idx),2);
    mean_S_s(:,k) = nanmean(S_s_all(:,idx),2);
    
    if doSmooth
        mean_S_d(:,k) = smoothdata(mean_S_d(:,k), 'gaussian', smoothWin);
        mean_S_s(:,k) = smoothdata(mean_S_s(:,k), 'gaussian', smoothWin);
    end

    % ---- baseline subtract at t = 0 ---- As of 4/6/26, not using this feature
    [~, idx0] = min(abs(common_tvec_d));  % find t=0 index robustly
    idx200 = find(common_tvec_d==200);

    mean_S_d_bs = mean_S_d - mean_S_d(idx200,:);
    mean_S_s_bs = mean_S_s - mean_S_d(idx200,:);

    GrandTrno(k) = sum(idx);
end

%% -------------------------------
% Plot
%% -------------------------------
ColorMapRT = [0 0 1; 0.1 0.3 1; 0.1 0.6 1; 0.1 0.9 1; 0.5 1 1; 0.9 1 1];

figure; clf;

ymin = min([mean_S_d(:); mean_S_s(:)]) - 0.1;
ymax = max([mean_S_d(:); mean_S_s(:)]) + 0.1;

%ymin = min([mean_S_d_bs(:); mean_S_s_bs(:)]) - 0.1;
%ymax = max([mean_S_d_bs(:); mean_S_s_bs(:)]) + 0.1;

% ---- dots ----
subplot(1,2,1); hold on;
line([0 0],[ymin ymax],'Color','k');

for k = 1:rt_groupno
    plot(common_tvec_d, mean_S_d(:,k),'Color',ColorMapRT(k,:), 'LineWidth',1.5);
    %plot(common_tvec_d, mean_S_d_bs(:,k),'Color',ColorMapRT(k,:), 'LineWidth',1.5);
end

xlabel('Time from dots (ms)');
ylabel('Projection');
title('Dots aligned');
ylim([ymin ymax]);
box off;
fig_setting();

% ---- sacc ----
subplot(1,2,2); hold on;
line([0 0],[ymin ymax],'Color','k');

for k = 1:rt_groupno
    plot(common_tvec_s, mean_S_s(:,k),'Color',ColorMapRT(k,:), 'LineWidth',1.5);
    %plot(common_tvec_s, mean_S_s_bs(:,k),'Color',ColorMapRT(k,:), 'LineWidth',1.5);
end

xlabel('Time from saccade (ms)');
title('Saccade aligned');
ylim([ymin ymax]);
box off;
fig_setting();

sgtitle(titlestring);

%% -------------------------------
% Save
%% -------------------------------
%{
save([path,'WhenProjections/',datfilename], ...
    'mean_S_d','mean_S_s','S_d_all','S_s_all','rt_group_all','GrandTrno', ...
    'common_tvec_d','common_tvec_s');

saveas(gcf,[path,'WhenProjections/figs/',figfilename]);
%}
end
