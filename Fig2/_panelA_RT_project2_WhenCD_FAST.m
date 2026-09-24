function RT_project2_WhenCD_FAST(dates,file1,taskid,useNorm)
% RT_project2_WhenCD_FAST
%
% Projects the "When" coding direction (WhenCD, trained by
% RT_decoder_when_v4_2026_FAST.m and saved under file1) onto held-out
% population spike counts from the same session, producing a
% time-varying "When projection" S_d (dots-aligned) and S_s
% (saccade-aligned). Trials are grouped into monkey-specific RT bins
% and the mean projection per RT group is plotted, dots- and
% saccade-aligned (2-panel figure).
%
% Trial selection and RT-cutoff logic match RT_decoder_when_v4_2026_FAST.m
% exactly (same taskid/response/RTcutoff filter), so the projection is
% computed on the same trial population the decoder was trained on
% (train and test trials both included here -- this script does not
% re-apply the decoder's own train/test split).
%
% INPUTS
%   dates    : session identifier string (e.g. 'N210513'), used to load
%              [datapath dates '.mat'] (trials, Neuron, Session).
%   file1    : filename (within CDpath) of the .mat saved by
%              RT_decoder_when_v4_2026_FAST.m for this session
%              (contains whenCD, whenCD_norm).
%   taskid   : scalar taskid to select trials (exact match).
%   useNorm  : logical; if true, uses whenCD_norm (norm-normalized
%              weights) instead of the raw weights.
%
% OUTPUT
%   None returned; saves S_d, S_s, rt_all, rt_group, whenCD, tvec_d,
%   tvec_s, useNorm, choice_all, correct_all, dir_all, coh_all,
%   orig_trial_idx to [outputpath dates '_taskid' taskid
%   '_WhenProjection_FAST.mat'], plus the summary figure to [outputpath
%   'figs/' dates '_taskid' taskid '_FAST.fig'].
%   The saved .mat file is the input consumed by
%   RT_project2_WhenCD_acrossSessions_FAST.m and the slope_bySession
%   scripts.
%
% REQUIRES ON PATH: fig_setting.m (included alongside this file).
%
% EXAMPLE
%   RT_project2_WhenCD_FAST('N210513', 'N210513_WhenCD_taskid_21_v4_FAST.mat', 21, false)

%% paths
datapath = ['~/So2026/NeuralData/session_mat/'];
CDpath =  ['~/So2026/NeuralData/output/WhenCD/'];
outputpath = ['~/So2026/NeuralData/output/WhenProjections/'];

%% load decoder
load([CDpath,file1])
%w = coef_when(2:end);
if useNorm
    whenCD = whenCD_norm;
end
whenCD;

%% load data
load([datapath,dates,'.mat']);  % trials, Neuron, Session
cellno = Session.n_neurons;

%% params (match decoder)
if dates(1)=='H'
    RTcutoff = 550;
elseif dates(1)=='N'||dates(1)=='D'
    RTcutoff = 400;
end

binwidth = 25;
starttime = -200;

% saccade-aligned window
pre_sacc = -650;
post_sacc = 350;

tic

%% -------------------------------
% Trial selection
%% -------------------------------
trialno = 0;

for j = 1:length(trials)

    if (trials(j).taskid == taskid) && ...
       (trials(j).response>=0) && ...
       ((trials(j).time_sacc(1)-trials(j).time_dots_on(1))*1000 >= RTcutoff)

        trialno = trialno + 1;

        valid_trials(trialno) = j;
        
        orig_trial_idx(trialno,1) = j;        
        
        % ---- choice (same logic as decoder) ----
        dir_sign = (trials(j).dot_dir ~= trials(j).t1_dir);  % 0 for up/right 1 for down/left
        
        if ((dir_sign == 0)&&(trials(j).response == 1))||...
                ((dir_sign==1)&&(trials(j).response == 0))
            choice_all(trialno,1) = 0; % right/up
        else
            choice_all(trialno,1) = 1; % left/down
        end
        
        % ---- correctness ----
        correct_all(trialno,1) = trials(j).response;
        
        % ---- store direction and coh ----
        dir_all(trialno,1) = trials(j).dot_dir;
        coh_all(trialno,1) = trials(j).dot_coh*(1-2*dir_sign);

        rt_all(trialno,1) = (trials(j).time_sacc(1)-trials(j).time_dots_on(1))*1000;

        dots_times(trialno) = trials(j).time_dots_on(1)*1000;
        sacc_times(trialno) = trials(j).time_sacc(1)*1000;

        % dots-aligned bins (variable)
        endtime_temp = (trials(j).time_sacc(1)*1000 - 50) - dots_times(trialno);
        binno(trialno) = round((endtime_temp-starttime)/binwidth);

        cnt_startt_array{trialno} = starttime:binwidth:starttime+(binno(trialno)-1)*binwidth;
    end
end

nTrials = trialno;
maxBin = max(binno);

%% -------------------------------
% Build DOTS-aligned All_cnt
%% -------------------------------
All_cnt = nan(nTrials, cellno, maxBin);

for i = 1:cellno
    spikes = Neuron{i}.spikes;

    for t = 1:nTrials
        j = valid_trials(t);

        spike_ts = spikes{j} * 1000;
        spike_ts_aligned = spike_ts - dots_times(t);

        edges = [cnt_startt_array{t}, cnt_startt_array{t}(end)+binwidth];
        cnt = histcounts(spike_ts_aligned, edges);

        All_cnt(t,i,1:length(cnt)) = cnt;
    end
end

%% -------------------------------
% Build SACCADE-aligned All_cnt
%% -------------------------------
tvec_s = pre_sacc:binwidth:post_sacc;
nT_s = length(tvec_s);

All_cnt_s = nan(nTrials, cellno, nT_s);

for i = 1:cellno
    spikes = Neuron{i}.spikes;

    for t = 1:nTrials
        j = valid_trials(t);

        spike_ts = spikes{j} * 1000;
        spike_ts_aligned = spike_ts - sacc_times(t);

        edges = [tvec_s, tvec_s(end)+binwidth];
        cnt = histcounts(spike_ts_aligned, edges);

        All_cnt_s(t,i,:) = cnt;
    end
end

%% -------------------------------
% Projection (vectorized)
%% -------------------------------

% ---- dots ----
All_cnt_perm = permute(All_cnt,[3 1 2]);  % time × trial × neuron
nT_d = size(All_cnt_perm,1);

X_d = reshape(All_cnt_perm, [], cellno);
proj_d = X_d * whenCD;
S_d = reshape(proj_d, nT_d, nTrials);

% ---- sacc ----
All_cnt_s_perm = permute(All_cnt_s,[3 1 2]);
X_s = reshape(All_cnt_s_perm, [], cellno);
proj_s = X_s * whenCD;
S_s = reshape(proj_s, nT_s, nTrials);

%% -------------------------------
% RT grouping
%% -------------------------------
if dates(1)=='H'
    rt_bounds = [0 600 750 900 1050 1200 2000];
elseif dates(1)=='N'
    rt_bounds = [0 400 480 560 640 720 2000];
end

rt_group = discretize(rt_all, rt_bounds);
rt_groupno = max(rt_group);

for k = 1:rt_groupno
    idx = (rt_group == k);

    mean_S_d(:,k) = nanmean(S_d(:,idx),2);
    mean_S_s(:,k) = nanmean(S_s(:,idx),2);
end

%% -------------------------------
% Time vectors (FIXED)
%% -------------------------------
tvec_d = starttime:binwidth:(starttime + (nT_d-1)*binwidth);

%% -------------------------------
% Plot (FIXED)
%% -------------------------------
ColorMapRT = [0 0 1; 0.1 0.3 1; 0.1 0.6 1; 0.1 0.9 1; 0.5 1 1; 0.9 1 1];

figure(6); clf;

ymin = min([min(min(mean_S_d)) min(min(mean_S_s))])-0.1;
ymax = max([max(max(mean_S_d)) max(max(mean_S_s))])+0.1;

% ---- dots aligned ----
subplot(1,2,1); hold on;
line([0 0],[ymin ymax],'Color','k');
for k = 1:rt_groupno
    plot(tvec_d, mean_S_d(:,k),'Color',ColorMapRT(k,:), 'LineWidth',1.5);
end
xlabel('Time from dots (ms)');
ylabel('Projection');
ylim([ymin ymax]);
box off;
fig_setting();
title('nascent WhenCD projection');

% ---- sacc aligned ----
subplot(1,2,2); hold on;
line([0 0],[ymin ymax],'Color','k');
for k = 1:rt_groupno
    plot(tvec_s, mean_S_s(:,k),'Color',ColorMapRT(k,:), 'LineWidth',1.5);
end
title('Saccade aligned');
xlabel('Time from saccade (ms)');
ylim([ymin ymax]);
box off;
fig_setting();

sgtitle([dates,' When projection, taskid = ', num2str(taskid)]);

%% -------------------------------
% Save
%% -------------------------------
outputpath = ['~/So2026/NeuralData/output/WhenProjections/'];

save([outputpath,dates,'_taskid',num2str(taskid),'_WhenProjection_FAST.mat'], ...
    'S_d','S_s','rt_all','rt_group','whenCD','tvec_d','tvec_s', ...
    'useNorm','choice_all','correct_all','dir_all','coh_all', 'orig_trial_idx');

saveas(gcf,[outputpath,'figs/',dates,'_taskid',num2str(taskid), '_FAST.fig']);
toc

end

