function RT_project2_WhenWhatCD_forCorr(dates, file_when, file_what, taskid, dec_t, choice_t, useNorm)
% RT_project2_WhenWhatCD_forCorr
%
% Projects the "When" coding direction (WhenCD, saved by
% RT_decoder_when_v4_2026_FAST.m under file_when) and two "What" coding
% directions (WhatCD, saved by RT_decoder_what_FAST.m under file_what,
% taken at dots-aligned time dec_t and saccade-aligned time choice_t)
% onto the same held-out population spike counts, producing dots- and
% saccade-aligned time-varying projections for all three CDs
% (S_d_when/S_s_when, S_d_what/S_s_what for the motion-aligned WhatCD,
% S_d_what_sacc/S_s_what_sacc for the saccade-aligned WhatCD). Unlike
% RT_project2_WhenCD_FAST.m / RT_project2_WhatCD_FAST.m, no RT-based
% trial attrition or grouping is applied here (all valid trials are
% kept and saved individually) -- this is the paired per-trial,
% per-timepoint data that RT_WhenWhat_correlation_Fig5_v2.m correlates
% the When and What projections against.
%
% INPUTS
%   dates     : session identifier string (e.g. 'N210513'), used to
%               load [datapath dates '.mat'] (trials, Neuron, Session).
%   file_when : filename (within WhenCD_path) of the .mat saved by
%               RT_decoder_when_v4_2026_FAST.m for this session
%               (contains coef_when, whenCD_norm).
%   file_what : filename (within WhatCD_path) of the .mat saved by
%               RT_decoder_what_FAST.m for this session (contains B_d,
%               B_s, whatCD_d_norm, whatCD_s_norm, searchT1, searchT2).
%   taskid    : scalar taskid to select trials (exact match).
%   dec_t     : dots-aligned time (ms, must be a value in searchT1) at
%               which to take the motion-aligned What decoder weights.
%   choice_t  : saccade-aligned time (ms, must be a value in searchT2)
%               at which to take the saccade-aligned What decoder
%               weights.
%   useNorm   : logical; if true, uses whenCD_norm/whatCD_d_norm/
%               whatCD_s_norm (norm-normalized weights) instead of the
%               raw coef_when(2:end)/B_d/B_s.
%
% OUTPUT
%   None returned; saves S_d_when, S_s_when, S_d_what, S_s_what,
%   S_d_what_sacc, S_s_what_sacc, tvec_d, tvec_s, rt_all, choice_all,
%   correct_all, coh_all, dir_all, taskid, orig_trial_idx, useNorm,
%   dec_t, choice_t to [outpath dates '_taskid' taskid
%   '_WhenWhatProjection_forCorr.mat']. The saved .mat file is the input
%   consumed by RT_WhenWhat_correlation_Fig5_v2.m.
%
% EXAMPLE
%   RT_project2_WhenWhatCD_forCorr('N210513', 'N210513_WhenCD_taskid_21_v4_FAST.mat', 'N210513_WhatCD_taskid21_FAST.mat', 21, 450, -200, false)

%% ===============================
% PATHS
%% ===============================
datapath = ['~/So2026/NeuralData/session_mat/'];
WhenCD_path = ['~/So2026/NeuralData/output/WhenCD/'];
WhatCD_path = ['~/So2026/NeuralData/output/WhatCD/'];
outpath = ['~/So2026/NeuralData/output/WhenWhatProjections/'];


tic
%% ===============================
% LOAD DECODERS
%% ===============================
load([WhenCD_path file_when])   % contains coef_when, whenCD_norm
load([WhatCD_path file_what])   % contains B_d, B_s, searchT1, searchT2

% ---- When weights ----
if useNorm
    w_when = whenCD_norm(:);
else
    w_when = coef_when(2:end);
end

% ---- What weights ----
if useNorm
    w_what_d = whatCD_d_norm{searchT1==dec_t}(:);
    w_what_s = whatCD_s_norm{searchT2==choice_t}(:);
else
    w_what_d = B_d{searchT1==dec_t}(:);
    w_what_s = B_s{searchT2==choice_t}(:);
end

%% ===============================
% LOAD DATA
%% ===============================
load([datapath dates '.mat']);  % trials, Neuron, Session
cellno = Session.n_neurons;

%% ===============================
% PARAMETERS (UNIFIED)
%% ===============================
binwidth = 25;

pre_dots = -200;
%post_dots = 650;
post_dots = 1200;

pre_sacc = -650;
post_sacc = 350;

tvec_d = pre_dots:binwidth:post_dots;
tvec_s = pre_sacc:binwidth:post_sacc;

nT_d = length(tvec_d);
nT_s = length(tvec_s);

%% ===============================
% TRIAL SELECTION (NO ATTRITION)
%% ===============================
trialno = 0;

for j = 1:length(trials)

    if (trials(j).taskid == taskid) && (trials(j).response >= 0)

        trialno = trialno + 1;

        valid_trials(trialno) = j;
        orig_trial_idx(trialno,1) = j;

        % ---- times ----
        dots_times(trialno) = trials(j).time_dots_on(1)*1000;
        sacc_times(trialno) = trials(j).time_sacc(1)*1000;

        % ---- RT ----
        rt_all(trialno,1) = (trials(j).time_sacc(1)-trials(j).time_dots_on(1))*1000;

        % ---- choice ----
        dir_sign = (trials(j).dot_dir ~= trials(j).t1_dir);

        if ((dir_sign==0)&&(trials(j).response==1)) || ...
           ((dir_sign==1)&&(trials(j).response==0))
            choice_all(trialno,1) = 0;
        else
            choice_all(trialno,1) = 1;
        end

        % ---- correctness ----
        correct_all(trialno,1) = trials(j).response;

        % ---- signed coherence ----
        coh = trials(j).dot_coh*(1-2*dir_sign);
        if coh==0
            coh = (choice_all(trialno)==0)*2 - 1;
        end
        coh_all(trialno,1) = coh;

        dir_all(trialno,1) = trials(j).dot_dir;

    end
end

nTrials = trialno;

%% ===============================
% BUILD SPIKE MATRICES
%% ===============================
All_cnt_d = nan(nTrials, cellno, nT_d);
All_cnt_s = nan(nTrials, cellno, nT_s);

edges_d = [tvec_d, tvec_d(end)+binwidth];
edges_s = [tvec_s, tvec_s(end)+binwidth];

for i = 1:cellno
    spikes = Neuron{i}.spikes;

    for t = 1:nTrials
        j = valid_trials(t);

        spk = spikes{j} * 1000;

        % dots
        All_cnt_d(t,i,:) = histcounts(spk - dots_times(t), edges_d);

        % sacc
        All_cnt_s(t,i,:) = histcounts(spk - sacc_times(t), edges_s);
    end
end

%% ===============================
% PROJECTION (VECTORIAL)
%% ===============================
X_d = reshape(permute(All_cnt_d,[3 1 2]), [], cellno);   % time × trial × neuron, then 2d --> (timextrial) x neuron 
X_s = reshape(permute(All_cnt_s,[3 1 2]), [], cellno);

% ---- When ----
S_d_when = reshape(X_d * w_when, nT_d, nTrials);
S_s_when = reshape(X_s * w_when, nT_s, nTrials);

% ---- What (motion-aligned) ----
S_d_what = reshape(X_d * w_what_d, nT_d, nTrials);
S_s_what = reshape(X_s * w_what_d, nT_s, nTrials);

% ---- What (sacc-aligned) ----
S_d_what_sacc = reshape(X_d * w_what_s, nT_d, nTrials);
S_s_what_sacc = reshape(X_s * w_what_s, nT_s, nTrials);

%% ===============================
% SAVE (CRITICAL: FULL INFO)
%% ===============================
coh_all = coh_all(:);
choice_all = choice_all(:);
correct_all = correct_all(:);
rt_all = rt_all(:);
save([outpath dates '_taskid' num2str(taskid) '_WhenWhatProjection_forCorr.mat'], ...
    'S_d_when','S_s_when', ...
    'S_d_what','S_s_what', ...
    'S_d_what_sacc','S_s_what_sacc', ...
    'tvec_d','tvec_s', ...
    'rt_all','choice_all','correct_all','coh_all','dir_all', 'taskid', ...
    'orig_trial_idx','useNorm','dec_t','choice_t');

toc
end