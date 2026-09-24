function RT_project2_WhatCD_FAST(dates,file1,taskid, dec_t, choice_t, useNorm)
% RT_project2_WhatCD_FAST
%
% Projects two fixed-time-window "What" (choice) coding directions --
% one trained dots-aligned at dec_t ms (Dec_coeff, from
% RT_decoder_what_FAST.m's whatCD_d/B_d), one trained saccade-aligned at
% choice_t ms (Choice_coeff, from whatCD_s/B_s) -- onto held-out
% population spike counts from the same session, producing time-varying
% "What projections" S_d_dec/S_s_dec (dots-trained CD, dots/saccade-
% aligned) and S_d_choice/S_s_choice (saccade-trained CD, dots/saccade-
% aligned). Trials are grouped by signed coherence and the mean
% projection per coherence (all trials, and correct trials only) is
% plotted for each CD, dots- and saccade-aligned (2 figures, 2x2 panels
% each).
%
% INPUTS
%   dates     : session identifier string (e.g. 'N210513'), used to load
%               [datapath dates '.mat'] (trials, Neuron, Session).
%   file1     : filename (within CDpath) of the .mat saved by
%               RT_decoder_what_FAST.m for this session (contains
%               B_d, B_s, whatCD_d(_norm), whatCD_s(_norm), searchT1,
%               searchT2).
%   taskid    : scalar taskid to select trials (exact match).
%   dec_t     : dots-aligned time (ms, must be a value in searchT1) at
%               which to take the dots-trained decoder weights.
%   choice_t  : saccade-aligned time (ms, must be a value in searchT2)
%               at which to take the saccade-trained decoder weights.
%   useNorm   : logical; if true, uses whatCD_d_norm/whatCD_s_norm
%               (norm-normalized weights) instead of the raw B_d/B_s.
%
% OUTPUT
%   None returned; saves S_d_dec, S_s_dec, S_d_choice, S_s_choice,
%   tvec_d, tvec_s, useNorm, taskid, dec_t, choice_t, coh_all, choice,
%   correct, rt to [outpath dates '_taskid' taskid
%   '_WhatProjection_FAST.mat'], plus the two summary figures to
%   [outpath 'figs/' dates '_taskid' taskid
%   '_WhatProjection_(Dots|Sacc)Window_FAST.fig']. The saved .mat file
%   is the input consumed by RT_project2_WhatCD_acrossSessions_FAST.m
%   and RT_project2_WhatCD_buildup_bySession_withTrace_v2.m.
%
% REQUIRES ON PATH: fig_setting.m (included alongside this file).
%
% EXAMPLE
%   RT_project2_WhatCD_FAST('N210513', 'N210513_WhatCD_taskid21_FAST.mat', 21, 450, -200, false)

%% paths
datapath = ['~/So2026/NeuralData/session_mat/'];
CDpath   = ['~/So2026/NeuralData/output/WhatCD/'];
outpath  = ['~/So2026/NeuralData/output/WhatProjections/'];

%% load decoder
load([CDpath,file1])

if useNorm
    Dec_coeff    = whatCD_d_norm{searchT1==dec_t};
    Choice_coeff = whatCD_s_norm{searchT2==choice_t};
else
    Dec_coeff    = B_d{searchT1==dec_t};
    Choice_coeff = B_s{searchT2==choice_t};
end

Dec_coeff    = Dec_coeff(:);
Choice_coeff = Choice_coeff(:);

%% load session
load([datapath,dates,'.mat']);  % trials, Neuron, Session
cellno = Session.n_neurons;

%% params
binwidth = 25;

pre_dots = -100; post_dots = 650;
pre_sacc = -550; post_sacc = 200;

tvec_d = pre_dots:binwidth:post_dots;
tvec_s = pre_sacc:binwidth:post_sacc;

nT_d = length(tvec_d);
nT_s = length(tvec_s);

coh_set = [-400 -200 -100 -50 -25 -1 1 25 50 100 200 400];

tic

%% -------------------------------
% Trial selection
%% -------------------------------
trialno = 0;

for j = 1:length(trials)

    if (trials(j).taskid == taskid) && (trials(j).response>=0)

        trialno = trialno + 1;

        valid_trials(trialno) = j;

        % times
        dots_times(trialno) = trials(j).time_dots_on(1)*1000;
        sacc_times(trialno) = trials(j).time_sacc(1)*1000;

        % RT
        rt(trialno) = (trials(j).time_sacc(1)-trials(j).time_dots_on(1))*1000;

        % choice
        dir_sign = (trials(j).dot_dir ~= trials(j).t1_dir);  % 0 for up/right, 1 for down/left

        if ((dir_sign==0)&&(trials(j).response==1)) || ...
           ((dir_sign==1)&&(trials(j).response==0))
            choice(trialno) = 0;        % up/right
        else
            choice(trialno) = 1;        % down/left
        end

        correct(trialno) = trials(j).response;

        % signed coherence
        coh = trials(j).dot_coh*(1-2*dir_sign);

        if coh==0
            coh = (choice(trialno)==0)*2 - 1; % ±1
        end

        coh_all(trialno) = coh;
    end
end

nTrials = trialno;

%% -------------------------------
% Build spike tensors
%% -------------------------------
All_cnt_d = nan(nTrials, cellno, nT_d);
All_cnt_s = nan(nTrials, cellno, nT_s);

edges_d = [tvec_d, tvec_d(end)+binwidth];
edges_s = [tvec_s, tvec_s(end)+binwidth];

for i = 1:cellno
    spikes = Neuron{i}.spikes;

    for t = 1:nTrials
        j = valid_trials(t);

        spk = spikes{j} * 1000;

        % dots aligned
        st = spk - dots_times(t);
        All_cnt_d(t,i,:) = histcounts(st, edges_d);

        % sacc aligned
        st = spk - sacc_times(t);
        All_cnt_s(t,i,:) = histcounts(st, edges_s);
    end
end

%% -------------------------------
% Projection (vectorized)
%% -------------------------------

% dots
X_d = reshape(permute(All_cnt_d,[3 1 2]), [], cellno);
S_d_dec = reshape(X_d * Dec_coeff, nT_d, nTrials);
S_d_choice = reshape(X_d * Choice_coeff, nT_d, nTrials);

% sacc
X_s = reshape(permute(All_cnt_s,[3 1 2]), [], cellno);
S_s_dec = reshape(X_s * Dec_coeff, nT_s, nTrials);
S_s_choice = reshape(X_s * Choice_coeff, nT_s, nTrials);

%% -------------------------------
% Coherence grouping
%% -------------------------------
coh_idx = nan(nTrials,1);
% --- sanity check ---
assert(size(S_d_dec,2) == length(coh_idx), 'Mismatch: S_d_dec vs coh_idx');
assert(size(S_d_dec,2) == length(correct), 'Mismatch: S_d_dec vs correct');
assert(length(Dec_coeff) == cellno, 'Coefficient size mismatch');

% --- force column vectors ---
coh_idx = coh_idx(:);
correct = correct(:);

for k = 1:length(coh_set)
    coh_idx(coh_all == coh_set(k)) = k;
end

%% -------------------------------
% Compute means (ALL vs CORRECT)
%% -------------------------------
nC = length(coh_set);
mean_d_dec = nan(nT_d,nC,2);
mean_s_dec = nan(nT_s,nC,2);
mean_d_choice = nan(nT_d,nC,2);
mean_s_choice = nan(nT_s,nC,2);

for k = 1:length(coh_set)

    idx_all = (coh_idx == k);
    idx_corr = idx_all & (correct(:)==1);
    
    %size(idx_all)
    %size(idx_corr)

    mean_d_dec(:,k,1) = nanmean(S_d_dec(:,idx_all),2);
    mean_s_dec(:,k,1) = nanmean(S_s_dec(:,idx_all),2);

    mean_d_dec(:,k,2) = nanmean(S_d_dec(:,idx_corr),2);
    mean_s_dec(:,k,2) = nanmean(S_s_dec(:,idx_corr),2);

    mean_d_choice(:,k,1) = nanmean(S_d_choice(:,idx_all),2);
    mean_s_choice(:,k,1) = nanmean(S_s_choice(:,idx_all),2);

    mean_d_choice(:,k,2) = nanmean(S_d_choice(:,idx_corr),2);
    mean_s_choice(:,k,2) = nanmean(S_s_choice(:,idx_corr),2);
end

%% -------------------------------
% Plot
%% -------------------------------
%ColorMapCoh = jet(length(coh_set));
ColorMapCoh = [0 0.5 0; 0 0.7 0.3; 0.4 0.8 0.6; 0.4 0.8 0.6; 0.6 0.9 0.8; 0.8 1 0.8; ...
                1 1 0.8 ;1 1 0.6; 1 0.9 0.5; 1 0.8 0.4; 1 0.6 0; 1 0 0];


outputpath = ['~/So2026/NeuralData/output/WhatProjections/'];

figure(1); clf;
ymin = min([mean_d_dec(:); mean_s_dec(:)], [], 'omitnan')-0.1;
ymax = max([mean_d_dec(:); mean_s_dec(:)], [], 'omitnan')+0.1;

for row = 1:2  % all vs correct

    % DECODER (dots-trained)
    subplot(2,2,(row-1)*2+1); hold on;
    line([0 0],[ymin ymax],'Color','k');
    for k = 1:length(coh_set)
        plot(tvec_d, mean_d_dec(:,k,row),'Color',ColorMapCoh(k,:), 'LineWidth',1.5);
    end
    title(['WhatCD trained using ',num2str(dec_t), 'ms from motion; (' num2str(row==2) ' correct)']);
    ylim([ymin ymax]);
    xlim([pre_dots post_dots]);
    fig_setting();

    % DECODER (sacc)
    subplot(2,2,(row-1)*2+2); hold on;
    line([0 0],[ymin ymax],'Color','k');
    for k = 1:length(coh_set)
        plot(tvec_s, mean_s_dec(:,k,row),'Color',ColorMapCoh(k,:), 'LineWidth',1.5);
    end
    ylim([ymin ymax]);
    xlim([pre_sacc post_sacc]);
    fig_setting();

end

sgtitle([dates ' What projection; taskid = ' num2str(taskid)]);
saveas(gcf,[outputpath,'figs/',dates,'_taskid',num2str(taskid), '_WhatProjection_DotsWindow_FAST.fig']);


figure(2); clf;
ymin = min([mean_d_choice(:); mean_s_choice(:)], [], 'omitnan')-0.1;
ymax = max([mean_d_choice(:); mean_s_choice(:)], [], 'omitnan')+0.1;

for row = 1:2  % all vs correct

    % DECODER (sacc-trained)
    subplot(2,2,(row-1)*2+1); hold on;
    line([0 0],[ymin ymax],'Color','k');
    for k = 1:length(coh_set)
        plot(tvec_d, mean_d_choice(:,k,row),'Color',ColorMapCoh(k,:), 'LineWidth',1.5);
    end
     title(['WhatCD trained using ',num2str(choice_t), 'ms from 1st sacc; (' num2str(row==2) ' correct)']);
    ylim([ymin ymax]);
    xlim([pre_dots post_dots]);
    fig_setting();


    % DECODER (sacc)
    subplot(2,2,(row-1)*2+2); hold on;
    line([0 0],[ymin ymax],'Color','k');
    for k = 1:length(coh_set)
        plot(tvec_s, mean_s_choice(:,k,row),'Color',ColorMapCoh(k,:), 'LineWidth',1.5);
    end
    ylim([ymin ymax]);
    xlim([pre_sacc post_sacc]);
    fig_setting();

end

sgtitle([dates ' What projection; taskid = ' num2str(taskid)]);
saveas(gcf,[outputpath,'figs/',dates,'_taskid',num2str(taskid), '_WhatProjection_SaccWindow_FAST.fig']);


%% -------------------------------
% Save
%% -------------------------------

outpath = ['~/So2026/NeuralData/output/WhatProjections/'];

save([outpath,dates,'_taskid',num2str(taskid),'_WhatProjection_FAST.mat'], ...
    'S_d_dec','S_s_dec','S_d_choice','S_s_choice', ...
    'tvec_d','tvec_s', 'useNorm', ...
    'taskid','dec_t', 'choice_t', 'coh_all','choice','correct','rt');

toc
end
