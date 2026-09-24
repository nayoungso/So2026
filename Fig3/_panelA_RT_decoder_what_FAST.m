function RT_decoder_what_FAST(dates,taskid)
% RT_decoder_what_FAST
%
% Trains a "What" (choice) decoder at each of several fixed time
% windows, both dots-aligned (searchT1 = 200:50:600 ms) and
% saccade-aligned (searchT2 = -500:50:0 ms): at each window, an
% L1-regularized (lassoglm) logistic regression predicts trial choice
% from population spike counts in a 100 ms bin centered on that window.
% Trials are split into train/test halves by alternating trial index
% (even/odd).
%
% For every pair of (train window, test window) -- within and across
% the dots/saccade alignments -- the held-out decoding accuracy is also
% computed (accuracy_dxd, accuracy_dxs, accuracy_sxd, accuracy_sxs),
% producing the 4-panel cross-temporal generalization figure. This
% cross-temporal accuracy is the input consumed by
% RT_decoder_what_crossAccuracy_acrossSession_FAST.m, which averages it
% across sessions. The per-window decoder weights (whatCD_d, whatCD_s,
% and their norm-normalized versions) are the input consumed by
% RT_project2_WhatCD_FAST.m, which projects them onto held-out trial
% data to produce the time-varying "What projection."
%
% INPUTS
%   dates  : session identifier string (e.g. 'N210513'), used to load
%            [datapath dates '.mat'] (must contain trials, Neuron,
%            Session) and to pick the monkey-specific RT cutoff
%            (dates(1)=='H' -> 550 ms; otherwise -> 400 ms) used to
%            exclude trials too fast for the saccade-aligned windows.
%   taskid : scalar taskid; only trials with trials(j).taskid == taskid
%            (exact match) and trials(j).response >= 0 and RT >=
%            RTcutoff are included.
%
% OUTPUT
%   None returned; saves B_d, B_s, whatCD_d(_norm), whatCD_s(_norm),
%   searchT1, searchT2, accuracy_d, accuracy_s, accuracy_dxd,
%   accuracy_dxs, accuracy_sxd, accuracy_sxs to [outpath dates
%   '_WhatCD_taskid' taskid '_FAST.mat'], plus the cross-temporal
%   accuracy figure to [outpath 'figs/' dates '_WhatCD_taskid' taskid
%   '_FAST.fig'].
%
% EXAMPLE
%   RT_decoder_what_FAST('N210513', 21)

%% paths
datapath = ['~/So2026/NeuralData/session_mat/'];
outpath  = ['~/So2026/NeuralData/output/WhatCD/'];

%% params
searchT1 = 200:50:600;   % dots aligned
searchT2 = -500:50:0;    % sacc aligned
binwidth = 100;
lambda = 0.01;

if dates(1)=='H'
    RTcutoff = 550;
else
    RTcutoff = 400;
end

%% load session
load([datapath,dates,'.mat']);  % trials, Neuron, Session
cellno = Session.n_neurons;

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

        % choice
        dir_sign = (trials(j).dot_dir ~= trials(j).t1_dir); % 0 for up/right 1 for down/left
        
        if ((dir_sign==0)&&(trials(j).response==1)) || ...
           ((dir_sign==1)&&(trials(j).response==0))
            ChoiceY_all(trialno,1) = 0;  % right/up
        else
            ChoiceY_all(trialno,1) = 1;  % left/down
        end

        dots_times(trialno) = trials(j).time_dots_on(1)*1000;
        sacc_times(trialno) = trials(j).time_sacc(1)*1000;
    end
end

nTrials = trialno;

%% -------------------------------
% Precompute spike times
%% -------------------------------
% store aligned spike times once
for i = 1:cellno
    spikes = Neuron{i}.spikes;

    for t = 1:nTrials
        j = valid_trials(t);

        spike_ts = spikes{j} * 1000;

        spk_d{i,t} = spike_ts - dots_times(t);
        spk_s{i,t} = spike_ts - sacc_times(t);
    end
end

%% -------------------------------
% Build count matrices
%% -------------------------------
nT1 = length(searchT1);
nT2 = length(searchT2);

All_cnt_d = zeros(nTrials, cellno, nT1);
All_cnt_s = zeros(nTrials, cellno, nT2);

for k = 1:nT1
    t0 = searchT1(k);

    edges = [t0-binwidth/2, t0+binwidth/2];

    for i = 1:cellno
        for t = 1:nTrials
            All_cnt_d(t,i,k) = sum(spk_d{i,t} >= edges(1) & spk_d{i,t} < edges(2));
        end
    end
end

for k = 1:nT2
    t0 = searchT2(k);

    edges = [t0-binwidth/2, t0+binwidth/2];

    for i = 1:cellno
        for t = 1:nTrials
            All_cnt_s(t,i,k) = sum(spk_s{i,t} >= edges(1) & spk_s{i,t} < edges(2));
        end
    end
end

%% -------------------------------
% Train/test split
%% -------------------------------
isTrain = mod((1:nTrials)',2)==0;
isTest  = ~isTrain;

Train_Y = ChoiceY_all(isTrain);
Test_Y  = ChoiceY_all(isTest);

%% -------------------------------
% Train decoders
%% -------------------------------
for k = 1:nT1

    Dtr = squeeze(All_cnt_d(isTrain,:,k));
    Dte = squeeze(All_cnt_d(isTest,:,k));

    [B_d{k},FitInfo] = lassoglm(Dtr,Train_Y,'binomial','lambda',lambda);
    Intercept_d{k} = FitInfo.Intercept;
    
    prob = glmval([Intercept_d{k};B_d{k}],Dte,'logit');
    accuracy_d(k) = mean((prob>0.5)==Test_Y);

    % ---- normalized raw weights ----
    w_dec = B_d{k};
    whatCD_d{k} = w_dec;
    whatCD_d_norm{k} = w_dec / (norm(w_dec) + eps);

end

for k = 1:nT2

    Dtr = squeeze(All_cnt_s(isTrain,:,k));
    Dte = squeeze(All_cnt_s(isTest,:,k));

    [B_s{k},FitInfo] = lassoglm(Dtr,Train_Y,'binomial','lambda',lambda);
    Intercept_s{k} = FitInfo.Intercept;
    
    prob = glmval([Intercept_s{k};B_s{k}],Dte,'logit');
    accuracy_s(k) = mean((prob>0.5)==Test_Y);

    % ---- normalized raw weights ----
    w_choice = B_s{k};
    whatCD_s{k} = w_choice;
    whatCD_s_norm{k} = w_choice / (norm(w_choice) + eps);

end

%% -------------------------------
% Cross accuracy
%% -------------------------------
for k = 1:nT1

    coef = [Intercept_d{k}; B_d{k}];

    for kk = 1:nT1
        Dte = squeeze(All_cnt_d(isTest,:,kk));
        prob = glmval(coef,Dte,'logit');
        accuracy_dxd(k,kk) = mean((prob>0.5)==Test_Y);
    end

    for kk = 1:nT2
        Dte = squeeze(All_cnt_s(isTest,:,kk));
        prob = glmval(coef,Dte,'logit');
        accuracy_dxs(k,kk) = mean((prob>0.5)==Test_Y);
    end
end

for k = 1:nT2

    coef = [Intercept_s{k}; B_s{k}];

    for kk = 1:nT1
        Dte = squeeze(All_cnt_d(isTest,:,kk));
        prob = glmval(coef,Dte,'logit');
        accuracy_sxd(k,kk) = mean((prob>0.5)==Test_Y);
    end

    for kk = 1:nT2
        Dte = squeeze(All_cnt_s(isTest,:,kk));
        prob = glmval(coef,Dte,'logit');
        accuracy_sxs(k,kk) = mean((prob>0.5)==Test_Y);
    end
end



%% -------------------------------
% Plot
%% -------------------------------
figure(15); clf;

subplot(221)
imagesc(searchT1,searchT1,accuracy_dxd); clim([0.5 1]);
xlabel('Tested on time window, from motion on (ms)');
ylabel('Trained on time window, from motion on (ms)');

subplot(222)
imagesc(searchT2,searchT1,accuracy_dxs); clim([0.5 1]);
xlabel('Tested on time window, from sacc (ms)');

subplot(223)
imagesc(searchT1,searchT2,accuracy_sxd); clim([0.5 1]);
xlabel('Tested on time window, from motion on (ms)');
ylabel('Trained on time window, from sacc (ms)');

subplot(224)
imagesc(searchT2,searchT2,accuracy_sxs); clim([0.5 1]);
xlabel('Tested on time window, from sacc (ms)');

sgtitle(['What decoder dynamics ',dates, '  taskid = ', num2str(taskid)]);

%% save
save([outpath,dates,'_WhatCD_taskid', num2str(taskid),'_FAST.mat'], ...
    'B_d','B_s','whatCD_d','whatCD_s', ...
    'whatCD_d_norm','whatCD_s_norm', ...
    'searchT1','searchT2', ...
    'accuracy_d','accuracy_s', ...
    'accuracy_dxd','accuracy_dxs','accuracy_sxd','accuracy_sxs');

saveas(gcf,[outpath,'figs/',dates,'_WhatCD_taskid', num2str(taskid),'_FAST.fig']);

toc
end

