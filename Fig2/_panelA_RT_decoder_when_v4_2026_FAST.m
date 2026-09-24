function RT_decoder_when_v4_2026_FAST(dates,taskid)
% RT_decoder_when_v4_2026_FAST
%
% Trains a "When" decoder: an L1-regularized (lassoglm) logistic
% regression that predicts, from population spike counts in a 25 ms bin,
% whether the upcoming saccade will land within the next 150 ms (a
% binary "about to terminate" label built from each trial's own RT).
% Time bins are pooled across all trials that survive that long (later
% bins include only trials with long enough RT), so the decoder is
% trained jointly across time and trials rather than at one fixed
% latency.
%
% Trials are split into train/test halves by alternating trial index
% (even/odd). The resulting weight vector (WhenCD, "When coding
% direction") is reported both raw and norm-normalized; AUC, balanced
% accuracy, and F1 are reported on the held-out test split. This
% decoder (WhenCD) is the input consumed by RT_project2_WhenCD_FAST.m,
% which projects it onto held-out trial data to produce the
% time-varying "When projection."
%
% INPUTS
%   dates  : session identifier string (e.g. 'N210513'), used to load
%            [path dates '.mat'] (must contain trials, Neuron, Session)
%            and to pick the monkey-specific RT cutoff (dates(1)=='H' ->
%            550 ms; dates(1)=='N'||'D' -> 400 ms) used to exclude
%            trials too fast to supply enough time bins.
%   taskid : scalar taskid; only trials with trials(j).taskid == taskid
%            (exact match) and trials(j).response >= 0 and RT >=
%            RTcutoff are included.
%
% OUTPUT
%   None returned; saves coef_when(_all), B(_all), whenCD(_norm),
%   AUC(_all), balanced_acc(_all), F1, and taskid to [outputpath dates
%   '_WhenCD_taskid_' taskid '_v4_FAST.mat'].
%
% EXAMPLE
%   RT_decoder_when_v4_2026_FAST('N210513', 21)

path = ['~/So2026/NeuralData/session_mat/'];
outputpath = ['~/So2026/NeuralData/output/WhenCD/'];

if dates(1)=='H'
    RTcutoff = 550;
elseif dates(1)=='N'||dates(1)=='D'
    RTcutoff = 400;
end

binwidth = 25;
lambda = 0.01;

load([path,dates,'.mat']);  % loads: trials, Neuron, Session
cellno = Session.n_neurons;

tic

%% -------------------------------
% First pass: trial selection + metadata
%% -------------------------------
trialno = 0;

for j = 1:length(trials)
    if (trials(j).taskid == taskid) && ...
       (trials(j).response>=0) && ...
       ((trials(j).time_sacc(1)-trials(j).time_dots_on(1))*1000 >= RTcutoff)

        trialno = trialno + 1;

        valid_trials(trialno) = j;

        rt_all(trialno,1) = (trials(j).time_sacc(1)-trials(j).time_dots_on(1))*1000;

        dir_sign = (trials(j).dot_dir ~= trials(j).t1_dir);

        if ((dir_sign == 0)&&(trials(j).response == 1))||...
           ((dir_sign==1)&&(trials(j).response == 0))
            ChoiceY_all(trialno,1) = 0;
        else
            ChoiceY_all(trialno,1) = 1;
        end

        endtime_temp = (trials(j).time_sacc(1)*1000 - 50) - trials(j).time_dots_on(1)*1000;
        starttime = -200;

        binno(trialno) = round((endtime_temp-starttime)/binwidth);

        cnt_startt_array{trialno} = starttime:binwidth:starttime+(binno(trialno)-1)*binwidth;

        dots_times(trialno) = trials(j).time_dots_on(1)*1000;
    end
end

nTrials = trialno;
maxBin = max(binno);

%% -------------------------------
% Preallocate 3D matrix: trials × neurons × time
%% -------------------------------
All_cnt = nan(nTrials, cellno, maxBin);

%% -------------------------------
% Main spike binning (vectorized per trial)
%% -------------------------------
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
% Build D and R (preallocated)
%% -------------------------------
totalRows = sum(binno);
D = zeros(totalRows, cellno);
R = zeros(totalRows,1);

D_train = [];
D_test = [];
R_train = [];
R_test = [];

rowIdx = 1;

isTrain_all = mod((1:nTrials)',2)==0;

isTest_all = mod((1:nTrials)',2)==1;  %NS

binT = starttime:binwidth:(maxBin-1)*binwidth;

for tt = 1:length(binT)

    idx_include = (binno >= tt)';
    trials_idx = find(idx_include);

    nInc = length(trials_idx);

    if nInc==0
        continue
    end

    % ----- R -----
    r = zeros(nTrials,1);
    r(rt_all >= binT(tt) & rt_all < (binT(tt)+150)) = 1;

    R_bin = r(idx_include);

    % ----- D -----
    D_temp = squeeze(All_cnt(trials_idx,:,tt));

    % store
    D(rowIdx:rowIdx+nInc-1,:) = D_temp;
    R(rowIdx:rowIdx+nInc-1) = R_bin;

    % train/test
    isTrain = isTrain_all(trials_idx);
    %isTest  = ~isTrain;
    isTest = isTest_all(trials_idx); %NS
    

    D_train = [D_train; D_temp(isTrain,:)];
    D_test  = [D_test;  D_temp(isTest,:)];

    R_train = [R_train; R_bin(isTrain)];
    R_test  = [R_test;  R_bin(isTest)];

    rowIdx = rowIdx + nInc;
end

% trim unused preallocated space
D = D(1:rowIdx-1,:);
R = R(1:rowIdx-1);

%% -------------------------------
% regression
%% -------------------------------
[B_all,FitInfo] = lassoglm(D,R,'binomial','link','logit','lambda',lambda);
coef_when_all = [FitInfo.Intercept; B_all];

probabilities_all = glmval(coef_when_all,D,'logit');
[~,~,~,AUC_all] = perfcurve(R,probabilities_all,1);

disp(['AUROC: ', num2str(AUC_all)]);

PredY_all = probabilities_all>0.5;

TPR = sum((PredY_all == 1) & (R == 1)) / sum(R == 1);
TNR = sum((PredY_all == 0) & (R == 0)) / sum(R == 0);

balanced_acc_all = (TPR + TNR) / 2;
disp(['Balanced Accuracy (all trials): ', num2str(balanced_acc_all)]);

%% -------------------------------
% train/test
%% -------------------------------
[B,FitInfo] = lassoglm(D_train,R_train,'binomial','link','logit','lambda',lambda);
coef_when = [FitInfo.Intercept; B];

% ---- NEW: normalized raw decoder weights ----
whenCD = B;
whenCD_norm = B / (norm(B) + eps);

probabilities = glmval(coef_when,D_test,'logit');
[~,~,~,AUC] = perfcurve(R_test,probabilities,1);

disp(['AUROC (separate train/test): ', num2str(AUC)]);

PredY = probabilities>0.5;

TPR = sum((PredY == 1) & (R_test == 1)) / sum(R_test == 1);
TNR = sum((PredY == 0) & (R_test == 0)) / sum(R_test == 0);

balanced_acc = (TPR + TNR) / 2;
disp(['Balanced Accuracy (separate train/test): ', num2str(balanced_acc)]);

precision = sum((PredY == 1) & (R_test == 1)) / sum(PredY == 1);
recall = TPR;
F1 = 2 * (precision * recall) / (precision + recall);

disp(['F1-score: ', num2str(F1)]);

%% save
save([outputpath, dates,'_WhenCD_taskid_',num2str(taskid),'_v4_FAST.mat'], ...
    'coef_when','coef_when_all', ...
    'B','B_all', ...
    'whenCD','whenCD_norm', ...
    'AUC','AUC_all', ...
    'balanced_acc','balanced_acc_all', ...
    'F1','taskid');

toc

end


