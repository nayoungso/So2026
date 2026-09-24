function results = RT_behav_compareModels_v2_AICBICbar(date, taskidList)
% RT_behav_compareModels_v2_AICBICbar
%
% Robustness variant of RT_behav_compareModels_v1_AICBICbar.m: fits the
% same three competing accounts of decision termination, to the same
% psychometric + chronometric summary data, but gives Models 1 and 2 the
% SAME asymmetric non-decision-time (tnd_up/tnd_dn) flexibility as
% Model 3, removing the "Model 3 only wins because it had an extra free
% parameter the others lacked" objection. Both AIC and BIC are reported
% for all three models, so a conclusion that holds under both criteria
% is more robust than one resting on AIC alone.
%
%   Model 1 -- imposed deadline (calcModel1_deadline_asymTnd.m):
%     evidence accumulates, but termination time is drawn independently
%     of the accumulated evidence. theta = [k, b, mu_T, sigma_T, tnd_up,
%     tnd_dn], 6 params.
%   Model 2 -- parallel/independent magnitude-driven timing
%     (calcModel2_magnitudeTiming_asymTnd.m): a separate accumulator,
%     driven by |coherence| with a collapsing bound, sets termination
%     time; a fully independent signed accumulator determines choice,
%     read out at that time. theta = [k_choice, b, k_time, B_time,
%     Balpha_time, Bbeta_time, tnd_up, tnd_dn], 8 params.
%   Model 3 -- standard drift-diffusion model (DDM) with a collapsing
%     bound and asymmetric non-decision time
%     (calcModel3_collapseBound_asymTnd.m): the SAME evidence stream
%     governs both termination and choice. theta = [k, b, B, Balpha,
%     Bbeta, tnd_up, tnd_dn], 7 params.
%
% All three models are fit with the SAME cost function
% (fitJointCost_generic.m: weighted least-squares on mean RT + binomial
% log-likelihood on choice), so their final negative log-likelihoods are
% directly comparable via AIC = 2*nParams + 2*nlogl and
% BIC = nParams*log(n) + 2*nlogl. BIC charges log(n) points per
% parameter (n = number of data points, see below), which grows with
% sample size, so it penalizes Model 2's extra parameter (8 vs. Model
% 3's 7) more harshly than AIC does -- if Model 3 wins under BOTH
% criteria, that's a materially more robust conclusion than winning
% under AIC alone. The summary figure's third panel plots a grouped bar
% graph of AIC and BIC for all three models side by side (x axis =
% model, y axis = information criterion value).
%
% BIC's n: this cost function is fit to per-coherence SUMMARY statistics
% (mean RT with SE, and binomial choice proportions), not raw per-trial
% RTs, so "n" isn't simply the total trial count. Each valid (non-NaN)
% T1-RT and T2-RT summary point is one data point in the likelihood (a
% Normal-distributed estimate of the mean), while the binomial choice
% term at each coherence already properly incorporates that coherence's
% real trial count inside the likelihood itself. n here = (# valid T1 RT
% points) + (# valid T2 RT points) + (# coherence levels, for the choice
% term) -- computed once from the data's own missingness pattern,
% identical across all three models since it only depends on the data,
% not on theta.
%
% REQUIRES ON PATH: fitJointCost_generic.m, calcModel1_deadline_asymTnd.m,
% calcModel2_magnitudeTiming_asymTnd.m,
% calcModel3_collapseBound_asymTnd.m, spectral_dtb.m, fminsearchbnd.m,
% and fig_setting.m (all included alongside this file).
%
% INPUTS
%   date       : monkey/session identifier string. Pass just the monkey
%                letter (e.g. 'H', 'N') to POOL ACROSS ALL DATES for that
%                monkey -- date is used as a dir() wildcard prefix
%                (dir([path date '*behav*.mat'])), so every session file
%                starting with that letter gets concatenated into one
%                trials array before fitting. Pass a full session string
%                (e.g. 'N210513') to restrict to a single session
%                instead. Run this function once per monkey (separately)
%                to keep monkeys from being pooled together.
%   taskidList : vector of taskid values to include, matched by EXACT
%                membership (ismember(trials(j).taskid, taskidList)),
%                e.g. [20 21].
%
% OUTPUT
%   results : struct with fields .model1, .model2, .model3, each
%             containing .theta, .nlogl, .nParams, .AIC, .BIC,
%             .t1_pred/.t2_pred/.p_pred, plus .data1, .coh_set, .nBIC
%             (the n used for all three BIC calculations).
%
% NOTES
%   - fminsearchbnd is used (not plain fminsearch) since several
%     parameters are positivity-constrained.
%   - Models 1/2's drift parameters are on the library's kappa~15-ish
%     scale (seconds-based spectral_dtb.m); don't compare them directly
%     to a ms-based drift-rate scale from elsewhere in the pipeline.
%
% EXAMPLE
%   results = RT_behav_compareModels_v2_AICBICbar('N', [20 21])
%   results = RT_behav_compareModels_v2_AICBICbar('H', [20 21])

%% -------------------------------
% Load data
%% -------------------------------
path = ['~/So2026/BehaviorOnly/'];
filename = dir(strcat(path,date,'*behav*.mat'));

trialno = 0;
for i = 1:length(filename)
    clear trials
    load([path,filename(i).name]);

    for j = 1:length(trials)
        if ismember(trials(j).taskid, taskidList) && (trials(j).response>=0)
            trialno = trialno+1;
            trials_temp(trialno) = trials(j);
        end
    end
end

clear trials
trials = trials_temp;
clear trials_temp
fprintf('%d trials loaded for %s, taskid in %s (pooled across %d session file(s))\n', ...
    length(trials), date, mat2str(taskidList), length(filename));

pos_coh_set = unique([trials(:).dot_coh]);
pos_coh_set = pos_coh_set/1000;
neg_coh_set = sort(pos_coh_set,'descend').*-1;
coh_set = unique([neg_coh_set pos_coh_set]);

for i = 1:length(trials)
    dir_sign(i) = 1-floor(mod(trials(i).dot_dir,360)/180)*2;
    coh(i) = trials(i).dot_coh * dir_sign(i)/1000;
    RT(i) = trials(i).time_sacc(1) - trials(i).time_dots_on(1);
end

corr_ind = logical([trials(:).response] == 1);
err_ind  = logical([trials(:).response] == 0);

extT1 = logical(dir_sign == 1);
extT2 = logical(dir_sign == -1);

T1_ind = logical( (extT1 & corr_ind) | (extT2 & err_ind) );
T2_ind = logical( (extT1 & err_ind)  | (extT2 & corr_ind) );

for i = 1:length(coh_set)

    coh1_ind = logical((coh==coh_set(i)) & T1_ind);
    coh2_ind = logical((coh==coh_set(i)) & T2_ind);

    coh1_corr_ind = logical(coh1_ind & corr_ind);
    coh2_corr_ind = logical(coh2_ind & corr_ind);

    if coh_set(i) == 0
        RT1_mean(i) = nanmean(RT(coh1_ind));
        RT1_se(i)   = nanstd(RT(coh1_ind))/sqrt(sum(coh1_ind));
        RT2_mean(i) = nanmean(RT(coh2_ind));
        RT2_se(i)   = nanstd(RT(coh2_ind))/sqrt(sum(coh2_ind));
    else
        RT1_mean(i) = nanmean(RT(coh1_corr_ind));
        RT1_se(i)   = nanstd(RT(coh1_corr_ind))/sqrt(sum(coh1_corr_ind));
        RT2_mean(i) = nanmean(RT(coh2_corr_ind));
        RT2_se(i)   = nanstd(RT(coh2_corr_ind))/sqrt(sum(coh2_corr_ind));
    end

    nT1_corr(i) = sum(coh1_corr_ind);
    n_corr(i)   = sum(coh1_corr_ind) + sum(coh2_corr_ind);
    nT1(i)      = sum(coh1_ind);
    ntotal(i)   = sum(coh1_ind) + sum(coh2_ind);

    if nT1_corr(i) < 2
        RT1_mean(i) = NaN;
        RT1_se(i)   = NaN;
    elseif n_corr(i) - nT1_corr(i) < 2
        RT2_mean(i) = NaN;
        RT2_se(i)   = NaN;
    end
end

data1 = [coh_set' RT1_mean' RT1_se' RT2_mean' RT2_se' nT1' ntotal'];

% ---- n for BIC: # valid T1-RT points + # valid T2-RT points + # coherence
% levels (choice term) -- same for all three models, since it only
% depends on data1's own missingness pattern, not on theta ----
nBIC = sum(~isnan(data1(:,3))) + sum(~isnan(data1(:,5))) + size(data1,1);
fprintf('BIC sample size (n) = %d (T1 RT points + T2 RT points + coherence levels)\n', nBIC);

%% -------------------------------
% Fit options
%% -------------------------------
opts = optimset('fminsearch');
opts = optimset(opts,'MaxFunEvals',10.^5,'MaxIter',10.^5,'Display','iter');

%% =================================================================
% Model 1: imposed deadline, asymmetric tnd
% theta = [k, b, mu_T, sigma_T, tnd_up, tnd_dn]
%% =================================================================
theta0_m1 = [0.3   0   300   120   400   450];
lo_m1     = [0    -2    10     5    100   100];
hi_m1     = [5     2   2000  1000   800   800];

[theta_m1, nlogl_m1] = fminsearchbnd( ...
    @(th) fitJointCost_generic(th, data1, @calcModel1_deadline_asymTnd), ...
    theta0_m1, lo_m1, hi_m1, opts);

nParams_m1 = numel(theta_m1);
AIC_m1 = 2*nParams_m1 + 2*nlogl_m1;
BIC_m1 = nParams_m1*log(nBIC) + 2*nlogl_m1;

[t1_m1, t2_m1, p_m1] = calcModel1_deadline_asymTnd(coh_set', theta_m1);

%% =================================================================
% Model 2: parallel/independent magnitude-driven timing (collapsing
% bound), asymmetric tnd
% theta = [k_choice, b, k_time, B_time, Balpha_time, Bbeta_time, tnd_up, tnd_dn]
%% =================================================================
theta0_m2 = [15     0    15    0.5   500    1     400   450];
lo_m2     = [0.1   -30   0.1   0.01   10    0.01   100   100];
hi_m2     = [60     30   60    5    2000   30     800   800];

[theta_m2, nlogl_m2] = fminsearchbnd( ...
    @(th) fitJointCost_generic(th, data1, @calcModel2_magnitudeTiming_asymTnd), ...
    theta0_m2, lo_m2, hi_m2, opts);

nParams_m2 = numel(theta_m2);
AIC_m2 = 2*nParams_m2 + 2*nlogl_m2;
BIC_m2 = nParams_m2*log(nBIC) + 2*nlogl_m2;

[t1_m2, t2_m2, p_m2] = calcModel2_magnitudeTiming_asymTnd(coh_set', theta_m2);

%% =================================================================
% Model 3: standard DDM, collapsing bound + asymmetric tnd (unchanged
% from v1)
% theta = [k, b, B, Balpha, Bbeta, tnd_up, tnd_dn]
%% =================================================================
theta0_m3 = [15    0    0.5   500    1     400   450];
lo_m3     = [0.1  -30   0.01   10    0.01   100   100];
hi_m3     = [60    30   5    2000   30     800   800];

[theta_m3, nlogl_m3] = fminsearchbnd( ...
    @(th) fitJointCost_generic(th, data1, @calcModel3_collapseBound_asymTnd), ...
    theta0_m3, lo_m3, hi_m3, opts);

nParams_m3 = numel(theta_m3);
AIC_m3 = 2*nParams_m3 + 2*nlogl_m3;
BIC_m3 = nParams_m3*log(nBIC) + 2*nlogl_m3;

[t1_m3, t2_m3, p_m3] = calcModel3_collapseBound_asymTnd(coh_set', theta_m3);

%% -------------------------------
% Report
%% -------------------------------
fprintf('\n===== Model comparison (v2, AIC/BIC bar variant): %s, taskid in %s =====\n', date, mat2str(taskidList));
fprintf('Model 1 (imposed deadline, asym tnd)      : nParams=%d  nlogl=%.2f  AIC=%.2f  BIC=%.2f\n', nParams_m1, nlogl_m1, AIC_m1, BIC_m1);
fprintf('Model 2 (magnitude timing, coll, asym tnd): nParams=%d  nlogl=%.2f  AIC=%.2f  BIC=%.2f\n', nParams_m2, nlogl_m2, AIC_m2, BIC_m2);
fprintf('Model 3 (DDM, collapsing bound, asym tnd) : nParams=%d  nlogl=%.2f  AIC=%.2f  BIC=%.2f\n', nParams_m3, nlogl_m3, AIC_m3, BIC_m3);

modelNames = {'Model 1 (imposed deadline)','Model 2 (magnitude timing)','Model 3 (DDM, collapsing bound)'};

[~, bestIdx_AIC] = min([AIC_m1, AIC_m2, AIC_m3]);
[~, bestIdx_BIC] = min([BIC_m1, BIC_m2, BIC_m3]);
fprintf('Lowest AIC: %s\n', modelNames{bestIdx_AIC});
fprintf('Lowest BIC: %s\n', modelNames{bestIdx_BIC});
if bestIdx_AIC ~= bestIdx_BIC
    fprintf('*** AIC and BIC disagree on the winning model -- the conclusion is NOT robust to the choice of criterion. ***\n');
else
    fprintf('AIC and BIC agree on the winning model.\n');
end

%% -------------------------------
% Plot: data + all three model fits overlaid
%% -------------------------------
xax     = min(coh_set):0.01:max(coh_set);
xax_pos = 0:0.01:max(coh_set);
xax_neg = min(coh_set):0.01:0;

[~, ~, p_line_m1]     = calcModel1_deadline_asymTnd(xax', theta_m1);
[t1_line_m1, ~, ~]    = calcModel1_deadline_asymTnd(xax_pos', theta_m1);
[~, t2_line_m1, ~]    = calcModel1_deadline_asymTnd(xax_neg', theta_m1);

[~, ~, p_line_m2]     = calcModel2_magnitudeTiming_asymTnd(xax', theta_m2);
[t1_line_m2, ~, ~]    = calcModel2_magnitudeTiming_asymTnd(xax_pos', theta_m2);
[~, t2_line_m2, ~]    = calcModel2_magnitudeTiming_asymTnd(xax_neg', theta_m2);

[~, ~, p_line_m3]     = calcModel3_collapseBound_asymTnd(xax', theta_m3);
[t1_line_m3, ~, ~]    = calcModel3_collapseBound_asymTnd(xax_pos', theta_m3);
[~, t2_line_m3, ~]    = calcModel3_collapseBound_asymTnd(xax_neg', theta_m3);

p_mean1 = data1(:,6)./data1(:,7);
p_se1   = sqrt(p_mean1.*(1-p_mean1)./data1(:,7));

modelColors = [0.85 0.33 0.10;   % Model 1: orange
               0.30 0.60 0.30;   % Model 2: green
               0.20 0.20 0.80];  % Model 3: blue

figure; clf;
movegui(gcf,'center');

subplot(1,3,1); hold on;
errorbar(coh_set, p_mean1, p_se1, 'o', 'Color','k', 'MarkerFaceColor','k', 'capsize',0);
plot(xax, p_line_m1, '-', 'Color', modelColors(1,:), 'LineWidth',1.5);
plot(xax, p_line_m2, '-', 'Color', modelColors(2,:), 'LineWidth',1.5);
plot(xax, p_line_m3, '-', 'Color', modelColors(3,:), 'LineWidth',1.5);
xlabel('Motion strength'); ylabel('P_{up}');
legend({'data','Model 1: deadline','Model 2: magnitude timing','Model 3: DDM (collapsing bound)'}, 'Location','best');
xlim([min(coh_set)-0.05 max(coh_set)+0.05]);
title('Psychometric function');
box off;
fig_setting();

subplot(1,3,2); hold on;
errorbar(coh_set, RT1_mean, RT1_se, 'o', 'Color','k', 'MarkerFaceColor','k', 'capsize',0);
errorbar(coh_set, RT2_mean, RT2_se, 'o', 'Color','k', 'MarkerFaceColor','k', 'capsize',0);
plot(xax_pos, t1_line_m1, '-', 'Color', modelColors(1,:), 'LineWidth',1.5);
plot(xax_neg, t2_line_m1, '-', 'Color', modelColors(1,:), 'LineWidth',1.5);
plot(xax_pos, t1_line_m2, '-', 'Color', modelColors(2,:), 'LineWidth',1.5);
plot(xax_neg, t2_line_m2, '-', 'Color', modelColors(2,:), 'LineWidth',1.5);
plot(xax_pos, t1_line_m3, '-', 'Color', modelColors(3,:), 'LineWidth',1.5);
plot(xax_neg, t2_line_m3, '-', 'Color', modelColors(3,:), 'LineWidth',1.5);
xlabel('Motion strength'); ylabel('Reaction time (ms)');
xlim([min(coh_set)-0.05 max(coh_set)+0.05]);
title('Chronometric function');
box off;
fig_setting();

%% -------------------------------
% AIC/BIC bar graph across the three models (replaces the "inferred
% bound" panel from RT_behav_compareModels_v2.m)
%% -------------------------------
AICBIC = [AIC_m1 BIC_m1; AIC_m2 BIC_m2; AIC_m3 BIC_m3];

subplot(1,3,3); hold on;
hBar = bar(AICBIC, 'grouped');
hBar(1).FaceColor = [0.5 0.5 0.5];
hBar(2).FaceColor = [0.85 0.85 0.85];
set(gca, 'XTick', 1:3, 'XTickLabel', {'Model 1','Model 2','Model 3'});
ylabel('Information criterion value');
legend({'AIC','BIC'}, 'Location','best');
title('Model comparison: AIC & BIC');
box off;
fig_setting();

sgtitle(sprintf('%s: Model comparison v2 (AIC: M1=%.0f M2=%.0f M3=%.0f | BIC: M1=%.0f M2=%.0f M3=%.0f)', ...
    date, AIC_m1, AIC_m2, AIC_m3, BIC_m1, BIC_m2, BIC_m3));

theta_m3

%% -------------------------------
% Package results
%% -------------------------------
results.data1 = data1;
results.coh_set = coh_set;
results.nBIC = nBIC;

results.model1.theta   = theta_m1;
results.model1.nlogl   = nlogl_m1;
results.model1.nParams = nParams_m1;
results.model1.AIC     = AIC_m1;
results.model1.BIC     = BIC_m1;
results.model1.t1_pred = t1_m1;
results.model1.t2_pred = t2_m1;
results.model1.p_pred  = p_m1;

results.model2.theta   = theta_m2;
results.model2.nlogl   = nlogl_m2;
results.model2.nParams = nParams_m2;
results.model2.AIC     = AIC_m2;
results.model2.BIC     = BIC_m2;
results.model2.t1_pred = t1_m2;
results.model2.t2_pred = t2_m2;
results.model2.p_pred  = p_m2;

results.model3.theta   = theta_m3;
results.model3.nlogl   = nlogl_m3;
results.model3.nParams = nParams_m3;
results.model3.AIC     = AIC_m3;
results.model3.BIC     = BIC_m3;
results.model3.t1_pred = t1_m3;
results.model3.t2_pred = t2_m3;
results.model3.p_pred  = p_m3;

end
