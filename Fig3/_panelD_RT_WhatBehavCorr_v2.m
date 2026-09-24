function RT_WhatBehavCorr_v2(filelist, useSaccWhat, t_begin, t_end, RTcutoff, Cohcutoff)
% RT_WhatBehavCorr_v2
%
% Tests whether the "What projection" (S_d_dec or S_d_choice, saved by
% RT_project2_WhatCD_FAST.m) relates to behavior on slow, low-coherence
% trials (RT >= RTcutoff, |coherence| <= Cohcutoff), after removing the
% coherence-driven component of both the projection and RT: at each
% time point, the projection and RT are each residualized by
% subtracting their own per-coherence-group mean (0%-coherence trials
% from both signs are pooled as one group), then the projection
% residual is sign-flipped to a choice-consistent direction
% (-sign(coherence)) before correlating with the RT residual.
%
% Two per-timepoint statistics are computed per session then combined
% across sessions (mean +/- SEM): (1) rho_RT, the correlation between
% the sign-aligned residual projection and the residual RT (restricted
% to left-choice trials for monkey N, both choices for monkey H, per
% prior buildup-rate analyses), and (2) beta_choice, the
% standardized (beta/SE) logistic-regression coefficient of choice on
% the (non-sign-flipped) residual projection. Significance across
% sessions is assessed two ways: a cluster-based sign-flip permutation
% test (clusterPermTest.m) and a pointwise one-sample t-test against 0
% with Benjamini-Hochberg FDR correction (fdr_bh.m) -- both shown on
% the resulting 2-panel figure (choice correlation, RT correlation).
%
% INPUTS
%   filelist    : cell array of saved WhatProjection filenames (as saved
%                 by RT_project2_WhatCD_FAST.m), each starting with 'H'
%                 or 'N' to identify the monkey.
%   useSaccWhat : logical; if true, uses S_d_choice (saccade-trained
%                 CD); if false, uses S_d_dec (dots-trained CD).
%   t_begin, t_end : dots-aligned time window (ms, must be values in
%                 tvec_d) over which the correlations are computed.
%   RTcutoff    : minimum RT (ms) for a trial to be included.
%   Cohcutoff   : maximum |coherence| for a trial to be included.
%
% OUTPUT
%   None returned (figure only).
%
% REQUIRES ON PATH: clusterPermTest.m, fdr_bh.m, shadedErrorBar.m
% (included alongside this file).
%
% EXAMPLE
%   RT_WhatBehavCorr_v2(filelist, false, 200, 500, 670, 64)

path = ['~/So2026/NeuralData/output/WhatProjections/'];

nSess = length(filelist);

%{
RTcutoff = 670; % only trials with RT >= 670
Cohcutoff = 64; % coherence less than 10%

t_begin = 200;
t_end = 500;
%}


for i = 1:nSess

    %% ===============================
    % MONKEY IDENTITY (from filename prefix)
    %% ===============================
    if strncmpi(filelist{i}, 'H', 1)
        isNeptune = false;
    elseif strncmpi(filelist{i}, 'N', 1)
        isNeptune = true;
    else
        error('RT_WhatBehavCorr:unknownMonkey', ...
            'Cannot determine monkey identity from filename "%s" (expected to start with ''H_'' or ''N_'').', ...
            filelist{i});
    end

    %% ===============================
    % LOAD
    %% ===============================
    load([path filelist{i}], ...
        'S_d_dec','S_d_choice','tvec_d','rt','choice','coh_all');

    %% ===============================
    % TIME INDICES
    %% ===============================
    t_begin_id = find(tvec_d == t_begin);
    t_end_id   = find(tvec_d == t_end);

    if useSaccWhat
        S = S_d_choice(t_begin_id:t_end_id,:);   % DV (time x trial)
    else
        S = S_d_dec(t_begin_id:t_end_id,:);   % DV (time x trial)
    end
    %% ===============================
    % TRIAL SELECTION
    %% ===============================
    idx = (rt >= RTcutoff) & (abs(coh_all) <= Cohcutoff);

    S = S(:,idx);
    rt_sel = rt(idx);
    choice_sel = choice(idx);
    coh_sel = coh_all(idx);

    %% ===============================
    % RESIDUALIZATION (BY COHERENCE)
    %% ===============================
    S_res = nan(size(S));
    rt_res = nan(size(rt_sel));

    ucoh = unique(coh_sel);

    for c = 1:length(ucoh)

        if abs(ucoh(c)) == 1
            idx_c = abs(coh_sel) == 1;   % 0% trials
        else
            idx_c = coh_sel == ucoh(c);
        end

        if sum(idx_c) < 5, continue; end

        % means
        mS  = mean(S(:,idx_c),2);
        mRT = mean(rt_sel(idx_c));

        % residuals
        S_res(:,idx_c) = S(:,idx_c) - mS;
        rt_res(idx_c)  = rt_sel(idx_c) - mRT;
    end


%% ===============================
% SIGN ALIGNMENT (CHOICE-CONSISTENT)
%% ===============================
sign_flip = -sign(coh_sel(:));   % ensures consistent direction

S_res_aligned = S_res .* sign_flip';


   %% ===============================
% RT CORRELATION
%% ===============================
% Monkey N: only left-choice trials show a significant DV relationship
% (per buildup rate analyses), so restrict the RT correlation to those
% trials. Monkey H uses both choice trials.
if isNeptune
    idx_RT = choice_sel(:) == 1;   % 1 left choice; 0 right
else
    idx_RT = true(size(choice_sel(:)));
end

nT = size(S_res,1);

rho_RT{i} = nan(1,nT);

for t = 1:nT

    x = S_res_aligned(t,:)';
    y = rt_res(:);

    valid = ~isnan(x) & ~isnan(y) & idx_RT;

    if sum(valid) > 10
        rho_RT{i}(t) = corr(x(valid), y(valid));
    end
end

    %% ===============================
    % CHOICE LOGISTIC REGRESSION
    %% ===============================
    beta_choice{i} = nan(1,nT);

    for t = 1:nT

        x = S_res(t,:)';
        y = choice_sel(:);

        valid = ~isnan(x) & ~isnan(y);

        if sum(valid) < 10, continue; end

        [b,~,stats] = glmfit(x(valid), y(valid), 'binomial');
        beta = b(2); se = stats.se(2);
        beta_choice{i}(t) = beta / se;
    end

end

%% ===============================
% COMBINE ACROSS SESSIONS
%% ===============================
rho_RT_mat = vertcat(rho_RT{:});
beta_choice_mat = vertcat(beta_choice{:});

% mean + SEM
rho_RT_mean = nanmean(rho_RT_mat,1);
rho_RT_se   = nanstd(rho_RT_mat,[],1) ./ sqrt(sum(~isnan(rho_RT_mat),1));

beta_choice_mean = nanmean(beta_choice_mat,1);
beta_choice_se   = nanstd(beta_choice_mat,[],1) ./ sqrt(sum(~isnan(beta_choice_mat),1));

tvec = tvec_d(t_begin_id:t_end_id);
nT   = length(tvec);

%% ===============================
% SIGNIFICANCE TESTING (correlation vs. 0, across sessions)
%% ===============================
% RT panel: Fisher-z transform so per-session correlations are approx.
% normal and comparable across sessions before combining.
z_RT_mat = atanh(rho_RT_mat);

% Choice panel: beta_choice_mat is already a per-session beta/SE (z-like
% statistic), so no transform is needed before combining.

nPerm = 5000; alpha = 0.05;

% --- cluster-based permutation test (sign-flip across sessions) ---
[sigMask_RT_clust,     clusters_RT]     = clusterPermTest(z_RT_mat,        nPerm, alpha);
[sigMask_choice_clust, clusters_choice] = clusterPermTest(beta_choice_mat, nPerm, alpha);

% --- pointwise one-sample t-test + FDR (Benjamini-Hochberg) ---
pval_RT     = pointwiseTtest(z_RT_mat);
pval_choice = pointwiseTtest(beta_choice_mat);

sigMask_RT_fdr     = fdr_bh(pval_RT,     alpha);
sigMask_choice_fdr = fdr_bh(pval_choice, alpha);

% report cluster windows to command line
reportClusters('rho_RT',        tvec, clusters_RT);
reportClusters('beta_choice',   tvec, clusters_choice);

%% ===============================
% PLOT
%% ===============================
figure; clf;

subplot(1,2,1)
shadedErrorBar(tvec, beta_choice_mean, beta_choice_se, 'k'); hold on
yline(0,'k--');
xlabel('Time'); ylabel('\beta / SE');
title('Choice correlation');
box off

yl = ylim;
yspan = diff(yl);
plot(tvec(sigMask_choice_fdr), beta_choice_mean(sigMask_choice_fdr), 'k.', 'MarkerSize', 10);
plotSigBar(tvec, sigMask_choice_clust, yl(1) - 0.08*yspan, 'k');
ylim([yl(1) - 0.2*yspan, yl(2)]);

subplot(1,2,2)
shadedErrorBar(tvec, rho_RT_mean, rho_RT_se, 'k'); hold on
yline(0,'k--');
xlabel('Time'); ylabel('\rho');
title('RT correlation');
box off

yl = ylim;
yspan = diff(yl);
plot(tvec(sigMask_RT_fdr), rho_RT_mean(sigMask_RT_fdr), 'k.', 'MarkerSize', 10);
plotSigBar(tvec, sigMask_RT_clust, yl(1) - 0.08*yspan, 'k');
ylim([yl(1) - 0.2*yspan, yl(2)]);

sgtitle('What projection vs. behavior');

end

%% ===============================
% LOCAL HELPERS
%% ===============================
function pvals = pointwiseTtest(zMat)
% one-sample t-test of each column against 0
nT = size(zMat,2);
pvals = nan(1,nT);
for t = 1:nT
    col = zMat(:,t);
    col = col(~isnan(col));
    if numel(col) >= 3
        [~,pvals(t)] = ttest(col);
    end
end
end

function reportClusters(label, tvec, clusters)
fprintf('--- cluster permutation test: %s ---\n', label);
if isempty(clusters)
    fprintf('  no candidate clusters\n');
    return
end
for c = 1:numel(clusters)
    idx = clusters(c).idx;
    fprintf('  window [%g %g] ms, mass=%.2f, p=%.4f%s\n', ...
        tvec(idx(1)), tvec(idx(end)), clusters(c).mass, clusters(c).pval, ...
        ternary(clusters(c).sig, ' *SIG*', ''));
end
end

function s = ternary(cond, a, b)
if cond, s = a; else, s = b; end
end

function plotSigBar(tvec, mask, yLevel, color)
% Draw each contiguous significant run as a single bar extending half a
% bin width beyond its first/last time point, so isolated single-bin runs
% are visible (not just a dot) and adjacent bins form one continuous bar.
if ~any(mask), return; end

tvec = tvec(:)';
mask = mask(:)';
halfwidth = median(diff(tvec)) / 2;

d = diff([0, mask, 0]);
starts = find(d == 1);
stops  = find(d == -1) - 1;

for k = 1:numel(starts)
    x0 = tvec(starts(k)) - halfwidth;
    x1 = tvec(stops(k))  + halfwidth;
    plot([x0 x1], [yLevel yLevel], '-', 'Color', color, 'LineWidth', 4);
end
end
