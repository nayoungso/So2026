function RT_WhenWhat_correlation_Fig5_v2(filelist, useSaccWhat, RTrange)
% RT_WhenWhat_correlation_Fig5_v2
%
% Tests whether the "When" and "What" projections (S_d_when/S_s_when,
% S_d_what/S_s_what or S_d_what_sacc/S_s_what_sacc, saved by
% RT_project2_WhenWhatCD_forCorr.m) covary on a trial-by-trial basis, by
% correlating them at every pair of (When time, What time) points
% (a full cross-temporal correlation matrix, not just concurrent time).
% Trials are pooled across all sessions in filelist (a mega-analysis,
% not a per-session summary). Before correlating, both projections are
% residualized within each (coherence, choice) condition (subtracting
% that condition's own mean) and the What residual is sign-flipped to a
% choice-consistent direction, so the correlation reflects trial-to-
% trial covariation beyond what coherence/choice alone would produce.
%
% Monkey identity is auto-detected from the filename prefix: Harry uses
% all correct trials (plus 0%-coherence trials), Neptune uses only
% left-correct trials (Neptune's What projection only shows a reliable
% relationship on left-choice trials, per prior buildup-rate analyses).
%
% Produces a 3x2 summary figure: (row 1) full cross-temporal
% correlation heatmap (motion-aligned, saccade-aligned); (row 2) same
% heatmap masked to bins that are both positive and p<0.05; (row 3) the
% concurrent (diagonal, same-time) correlation trace, with significance
% marked two ways -- pointwise Benjamini-Hochberg FDR on the analytic
% Pearson p-values (dots), and a trial-shuffle cluster-based permutation
% test (bar), both computed by the local helper functions below.
%
% INPUTS
%   filelist    : cell array of saved WhenWhatProjection filenames (as
%                 saved by RT_project2_WhenWhatCD_forCorr.m), each
%                 starting with 'H' or 'N' to identify the monkey.
%   useSaccWhat : 0 = use the motion-aligned WhatCD projection
%                 (S_d_what/S_s_what); 1 = use the saccade-aligned
%                 WhatCD projection (S_d_what_sacc/S_s_what_sacc).
%   RTrange     : [] for no RT filter, or [min max] in ms to restrict
%                 to trials within that RT range.
%
% OUTPUT
%   None returned (figure only).
%
% REQUIRES ON PATH: fdr_bh.m (included alongside this file). All other
% helpers (compute_residuals, compute_corr, clusterPermTestDiagCorr,
% get_trial_index, rowCorr, rho2t, findClustersDiag, reportClusters,
% ternary, plotSigBar) are defined locally below.
%
% NOTES
%   - The saccade-aligned display window is requested as [-400 300]; it
%     is clamped to whatever tvec_s actually covers, with a warning if
%     clamped.
%
% EXAMPLE
%   RT_WhenWhat_correlation_Fig5_v2(filelist, 0, [])

path = ['~/So2026/NeuralData/output/WhenWhatProjections/'];

%% ===============================
% DEFAULTS
%% ===============================
if nargin < 3
    RTrange = [];
end

%% ===============================
% ACCUMULATORS
%% ===============================
Gr_When_d = [];
Gr_What_d = [];

Gr_When_s = [];
Gr_What_s = [];

%% ===============================
% LOOP OVER SESSIONS
%% ===============================
for ii = 1:length(filelist)

    load([path filelist{ii}]);

    %% -------------------------------
    % MONKEY IDENTITY (from filename prefix) -> TRIAL MODE
    %% -------------------------------
    if strncmpi(filelist{ii}, 'H', 1)
        isNeptune = false;
    elseif strncmpi(filelist{ii}, 'N', 1)
        isNeptune = true;
    else
        error('RT_WhenWhat_correlation_Fig5_v2:unknownMonkey', ...
            'Cannot determine monkey identity from filename "%s" (expected to start with ''H'' or ''N'').', ...
            filelist{ii});
    end

    if isNeptune
        trialMode = 'left-correct';   % Neptune: only left-choice correct trials
    else
        trialMode = 'correct';        % Harry: all correct trials
    end

    %% -------------------------------
    % SELECT WHAT VERSION
    %% -------------------------------
    if useSaccWhat
        S_what_d = S_d_what_sacc;
        S_what_s = S_s_what_sacc;
    else
        S_what_d = S_d_what;
        S_what_s = S_s_what;
    end

    %% -------------------------------
    % TRIAL SELECTION (FLEXIBLE)
    %% -------------------------------
    idx = get_trial_index(correct_all, rt_all, coh_all, trialMode, RTrange);

    % dots
    S_when_d = S_d_when(:,idx);
    S_what_d = S_what_d(:,idx);

    % sacc
    S_when_s = S_s_when(:,idx);
    S_what_s = S_what_s(:,idx);

    coh = coh_all(idx);
    choice = choice_all(idx);

    %% -------------------------------
    % RESIDUALIZATION
    %% -------------------------------
    [Swd, Swd2] = compute_residuals(S_when_d, S_what_d, coh, choice, trialMode);
    [Sws, Sws2] = compute_residuals(S_when_s, S_what_s, coh, choice, trialMode);

    %% -------------------------------
    % CONCATENATE
    %% -------------------------------
    Gr_When_d = [Gr_When_d Swd];
    Gr_What_d = [Gr_What_d Swd2];

    Gr_When_s = [Gr_When_s Sws];
    Gr_What_s = [Gr_What_s Sws2];

end

%% ===============================
% CORRELATIONS
%% ===============================
[r_d, p_d, r_diag_d, p_diag_d] = compute_corr(Gr_When_d, Gr_What_d);  % this results in When as rows (y) & What as columns (x)
[r_s, p_s, r_diag_s, p_diag_s] = compute_corr(Gr_When_s, Gr_What_s);

%% ===============================
% SIGNIFICANCE TESTING (concurrent correlation, row 3)
%% ===============================
% Pointwise FDR (Benjamini-Hochberg) on the analytic Pearson p-values,
% and a trial-shuffle cluster-based permutation test (trials are the
% exchangeable unit, since Gr_When/Gr_What are already trial-pooled
% across sessions) -- same rigor as RT_WhatBehavCorr.m.
nPerm = 5000; alpha = 0.05;

sigMask_diag_d_fdr = fdr_bh(p_diag_d, alpha);
sigMask_diag_s_fdr = fdr_bh(p_diag_s, alpha);

[sigMask_diag_d_clust, clusters_diag_d] = clusterPermTestDiagCorr(Gr_When_d, Gr_What_d, nPerm, alpha);
[sigMask_diag_s_clust, clusters_diag_s] = clusterPermTestDiagCorr(Gr_When_s, Gr_What_s, nPerm, alpha);

% report cluster windows to command line
reportClusters('concurrent corr (motion-aligned)',  tvec_d, clusters_diag_d);
reportClusters('concurrent corr (saccade-aligned)',  tvec_s, clusters_diag_s);

%% ===============================
% SACCADE-ALIGNED DISPLAY WINDOW
%% ===============================
xlim_s_desired = [-400 300];
xlim_s = [max(xlim_s_desired(1), min(tvec_s)), min(xlim_s_desired(2), max(tvec_s))];
if ~isequal(xlim_s, xlim_s_desired)
    fprintf(['Requested saccade-aligned window [%g %g] exceeds available data ' ...
        '[%g %g]; using [%g %g] instead.\n'], ...
        xlim_s_desired(1), xlim_s_desired(2), min(tvec_s), max(tvec_s), xlim_s(1), xlim_s(2));
end

%% ===============================
% PLOT (3×2)
%% ===============================
figure(); clf;

% ---- (1) motion-aligned heatmap ----
subplot(3,2,1)
imagesc(tvec_d, tvec_d, r_d);
set(gca,'YDir','normal'); axis square;
hold on; xline(0,'k:'); yline(0,'k:');
title('Motion-aligned (Fig 5a)');
ylabel('WhenProj time'); xlabel('WhatProj time');
colorbar
xlim([-200 650]); ylim([-200 650]);

subplot(3,2,3)
imagesc(tvec_d, tvec_d, (r_d>0) & (p_d<0.05));
set(gca,'YDir','normal'); axis square;
hold on; xline(0,'k:'); yline(0,'k:');
title('positive & p<0.05?');
ylabel('WhenProj time'); xlabel('WhatProj time');

xlim([-200 650]); ylim([-200 650]);

% ---- (2) sacc-aligned heatmap ----
subplot(3,2,2)
imagesc(tvec_s, tvec_s, r_s);
set(gca,'YDir','normal'); axis square;
hold on; xline(0,'k:'); yline(0,'k:');
title('Saccade-aligned (Fig 5a)');
ylabel('WhenProj time'); xlabel('WhatProj time');
colorbar
xlim(xlim_s); ylim(xlim_s)

subplot(3,2,4)
imagesc(tvec_s, tvec_s, (r_s>0) & (p_s<0.05));
set(gca,'YDir','normal'); axis square;
hold on; xline(0,'k:'); yline(0,'k:');
title('positive & p<0.05?');
ylabel('WhenProj time'); xlabel('WhatProj time');
xlim(xlim_s); ylim(xlim_s)

% ---- (3) motion concurrent ----
subplot(3,2,5)
plot(tvec_d, r_diag_d,'k','LineWidth',2); hold on;
yline(0,'k--');
xline(0,'k--');
xlabel('Time'); ylabel('Correlation');
title('Motion-aligned concurrent (Fig 5b)');
box off
xlim([-200 650]);

yl = ylim;
yspan = diff(yl);
plot(tvec_d(sigMask_diag_d_fdr), r_diag_d(sigMask_diag_d_fdr), 'k.', 'MarkerSize', 10);
plotSigBar(tvec_d, sigMask_diag_d_clust, yl(1) - 0.08*yspan, 'k');
ylim([yl(1) - 0.2*yspan, yl(2)]);

% ---- (4) sacc concurrent ----
subplot(3,2,6)
plot(tvec_s, r_diag_s,'k','LineWidth',2); hold on;
yline(0,'k--');
xline(0,'k--');
xlabel('Time'); ylabel('Correlation');
title('Sacc-aligned concurrent (Fig 5b)');
box off
xlim(xlim_s);

yl = ylim;
yspan = diff(yl);
plot(tvec_s(sigMask_diag_s_fdr), r_diag_s(sigMask_diag_s_fdr), 'k.', 'MarkerSize', 10);
plotSigBar(tvec_s, sigMask_diag_s_clust, yl(1) - 0.08*yspan, 'k');
ylim([yl(1) - 0.2*yspan, yl(2)]);

sgtitle('When-What correlation | Harry: correct trials, Neptune: left-correct trials');

end


function idx = get_trial_index(correct_all, rt_all, coh_all, trialMode, RTrange)

    % base selection
    switch trialMode
        case 'all'
            idx = true(size(correct_all));

        case 'correct'
            % include:
            % (1) correct trials
            % (2) 0% coherence trials (coded as ±1)
            idx = (correct_all == 1) | (abs(coh_all) == 1);
            %idx = (correct_all == 1) & (abs(coh_all) < 10);  % low coherence only


            sum(idx)
        case 'error'
            % true errors only (exclude 0% trials)
            idx = (correct_all == 0) & (abs(coh_all) ~= 1);
            %idx = (correct_all == 0) & (abs(coh_all) < 10);  % low coherence only

            %sum(idx)
        case 'left-correct'
            % special case for neptune
            idx = ((correct_all == 1)&(coh_all < 0)) | (coh_all==-1);
        case 'right-correct'
            % special case for neptune
            idx = ((correct_all == 1)&(coh_all > 0)) | (coh_all==1);
        otherwise
            error('Unknown trialMode');
    end

    % optional RT filtering
    if ~isempty(RTrange)
        idx = idx & (rt_all >= RTrange(1)) & (rt_all <= RTrange(2));
    end

    idx = idx(:); % enforce column
end

function [S1_res, S2_res] = compute_residuals(S1, S2, coh, choice,trialMode)

    S1_res = nan(size(S1));
    S2_res = nan(size(S2));

    cond = unique([coh choice],'rows');

    for k = 1:size(cond,1)

        idx_c = (coh==cond(k,1)) & (choice==cond(k,2));

        if sum(idx_c) < 1
            continue
        end

        % mean per condition
        m1 = mean(S1(:,idx_c),2);
        m2 = mean(S2(:,idx_c),2);

        % residuals
        r1 = S1(:,idx_c) - m1;
        r2 = S2(:,idx_c) - m2;


        % -------------------------------
        % SIGN ALIGNMENT for What projection (CRITICAL)
        % -------------------------------
        % choice: 0 or 1 → convert to ±1
        ch_sign = (choice(idx_c)*2 - 1);   % 0 (up/right)→-1, 1(down/left)→+1

        if strcmp(trialMode,'right-correct')
            ch_sign = abs(ch_sign);
        end

        % apply to each trial, such that deviation (residual) supporting
        % the choice is considered positive deviation
        r2 = r2 .* ch_sign';  % comment out for Neptune?!


        % store
        S1_res(:,idx_c) = r1;
        S2_res(:,idx_c) = r2;
    end
end

function [r, p, r_diag, p_diag] = compute_corr(Gr1, Gr2)

    nT = size(Gr1,1);

    r = nan(nT,nT);
    p = nan(nT,nT);

    for i = 1:nT
        for j = 1:nT

            x = Gr1(i,:)';
            y = Gr2(j,:)';

            valid = ~isnan(x) & ~isnan(y);

            if sum(valid) > 1
                [r(i,j), p(i,j)] = corr(x(valid), y(valid));
            end
        end
    end

    r_diag = diag(r)';
    p_diag = diag(p)';
end

function [sigMask, clusters] = clusterPermTestDiagCorr(X, Y, nPerm, alpha)
%CLUSTERPERMTESTDIAGCORR Trial-shuffle cluster-based permutation test for
% the concurrent (same-time) correlation between two pooled time x trial
% matrices (mega-analysis across sessions: trials are pooled, not
% per-session summary stats).
%   X, Y : nT x nTrials, residualized/sign-aligned signals, paired by
%          trial and time
%   Trials are the exchangeable unit under H0, so each permutation
%   shuffles the trial correspondence between X and Y once and applies
%   it across the whole time course, preserving temporal autocorrelation
%   within a trial (same rationale as clusterPermTestPooledCorr in
%   RT_WhatBehavCorr_v3.m).

if nargin < 3 || isempty(nPerm), nPerm = 5000; end
if nargin < 4 || isempty(alpha), alpha = 0.05; end

validTrial = ~any(isnan(X),1) & ~any(isnan(Y),1);
X = X(:,validTrial);
Y = Y(:,validTrial);

nTrials = size(X,2);
df = nTrials - 2;
threshold = tinv(1-alpha/2, df);

tobs = rho2t(rowCorr(X,Y), nTrials);
clusters = findClustersDiag(abs(tobs) > threshold, tobs);

nullMax = zeros(1,nPerm);
for p = 1:nPerm
    permOrder = randperm(nTrials);
    tp = rho2t(rowCorr(X, Y(:,permOrder)), nTrials);
    cl_p = findClustersDiag(abs(tp) > threshold, tp);
    if isempty(cl_p)
        nullMax(p) = 0;
    else
        nullMax(p) = max([cl_p.mass]);
    end
end

sigMask = false(1,size(X,1));
for c = 1:numel(clusters)
    clusters(c).pval = (1 + sum(nullMax >= clusters(c).mass)) / (nPerm + 1);
    clusters(c).sig  = clusters(c).pval < alpha;
    if clusters(c).sig
        sigMask(clusters(c).idx) = true;
    end
end

end

function r = rowCorr(X, Y)
% Pearson correlation between corresponding rows of X and Y (each row a
% timepoint, columns are trials).
mx = mean(X,2); my = mean(Y,2);
Xc = X - mx; Yc = Y - my;
r = sum(Xc.*Yc,2) ./ sqrt(sum(Xc.^2,2) .* sum(Yc.^2,2));
r = r';
end

function t = rho2t(rho, n)
t = rho .* sqrt(n-2) ./ sqrt(1 - rho.^2);
end

function clusters = findClustersDiag(candidate, tvals)
clusters = struct('idx',{},'mass',{},'pval',{},'sig',{});
d      = diff([0 candidate(:)' 0]);
starts = find(d == 1);
ends   = find(d == -1) - 1;
for k = 1:numel(starts)
    idx = starts(k):ends(k);
    clusters(k).idx  = idx;
    clusters(k).mass = sum(abs(tvals(idx)));
    clusters(k).pval = NaN;
    clusters(k).sig  = false;
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
