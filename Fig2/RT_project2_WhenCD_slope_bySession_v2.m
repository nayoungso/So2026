function RT_project2_WhenCD_slope_bySession_v2(monkID, filelist, t_start, t_end, doSmooth, smoothWin)
% RT_project2_WhenCD_slope_bySession_v2
%
% Compute session-wise slopes of When projection (S_d) over a specified time window,
% then summarize across sessions as mean +/- SEM.
%
% Each saved WhenProjection file (as produced by RT_project2_WhenCD_FAST.m)
% supplies S_d (dots-aligned projection) and rt_group (per-trial RT-group
% label) for one session; this script fits a linear slope of S_d vs. time
% within [t_start, t_end] for each session x RT-group cell.
%
% INPUTS
%   monkID    : 'H', 'N', or 'A'
%   filelist  : cell array of saved WhenProjection filenames
%   t_start   : slope window start time in ms (dots aligned)
%   t_end     : slope window end time in ms (dots aligned)
%   doSmooth  : 0/1, smooth the mean trace with a Gaussian window before display
%   smoothWin : smoothdata window size (samples), used only if doSmooth
%
% OUTPUT
%   None returned; saves the session-level slope summary to [path
%   datfilename] and the summary figure to [path 'figs/' figfilename]
%   (filenames depend on monkID and the [t_start, t_end] window).
%
% REQUIRES ON PATH: fig_setting.m (included alongside this file).
%
% NOTES
%   - Uses ONLY the saved S_d projection (nascent weights).
%   - Assumes that if normalized weights were used during projection generation,
%     they are already reflected in S_d in the saved file.
%   - Computes slopes within each session and RT group.
%   - Plots mean +/- SEM across sessions.
%   - Also computes a per-session linear trend of slope vs mean RT and tests
%     whether the slope of that trend differs from zero across sessions.
%   - Also computes a pooled (session x RT-bin) correlation between slope
%     and mean RT, with all RT bins and with the fastest/slowest RT bin
%     excluded, to compare against the coarser 6-point group-mean
%     correlation.
%   - Because points from the same session are not independent, also
%     assesses whether that pooled correlation is inflated by
%     pseudoreplication: (1) a within-session demeaned correlation, and
%     (2) a linear mixed-effects model with Session as a random effect.
%
% EXAMPLE
%   RT_project2_WhenCD_slope_bySession_v2('H', filelist, 200, 500, 0, [])

%% paths
path = ['~/So2026/NeuralData/output/WhenProjections/'];

%% -------------------------------
% Titles / filenames
%% -------------------------------
if monkID=='H'
    titlestring = sprintf('When projection slope across sessions (Harry), %d to %d ms', t_start, t_end);
    figfilename = sprintf('H_WhenSlope_bySession_%dto%d_FAST.fig', t_start, t_end);
    datfilename = sprintf('H_WhenSlope_bySession_%dto%d_FAST.mat', t_start, t_end);

elseif monkID=='N'
    titlestring = sprintf('When projection slope across sessions (Neptune), %d to %d ms', t_start, t_end);
    figfilename = sprintf('N_WhenSlope_bySession_%dto%d_FAST.fig', t_start, t_end);
    datfilename = sprintf('N_WhenSlope_bySession_%dto%d_FAST.mat', t_start, t_end);

elseif monkID=='A'
    titlestring = sprintf('When projection slope across sessions (Both monkeys), %d to %d ms', t_start, t_end);
    figfilename = sprintf('All_WhenSlope_bySession_%dto%d_FAST.fig', t_start, t_end);
    datfilename = sprintf('All_WhenSlope_bySession_%dto%d_FAST.mat', t_start, t_end);

else
    error('monkID must be ''H'', ''N'', or ''A''.');
end

%% -------------------------------
% Parameters
%% -------------------------------
ColorMapRT = [0 0 1; 0.1 0.3 1; 0.1 0.6 1; 0.1 0.9 1; 0.5 1 1; 0.9 1 1];

nSess = length(filelist);

% we will discover RT group number from data
rt_groupno_max = 0;

% store results in cells first because some sessions may differ slightly
slope_sess_cell = cell(nSess,1);
meanRT_sess_cell = cell(nSess,1);
meanTrace_sess_cell = cell(nSess,1);
tvec_sess_cell = cell(nSess,1);
sessname = cell(nSess,1);

trend_beta = nan(nSess,1);      % slope ~ meanRT, session-level beta
trend_intercept = nan(nSess,1);
trend_r = nan(nSess,1);
trend_p = nan(nSess,1);

%% -------------------------------
% Loop through sessions
%% -------------------------------
for ii = 1:nSess

    D = load([path, filelist{ii}], 'S_d', 'rt_group', 'rt_all', 'tvec_d', 'useNorm');

    if ~isfield(D,'S_d') || ~isfield(D,'rt_group') || ~isfield(D,'rt_all') || ~isfield(D,'tvec_d')
        error('File %s is missing one or more required variables: S_d, rt_group, rt_all, tvec_d.', filelist{ii});
    end

    S_d = D.S_d;               % time x trial
    rt_group = D.rt_group(:);  % trial x 1
    rt_all = D.rt_all(:);      % trial x 1
    tvec_d = D.tvec_d(:);      % time x 1

    sessname{ii} = filelist{ii};

    % optional check: if the file saved useNorm flag, warn if false
    if isfield(D,'useNorm')
        if ~D.useNorm
            warning('File %s has useNorm == false. S_d may not reflect normalized weights.', filelist{ii});
        end
    end

    idx_win = find(tvec_d >= t_start & tvec_d <= t_end);
    if numel(idx_win) < 2
        error('Slope window [%d, %d] has fewer than 2 bins in session %s.', t_start, t_end, filelist{ii});   %% don't expect to happen.
    end

    t_fit = tvec_d(idx_win);

    rt_groupno = max(rt_group);
    rt_groupno_max = max(rt_groupno_max, rt_groupno);

    slope_this = nan(1, rt_groupno);
    meanRT_this = nan(1, rt_groupno);
    meanTrace_this = nan(length(tvec_d), rt_groupno);

    for k = 1:rt_groupno
        idx = (rt_group == k);

        if sum(idx) < 3
            % not enough trials for reliable mean / slope
            continue;
        end

        y = nanmean(S_d(:,idx), 2);
        % ---- smoothing (NEW) ----
        if doSmooth
            y = smoothdata(y, 'gaussian', smoothWin);
        end
        meanTrace_this(:,k) = y;
        meanRT_this(k) = nanmean(rt_all(idx));

        y_fit = y(idx_win);
        good = ~isnan(t_fit) & ~isnan(y_fit);

        if sum(good) >= 2
            p = polyfit(t_fit(good), y_fit(good), 1);
            slope_this(k) = p(1);   % units: projection units / ms
        end
    end

    slope_sess_cell{ii} = slope_this;
    meanRT_sess_cell{ii} = meanRT_this;
    meanTrace_sess_cell{ii} = meanTrace_this;
    tvec_sess_cell{ii} = tvec_d;

    % ---- session-level linear trend: slope vs mean RT ----
    goodTrend = ~isnan(slope_this) & ~isnan(meanRT_this);
    if sum(goodTrend) >= 2
        X = meanRT_this(goodTrend)';
        Y = slope_this(goodTrend)';

        p_lin = polyfit(X, Y, 1);
        trend_beta(ii) = p_lin(1);
        trend_intercept(ii) = p_lin(2);

        R = corrcoef(X, Y);
        if numel(R) >= 4
            trend_r(ii) = R(1,2);
        end

        [Rtmp, Ptmp] = corr(X, Y, 'type', 'Pearson');
        trend_r(ii) = Rtmp;
        trend_p(ii) = Ptmp;
    end
end

%% -------------------------------
% Convert cells to padded matrices
%% -------------------------------
slope_sess = nan(nSess, rt_groupno_max);
meanRT_sess = nan(nSess, rt_groupno_max);

for ii = 1:nSess
    tmp1 = slope_sess_cell{ii};
    tmp2 = meanRT_sess_cell{ii};

    slope_sess(ii,1:length(tmp1)) = tmp1;
    meanRT_sess(ii,1:length(tmp2)) = tmp2;
end

%% -------------------------------
% Pooled (session x RT-bin) correlation between slope and mean RT
%% -------------------------------
% Each session contributes up to rt_groupno_max (meanRT, slope) pairs;
% pooling across sessions gives many more points (nSess x rt_groupno_max)
% than the 6-point across-session group-mean regression computed below.
% NOTE: points from the same session are not independent draws, so this
% is a descriptive/exploratory correlation, not a replacement for the
% session-level trend_beta signed-rank test above.

X_all = meanRT_sess(:);
Y_all = slope_sess(:);
goodAll = ~isnan(X_all) & ~isnan(Y_all);
if sum(goodAll) >= 3
    [r_pooled_all, p_pooled_all] = corr(X_all(goodAll), Y_all(goodAll), 'type', 'Pearson');
else
    r_pooled_all = NaN;
    p_pooled_all = NaN;
end

if rt_groupno_max >= 3
    midBins = 2:(rt_groupno_max-1);
    X_mid = meanRT_sess(:,midBins);
    Y_mid = slope_sess(:,midBins);
    X_mid = X_mid(:);
    Y_mid = Y_mid(:);
    goodMid = ~isnan(X_mid) & ~isnan(Y_mid);
    if sum(goodMid) >= 3
        [r_pooled_mid, p_pooled_mid] = corr(X_mid(goodMid), Y_mid(goodMid), 'type', 'Pearson');
    else
        r_pooled_mid = NaN;
        p_pooled_mid = NaN;
    end
else
    midBins = [];
    r_pooled_mid = NaN;
    p_pooled_mid = NaN;
end

%% -------------------------------
% Pseudoreplication checks for the pooled correlation
%% -------------------------------
% (1) Within-session demeaning: subtract each session's own mean RT and
% mean slope before pooling. If the raw pooled correlation above is
% driven mainly by between-session differences (e.g., sessions with
% faster RT overall also having shallower slopes overall), this
% within-session correlation should collapse toward zero.
X_dm = meanRT_sess - nanmean(meanRT_sess, 2);
Y_dm = slope_sess - nanmean(slope_sess, 2);
x_dm = X_dm(:);
y_dm = Y_dm(:);
goodDM = ~isnan(x_dm) & ~isnan(y_dm);
if sum(goodDM) >= 3
    [r_within, p_within] = corr(x_dm(goodDM), y_dm(goodDM), 'type', 'Pearson');
else
    r_within = NaN;
    p_within = NaN;
end

% (1b) Same within-session demeaned correlation, restricted to the
% middle RT bins (fastest/slowest excluded) -- robustness control.
if ~isempty(midBins)
    x_dm_mid = X_dm(:,midBins);
    y_dm_mid = Y_dm(:,midBins);
    x_dm_mid = x_dm_mid(:);
    y_dm_mid = y_dm_mid(:);
    goodDMmid = ~isnan(x_dm_mid) & ~isnan(y_dm_mid);
    if sum(goodDMmid) >= 3
        [r_within_mid, p_within_mid] = corr(x_dm_mid(goodDMmid), y_dm_mid(goodDMmid), 'type', 'Pearson');
    else
        r_within_mid = NaN;
        p_within_mid = NaN;
    end
else
    r_within_mid = NaN;
    p_within_mid = NaN;
end

% (2) Linear mixed-effects model: Session as a random effect, so the
% fixed effect of RT on slope is estimated after accounting for
% between-session clustering (correct effective N).
sess_id = repmat((1:nSess)', 1, rt_groupno_max);
sess_id = sess_id(:);
RTvec = meanRT_sess(:);
slopevec = slope_sess(:);
goodLME = ~isnan(RTvec) & ~isnan(slopevec);

lme_intOnly = [];
lme_slope = [];
beta_RT_intOnly = NaN;
p_RT_intOnly = NaN;
beta_RT_slope = NaN;
p_RT_slope_lme = NaN;
p_compare_lme = NaN;

if sum(goodLME) >= 3
    tbl = table(sess_id(goodLME), RTvec(goodLME), slopevec(goodLME), ...
        'VariableNames', {'Session','RT','Slope'});
    tbl.Session = categorical(tbl.Session);

    lme_intOnly = fitlme(tbl, 'Slope ~ RT + (1|Session)');
    isRT = strcmp(lme_intOnly.Coefficients.Name, 'RT');
    beta_RT_intOnly = lme_intOnly.Coefficients.Estimate(isRT);
    p_RT_intOnly = lme_intOnly.Coefficients.pValue(isRT);

    try
        lme_slope = fitlme(tbl, 'Slope ~ RT + (RT|Session)');
        isRT = strcmp(lme_slope.Coefficients.Name, 'RT');
        beta_RT_slope = lme_slope.Coefficients.Estimate(isRT);
        p_RT_slope_lme = lme_slope.Coefficients.pValue(isRT);

        comp = compare(lme_intOnly, lme_slope);
        p_compare_lme = comp.pValue(2);
    catch ME
        warning('Random-slope LME did not fit (%s); reporting random-intercept-only model.', ME.message);
    end
end

%% -------------------------------
% Summary across sessions
%% -------------------------------
mean_slope = nanmean(slope_sess, 1);
sem_slope = nanstd(slope_sess, 0, 1) ./ sqrt(sum(~isnan(slope_sess), 1));

mean_RTgroup = nanmean(meanRT_sess, 1);
sem_RTgroup = nanstd(meanRT_sess, 0, 1) ./ sqrt(sum(~isnan(meanRT_sess), 1));

n_per_group = sum(~isnan(slope_sess), 1);

%% -------------------------------
% Stats across sessions
%% -------------------------------
% 1) Trend test: does slope change with RT across sessions?
good_beta = ~isnan(trend_beta);
if sum(good_beta) >= 2
    [~, p_ttest_beta, ~, stats_ttest_beta] = ttest(trend_beta(good_beta), 0);
    p_signrank_beta = signrank(trend_beta(good_beta), 0);
else
    p_ttest_beta = NaN;
    stats_ttest_beta = struct('tstat', NaN, 'df', NaN);
    p_signrank_beta = NaN;
end

% 2) Optional per-group one-sample tests vs zero slope
p_ttest_group = nan(1, rt_groupno_max);
p_signrank_group = nan(1, rt_groupno_max);
for k = 1:rt_groupno_max
    x = slope_sess(:,k);
    x = x(~isnan(x));
    if numel(x) >= 2
        [~, p_ttest_group(k)] = ttest(x, 0);
        p_signrank_group(k) = signrank(x, 0);
    end
end

%% -------------------------------
% Grand-average traces across sessions (for display only)
%% -------------------------------
% align to a common tvec if sessions differ slightly
binwidth = 25;
t_min = -200;
t_max = 1500;  % generous upper bound; will trim later

common_tvec = t_min:binwidth:t_max;
common_trace_sess = nan(length(common_tvec), rt_groupno_max, nSess);

for ii = 1:nSess
    tvec_d = tvec_sess_cell{ii};
    meanTrace_this = meanTrace_sess_cell{ii};

    [~, idx_common, idx_local] = intersect(common_tvec, tvec_d);

    for k = 1:size(meanTrace_this,2)
        common_trace_sess(idx_common, k, ii) = meanTrace_this(idx_local, k);
    end
end

% keep only time bins that are present in at least one session
keep_t = any(any(~isnan(common_trace_sess), 2), 3);
common_tvec = common_tvec(keep_t);
common_trace_sess = common_trace_sess(keep_t,:,:);

mean_trace = nanmean(common_trace_sess, 3);
sem_trace = nanstd(common_trace_sess, 0, 3) ./ sqrt(sum(~isnan(common_trace_sess), 3));

%% -------------------------------
% Plot
%% -------------------------------
figure; clf;

% ---- Panel 1: session-averaged traces ----
subplot(1,2,1); hold on;

ymin = min(mean_trace(:) - sem_trace(:));
ymax = max(mean_trace(:) + sem_trace(:));
if isempty(ymin) || isnan(ymin), ymin = -1; end
if isempty(ymax) || isnan(ymax), ymax = 1; end

patch([t_start t_end t_end t_start], [ymin ymin ymax ymax], ...
      [0.92 0.92 0.92], 'EdgeColor', 'none', 'FaceAlpha', 0.35);
line([0 0], [ymin ymax], 'Color', 'k');

for k = 1:rt_groupno_max
    if all(isnan(mean_trace(:,k)))
        continue;
    end

    % mean
    plot(common_tvec, mean_trace(:,k), 'Color', ColorMapRT(k,:), 'LineWidth', 1.8);

    % sem
    x = common_tvec(:);
    y = mean_trace(:,k);
    e = sem_trace(:,k);

    good = ~isnan(x) & ~isnan(y) & ~isnan(e);
    if any(good)
        xx = [x(good); flipud(x(good))];
        yy = [y(good)-e(good); flipud(y(good)+e(good))];
        patch(xx, yy, ColorMapRT(k,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    end
end

xlabel('Time from dots (ms)');
ylabel('When projection');
title('Session-averaged projections');
ylim([ymin ymax]);
box off;
fig_setting();

% ---- Panel 2: slope vs RT (session mean +/- SEM) ----
subplot(1,2,2); hold on;

for k = 1:rt_groupno_max
    if isnan(mean_slope(k)) || isnan(mean_RTgroup(k))
        continue;
    end

    % horizontal SEM (RT)
    line([mean_RTgroup(k)-sem_RTgroup(k), mean_RTgroup(k)+sem_RTgroup(k)], ...
         [mean_slope(k), mean_slope(k)], ...
         'Color', ColorMapRT(k,:), 'LineWidth', 1.2);

    % vertical SEM (slope)
    line([mean_RTgroup(k), mean_RTgroup(k)], ...
         [mean_slope(k)-sem_slope(k), mean_slope(k)+sem_slope(k)], ...
         'Color', ColorMapRT(k,:), 'LineWidth', 1.2);

    plot(mean_RTgroup(k), mean_slope(k), 'o', ...
        'MarkerSize', 8, ...
        'MarkerFaceColor', ColorMapRT(k,:), ...
        'MarkerEdgeColor', ColorMapRT(k,:));
end

% regression line through group means
goodPlot = ~isnan(mean_RTgroup) & ~isnan(mean_slope);

if sum(goodPlot) >= 2
    xfit = mean_RTgroup(goodPlot);
    yfit = mean_slope(goodPlot);

    xfit = xfit(:);
    yfit = yfit(:);

    Xmat = [ones(length(xfit),1), xfit];
    b = Xmat \ yfit;   % b(1)=intercept, b(2)=slope

    xline = linspace(min(xfit), max(xfit), 100)';
    yline = [ones(size(xline)), xline] * b;

    plot(xline, yline, 'k-', 'LineWidth', 1.5);

    % optional descriptive printout
    [r_tmp, p_tmp] = corr(xfit, yfit, 'type', 'Pearson');
    fprintf('Mean-point regression: intercept = %.5f, beta = %.5f, r = %.4f, p = %.4g\n', ...
        b(1), b(2), r_tmp, p_tmp);
end

text(0.05, 0.95, sprintf('Pooled (all bins): r = %.2f, p = %.3f', r_pooled_all, p_pooled_all), ...
    'Units', 'normalized', 'VerticalAlignment', 'top');
if ~isnan(r_pooled_mid)
    text(0.05, 0.87, sprintf('Pooled (mid bins only): r = %.2f, p = %.3f', r_pooled_mid, p_pooled_mid), ...
        'Units', 'normalized', 'VerticalAlignment', 'top');
end
if ~isnan(r_within)
    text(0.05, 0.79, sprintf('Within-session (demeaned): r = %.2f, p = %.3f', r_within, p_within), ...
        'Units', 'normalized', 'VerticalAlignment', 'top');
end
if ~isnan(r_within_mid)
    text(0.05, 0.71, sprintf('Within-session (demeaned, mid bins): r = %.2f, p = %.3f', r_within_mid, p_within_mid), ...
        'Units', 'normalized', 'VerticalAlignment', 'top');
end

xlabel('Mean RT of group (ms)');
ylabel(sprintf('Slope (%d to %d ms)', t_start, t_end));
title('Session-wise slope summary');
box off;
fig_setting();

sgtitle(titlestring);

%% -------------------------------
% Print key stats
%% -------------------------------
fprintf('\n============================================\n');
fprintf('%s\n', titlestring);
fprintf('Window: %d to %d ms\n', t_start, t_end);
fprintf('Number of sessions: %d\n', nSess);
fprintf('--------------------------------------------\n');
fprintf('Session-level trend test: slope ~ mean RT\n');
fprintf('ttest on session beta vs 0: p = %.4g, t(%d) = %.3f\n', ...
    p_ttest_beta, stats_ttest_beta.df, stats_ttest_beta.tstat);
fprintf('signrank on session beta vs 0: p = %.4g\n', p_signrank_beta);
fprintf('--------------------------------------------\n');
fprintf('Pooled (session x RT-bin) correlation: slope vs mean RT\n');
fprintf('All %d RT bins x %d sessions (n = %d pairs): r = %.4f, p = %.4g\n', ...
    rt_groupno_max, nSess, sum(goodAll), r_pooled_all, p_pooled_all);
if ~isempty(midBins)
    fprintf('Middle %d RT bins (excl. fastest/slowest) x %d sessions (n = %d pairs): r = %.4f, p = %.4g\n', ...
        numel(midBins), nSess, sum(goodMid), r_pooled_mid, p_pooled_mid);
end
fprintf('--------------------------------------------\n');
fprintf('Pseudoreplication checks on the pooled correlation\n');
fprintf('Within-session (demeaned) correlation: r = %.4f, p = %.4g\n', r_within, p_within);
if ~isnan(r_within_mid)
    fprintf('Within-session (demeaned), middle %d RT bins only: r = %.4f, p = %.4g\n', ...
        numel(midBins), r_within_mid, p_within_mid);
end
if ~isempty(lme_intOnly)
    fprintf('Mixed-effects (Slope ~ RT + (1|Session)): beta_RT = %.6g, p = %.4g\n', beta_RT_intOnly, p_RT_intOnly);
end
if ~isempty(lme_slope)
    fprintf('Mixed-effects (Slope ~ RT + (RT|Session)): beta_RT = %.6g, p = %.4g\n', beta_RT_slope, p_RT_slope_lme);
    fprintf('LRT comparing random-intercept vs random-intercept+slope: p = %.4g\n', p_compare_lme);
end
fprintf('--------------------------------------------\n');
for k = 1:rt_groupno_max
    fprintf('Group %d: nSess = %d, meanRT = %.1f ms, meanSlope = %.5f, p_ttest_vs0 = %.4g, p_signrank_vs0 = %.4g\n', ...
        k, n_per_group(k), mean_RTgroup(k), mean_slope(k), p_ttest_group(k), p_signrank_group(k));
end
fprintf('============================================\n\n');

%% -------------------------------
% Save
%% -------------------------------
%{
save([path, datfilename], ...
    'filelist', 'sessname', ...
    't_start', 't_end', ...
    'slope_sess', 'meanRT_sess', ...
    'mean_slope', 'sem_slope', ...
    'mean_RTgroup', 'sem_RTgroup', ...
    'n_per_group', ...
    'trend_beta', 'trend_intercept', 'trend_r', 'trend_p', ...
    'p_ttest_beta', 'p_signrank_beta', 'stats_ttest_beta', ...
    'p_ttest_group', 'p_signrank_group', ...
    'r_pooled_all', 'p_pooled_all', 'r_pooled_mid', 'p_pooled_mid', ...
    'r_within', 'p_within', 'r_within_mid', 'p_within_mid', ...
    'beta_RT_intOnly', 'p_RT_intOnly', 'beta_RT_slope', 'p_RT_slope_lme', 'p_compare_lme', ...
    'common_tvec', 'mean_trace', 'sem_trace');

saveas(gcf, [path, 'figs/', figfilename]);
%}

end