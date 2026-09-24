function RT_project2_WhenCD_slope_bySession_choiceSplit_v2(monkID, filelist, t_start, t_end, doSmooth, smoothWin)
% RT_project2_WhenCD_slope_bySession_choiceSplit_v2
%
% choice0 = right/up; choice 1 = left/dn
%
% Compute session-wise slopes of When projection (S_d) over a specified time window,
% separately for RT group and choice, then summarize across sessions as mean +/- SEM.
%
% Each saved WhenProjection file (as produced by RT_project2_WhenCD_FAST.m)
% supplies S_d (dots-aligned projection), rt_group, and choice_all for one
% session; this script fits a linear slope of S_d vs. time within
% [t_start, t_end] for each session x RT-group x choice cell.
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
%   - Uses ONLY saved S_d projection (nascent weights).
%   - Splits by RT group and by choice_all (0 or 1).
%   - Uses simple linear least-squares fit: y = intercept + slope * t
%   - Main session-level stat: trend of slope vs mean RT, separately by choice.
%   - Also compares trend beta between choices across sessions.
%   - Also computes a pooled (session x RT-bin) correlation between slope
%     and mean RT, per choice, with all RT bins and with the
%     fastest/slowest RT bin excluded.
%   - Because points from the same session are not independent, also
%     assesses whether that pooled correlation is inflated by
%     pseudoreplication, per choice: (1) a within-session demeaned
%     correlation (all bins, and mid bins only), and (2) a linear
%     mixed-effects model with Session as a random effect.
%
% EXAMPLE
%   RT_project2_WhenCD_slope_bySession_choiceSplit_v2('H', filelist, 200, 500, 0, [])

%% paths
path = ['~/So2026/NeuralData/output/WhenProjections/'];

%% titles / filenames
if monkID=='H'
    titlestring = sprintf('When slope across sessions by choice (Harry), %d to %d ms', t_start, t_end);
    figfilename = sprintf('H_WhenSlope_bySession_choiceSplit_%dto%d.fig', t_start, t_end);
    datfilename = sprintf('H_WhenSlope_bySession_choiceSplit_%dto%d.mat', t_start, t_end);
elseif monkID=='N'
    titlestring = sprintf('When slope across sessions by choice (Neptune), %d to %d ms', t_start, t_end);
    figfilename = sprintf('N_WhenSlope_bySession_choiceSplit_%dto%d.fig', t_start, t_end);
    datfilename = sprintf('N_WhenSlope_bySession_choiceSplit_%dto%d.mat', t_start, t_end);
elseif monkID=='A'
    titlestring = sprintf('When slope across sessions by choice (Both monkeys), %d to %d ms', t_start, t_end);
    figfilename = sprintf('All_WhenSlope_bySession_choiceSplit_%dto%d.fig', t_start, t_end);
    datfilename = sprintf('All_WhenSlope_bySession_choiceSplit_%dto%d.mat', t_start, t_end);
else
    error('monkID must be ''H'', ''N'', or ''A''.');
end

%% parameters
ColorMapRT = [0 0 1; 0.1 0.3 1; 0.1 0.6 1; 0.1 0.9 1; 0.5 1 1; 0.9 1 1];
markerByChoice = {'o','^'};
lineByChoice = {'-','--'};
choiceLabel = {'Choice 0','Choice 1'};

nSess = length(filelist);
rt_groupno_max = 0;

slope_sess_cell = cell(nSess,1);      % each cell: rt_group x 2
meanRT_sess_cell = cell(nSess,1);     % each cell: rt_group x 2
meanTrace_sess_cell = cell(nSess,1);  % each cell: time x rt_group x 2
tvec_sess_cell = cell(nSess,1);
sessname = cell(nSess,1);

trend_beta = nan(nSess,2);        % per session, per choice
trend_intercept = nan(nSess,2);
trend_r = nan(nSess,2);
trend_p = nan(nSess,2);

%% loop through sessions
for ii = 1:nSess

    D = load([path, filelist{ii}], 'S_d', 'rt_group', 'rt_all', 'tvec_d', 'useNorm', 'choice_all');

    if ~isfield(D,'S_d') || ~isfield(D,'rt_group') || ~isfield(D,'rt_all') || ...
       ~isfield(D,'tvec_d') || ~isfield(D,'choice_all')
        error('File %s missing required variables.', filelist{ii});
    end

    S_d = D.S_d;                   % time x trial
    rt_group = D.rt_group(:);      % trial x 1
    rt_all = D.rt_all(:);          % trial x 1
    tvec_d = D.tvec_d(:);          % time x 1
    choice_all = D.choice_all(:);  % trial x 1

    sessname{ii} = filelist{ii};

    if isfield(D,'useNorm')
        if ~D.useNorm
            warning('File %s has useNorm == false.', filelist{ii});
        end
    end

    idx_win = find(tvec_d >= t_start & tvec_d <= t_end);
    if numel(idx_win) < 2
        error('Slope window [%d, %d] has fewer than 2 bins in session %s.', t_start, t_end, filelist{ii});
    end

    t_fit = tvec_d(idx_win);
    rt_groupno = max(rt_group);
    rt_groupno_max = max(rt_groupno_max, rt_groupno);

    slope_this = nan(rt_groupno, 2);
    meanRT_this = nan(rt_groupno, 2);
    meanTrace_this = nan(length(tvec_d), rt_groupno, 2);

    for k = 1:rt_groupno
        for ch = 0:1
            idx = (rt_group == k) & (choice_all == ch);

            if sum(idx) < 3
                continue;
            end

            y = nanmean(S_d(:,idx), 2);
            % ---- smoothing (NEW) ----
            if doSmooth
                y = smoothdata(y, 'gaussian', smoothWin);
            end
            meanTrace_this(:,k,ch+1) = y;
            meanRT_this(k,ch+1) = nanmean(rt_all(idx));

            y_fit = y(idx_win);
            good = ~isnan(t_fit) & ~isnan(y_fit);

            if sum(good) >= 2
                X = [ones(sum(good),1), t_fit(good)];
                b = X \ y_fit(good);
                slope_this(k,ch+1) = b(2);
            end
        end
    end

    slope_sess_cell{ii} = slope_this;
    meanRT_sess_cell{ii} = meanRT_this;
    meanTrace_sess_cell{ii} = meanTrace_this;
    tvec_sess_cell{ii} = tvec_d;

    % session-level trend: slope vs meanRT, separately by choice
    for ch = 1:2
        X = meanRT_this(:,ch);
        Y = slope_this(:,ch);
        goodTrend = ~isnan(X) & ~isnan(Y);

        if sum(goodTrend) >= 2
            Xg = X(goodTrend);
            Yg = Y(goodTrend);

            Xmat = [ones(sum(goodTrend),1), Xg];
            b = Xmat \ Yg;

            trend_intercept(ii,ch) = b(1);
            trend_beta(ii,ch) = b(2);

            [Rtmp, Ptmp] = corr(Xg, Yg, 'type', 'Pearson');
            trend_r(ii,ch) = Rtmp;
            trend_p(ii,ch) = Ptmp;
        end
    end
end

%% convert cells to padded arrays
slope_sess = nan(nSess, rt_groupno_max, 2);
meanRT_sess = nan(nSess, rt_groupno_max, 2);

for ii = 1:nSess
    tmp1 = slope_sess_cell{ii};
    tmp2 = meanRT_sess_cell{ii};

    slope_sess(ii,1:size(tmp1,1),:) = tmp1;
    meanRT_sess(ii,1:size(tmp2,1),:) = tmp2;
end

%% -------------------------------
% Pooled (session x RT-bin) correlation between slope and mean RT, per choice
%% -------------------------------
% Each session contributes up to rt_groupno_max (meanRT, slope) pairs per
% choice; pooling across sessions gives many more points (nSess x
% rt_groupno_max) than the 6-point across-session group-mean regression
% computed in the plotting section below.

if rt_groupno_max >= 3
    midBins = 2:(rt_groupno_max-1);
else
    midBins = [];
end

r_pooled_all = nan(1,2);
p_pooled_all = nan(1,2);
r_pooled_mid = nan(1,2);
p_pooled_mid = nan(1,2);

for ch = 1:2
    X_all = meanRT_sess(:,:,ch);
    Y_all = slope_sess(:,:,ch);
    x_all = X_all(:);
    y_all = Y_all(:);
    goodAll = ~isnan(x_all) & ~isnan(y_all);
    if sum(goodAll) >= 3
        [r_pooled_all(ch), p_pooled_all(ch)] = corr(x_all(goodAll), y_all(goodAll), 'type', 'Pearson');
    end

    if ~isempty(midBins)
        X_mid = meanRT_sess(:,midBins,ch);
        Y_mid = slope_sess(:,midBins,ch);
        x_mid = X_mid(:);
        y_mid = Y_mid(:);
        goodMid = ~isnan(x_mid) & ~isnan(y_mid);
        if sum(goodMid) >= 3
            [r_pooled_mid(ch), p_pooled_mid(ch)] = corr(x_mid(goodMid), y_mid(goodMid), 'type', 'Pearson');
        end
    end
end

%% -------------------------------
% Pseudoreplication checks for the pooled correlation, per choice
%% -------------------------------
% (1) Within-session demeaning: subtract each session's own mean RT and
% mean slope (per choice) before pooling. If the raw pooled correlation
% above is driven mainly by between-session differences, this
% within-session correlation should collapse toward zero.
X_dm = meanRT_sess - nanmean(meanRT_sess, 2);
Y_dm = slope_sess - nanmean(slope_sess, 2);

r_within = nan(1,2);
p_within = nan(1,2);
r_within_mid = nan(1,2);
p_within_mid = nan(1,2);

for ch = 1:2
    x_dm = X_dm(:,:,ch);
    y_dm = Y_dm(:,:,ch);
    x_dm = x_dm(:);
    y_dm = y_dm(:);
    goodDM = ~isnan(x_dm) & ~isnan(y_dm);
    if sum(goodDM) >= 3
        [r_within(ch), p_within(ch)] = corr(x_dm(goodDM), y_dm(goodDM), 'type', 'Pearson');
    end

    if ~isempty(midBins)
        x_dm_mid = X_dm(:,midBins,ch);
        y_dm_mid = Y_dm(:,midBins,ch);
        x_dm_mid = x_dm_mid(:);
        y_dm_mid = y_dm_mid(:);
        goodDMmid = ~isnan(x_dm_mid) & ~isnan(y_dm_mid);
        if sum(goodDMmid) >= 3
            [r_within_mid(ch), p_within_mid(ch)] = corr(x_dm_mid(goodDMmid), y_dm_mid(goodDMmid), 'type', 'Pearson');
        end
    end
end

% (2) Linear mixed-effects model: Session as a random effect, per choice.
lme_intOnly = cell(1,2);
lme_slope = cell(1,2);
beta_RT_intOnly = nan(1,2);
p_RT_intOnly = nan(1,2);
beta_RT_slope = nan(1,2);
p_RT_slope_lme = nan(1,2);
p_compare_lme = nan(1,2);

sess_id = repmat((1:nSess)', 1, rt_groupno_max);
sess_id = sess_id(:);

for ch = 1:2
    RTvec = reshape(meanRT_sess(:,:,ch), [], 1);
    slopevec = reshape(slope_sess(:,:,ch), [], 1);
    goodLME = ~isnan(RTvec) & ~isnan(slopevec);

    if sum(goodLME) >= 3
        tbl = table(sess_id(goodLME), RTvec(goodLME), slopevec(goodLME), ...
            'VariableNames', {'Session','RT','Slope'});
        tbl.Session = categorical(tbl.Session);

        lme_intOnly{ch} = fitlme(tbl, 'Slope ~ RT + (1|Session)');
        isRT = strcmp(lme_intOnly{ch}.Coefficients.Name, 'RT');
        beta_RT_intOnly(ch) = lme_intOnly{ch}.Coefficients.Estimate(isRT);
        p_RT_intOnly(ch) = lme_intOnly{ch}.Coefficients.pValue(isRT);

        try
            lme_slope{ch} = fitlme(tbl, 'Slope ~ RT + (RT|Session)');
            isRT = strcmp(lme_slope{ch}.Coefficients.Name, 'RT');
            beta_RT_slope(ch) = lme_slope{ch}.Coefficients.Estimate(isRT);
            p_RT_slope_lme(ch) = lme_slope{ch}.Coefficients.pValue(isRT);

            comp = compare(lme_intOnly{ch}, lme_slope{ch});
            p_compare_lme(ch) = comp.pValue(2);
        catch ME
            warning('Random-slope LME did not fit for %s (%s); reporting random-intercept-only model.', ...
                choiceLabel{ch}, ME.message);
        end
    end
end

%% summary across sessions
mean_slope = squeeze(nanmean(slope_sess, 1));  % rt_group x 2
sem_slope = squeeze(nanstd(slope_sess, 0, 1) ./ sqrt(sum(~isnan(slope_sess), 1)));

mean_RTgroup = squeeze(nanmean(meanRT_sess, 1));
sem_RTgroup = squeeze(nanstd(meanRT_sess, 0, 1) ./ sqrt(sum(~isnan(meanRT_sess), 1)));

n_per_group = squeeze(sum(~isnan(slope_sess), 1)); % rt_group x 2

%% stats across sessions
% trend beta vs zero for each choice
p_ttest_beta = nan(1,2);
p_signrank_beta = nan(1,2);
stats_ttest_beta = cell(1,2);

for ch = 1:2
    good_beta = ~isnan(trend_beta(:,ch));
    if sum(good_beta) >= 2
        [~, p_ttest_beta(ch), ~, stats_tmp] = ttest(trend_beta(good_beta,ch), 0);
        p_signrank_beta(ch) = signrank(trend_beta(good_beta,ch), 0);
        stats_ttest_beta{ch} = stats_tmp;
    else
        stats_ttest_beta{ch} = struct('tstat', NaN, 'df', NaN);
    end
end

% paired difference in trend beta between choices
good_pair = ~isnan(trend_beta(:,1)) & ~isnan(trend_beta(:,2));
if sum(good_pair) >= 2
    [~, p_ttest_beta_choiceDiff, ~, stats_ttest_beta_choiceDiff] = ...
        ttest(trend_beta(good_pair,1), trend_beta(good_pair,2));
    p_signrank_beta_choiceDiff = signrank(trend_beta(good_pair,1), trend_beta(good_pair,2));
else
    p_ttest_beta_choiceDiff = NaN;
    p_signrank_beta_choiceDiff = NaN;
    stats_ttest_beta_choiceDiff = struct('tstat', NaN, 'df', NaN);
end

%% grand-average traces across sessions for display
binwidth = 25;
t_min = -200;
t_max = 1500;
common_tvec = t_min:binwidth:t_max;

common_trace_sess = nan(length(common_tvec), rt_groupno_max, 2, nSess);

for ii = 1:nSess
    tvec_d = tvec_sess_cell{ii};
    meanTrace_this = meanTrace_sess_cell{ii};

    [~, idx_common, idx_local] = intersect(common_tvec, tvec_d);

    for k = 1:size(meanTrace_this,2)
        for ch = 1:2
            common_trace_sess(idx_common, k, ch, ii) = meanTrace_this(idx_local, k, ch);
        end
    end
end

keep_t = squeeze(any(any(any(~isnan(common_trace_sess), 2), 3), 4));
common_tvec = common_tvec(keep_t);
common_trace_sess = common_trace_sess(keep_t,:,:,:);

mean_trace = nanmean(common_trace_sess, 4); % time x rt_group x choice
sem_trace = nanstd(common_trace_sess, 0, 4) ./ sqrt(sum(~isnan(common_trace_sess), 4));

%% plot
figure; clf;

% Panel 1: traces
subplot(1,2,1); hold on;

ymin = min(mean_trace(:) - sem_trace(:));
ymax = max(mean_trace(:) + sem_trace(:));
if isempty(ymin) || isnan(ymin), ymin = -1; end
if isempty(ymax) || isnan(ymax), ymax = 1; end

patch([t_start t_end t_end t_start], [ymin ymin ymax ymax], ...
      [0.92 0.92 0.92], 'EdgeColor', 'none', 'FaceAlpha', 0.35);
line([0 0], [ymin ymax], 'Color', 'k');

for ch = 1:2
    for k = 1:rt_groupno_max
        y = mean_trace(:,k,ch);
        e = sem_trace(:,k,ch);

        if all(isnan(y))
            continue;
        end

        plot(common_tvec, y, 'Color', ColorMapRT(k,:), ...
            'LineWidth', 1.7, 'LineStyle', lineByChoice{ch});

        x = common_tvec(:);
        good = ~isnan(x) & ~isnan(y) & ~isnan(e);
        if any(good)
            xx = [x(good); flipud(x(good))];
            yy = [y(good)-e(good); flipud(y(good)+e(good))];
            patch(xx, yy, ColorMapRT(k,:), 'FaceAlpha', 0.10, 'EdgeColor', 'none');
        end
    end
end

xlabel('Time from dots (ms)');
ylabel('When projection');
title('Session-averaged projections by choice');
ylim([ymin ymax]);
box off;
fig_setting();

% Panel 2: slope summary
subplot(1,2,2); hold on;

for ch = 1:2
    for k = 1:rt_groupno_max
        if isnan(mean_slope(k,ch)) || isnan(mean_RTgroup(k,ch))
            continue;
        end

        line([mean_RTgroup(k,ch)-sem_RTgroup(k,ch), mean_RTgroup(k,ch)+sem_RTgroup(k,ch)], ...
             [mean_slope(k,ch), mean_slope(k,ch)], ...
             'Color', ColorMapRT(k,:), 'LineWidth', 1.2, 'LineStyle', lineByChoice{ch});

        line([mean_RTgroup(k,ch), mean_RTgroup(k,ch)], ...
             [mean_slope(k,ch)-sem_slope(k,ch), mean_slope(k,ch)+sem_slope(k,ch)], ...
             'Color', ColorMapRT(k,:), 'LineWidth', 1.2, 'LineStyle', lineByChoice{ch});

        plot(mean_RTgroup(k,ch), mean_slope(k,ch), markerByChoice{ch}, ...
            'MarkerSize', 8, ...
            'MarkerFaceColor', ColorMapRT(k,:), ...
            'MarkerEdgeColor', ColorMapRT(k,:));
    end

    % ---- regression line through the group means ----
    goodPlot = ~isnan(mean_RTgroup(:,ch)) & ~isnan(mean_slope(:,ch));

    if sum(goodPlot) >= 2
        xfit = mean_RTgroup(goodPlot,ch);
        yfit = mean_slope(goodPlot,ch);

        Xmat = [ones(sum(goodPlot),1), xfit];
        b = Xmat \ yfit;

        xline = linspace(min(xfit), max(xfit), 100)';
        yline = [ones(size(xline)), xline] * b;

        plot(xline, yline, 'k', 'LineWidth', 1.5, 'LineStyle', lineByChoice{ch});

        % optional descriptive printout
        [r_tmp, p_tmp] = corr(xfit, yfit, 'type', 'Pearson');
        fprintf('Mean-point regression: intercept = %.5f, beta = %.5f, r = %.4f, p = %.4g\n', ...
            b(1), b(2), r_tmp, p_tmp);
    end
end

text(0.05, 0.95, sprintf('Within-session (demeaned): %s r=%.2f p=%.3f; %s r=%.2f p=%.3f', ...
    choiceLabel{1}, r_within(1), p_within(1), choiceLabel{2}, r_within(2), p_within(2)), ...
    'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 8);
if ~isempty(midBins)
    text(0.05, 0.87, sprintf('Demeaned (mid bins): %s r=%.2f p=%.3f; %s r=%.2f p=%.3f', ...
        choiceLabel{1}, r_within_mid(1), p_within_mid(1), choiceLabel{2}, r_within_mid(2), p_within_mid(2)), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 8);
end

xlabel('Mean RT of group (ms)');
ylabel(sprintf('Slope (%d to %d ms)', t_start, t_end));
title('Session-wise slope summary by choice');
box off;
fig_setting();

legend({'Choice 0','Choice 1'}, 'Location', 'best');

sgtitle(titlestring);

%% print stats
fprintf('\n====================================================\n');
fprintf('%s\n', titlestring);
fprintf('Window: %d to %d ms\n', t_start, t_end);
fprintf('Number of sessions: %d\n', nSess);
fprintf('----------------------------------------------------\n');

for ch = 1:2
    fprintf('%s: trend beta vs 0\n', choiceLabel{ch});
    fprintf('ttest: p = %.4g, t(%d) = %.3f\n', ...
        p_ttest_beta(ch), stats_ttest_beta{ch}.df, stats_ttest_beta{ch}.tstat);
    fprintf('signrank: p = %.4g\n', p_signrank_beta(ch));
end

fprintf('----------------------------------------------------\n');
fprintf('Choice difference in session-level trend beta\n');
fprintf('paired ttest: p = %.4g, t(%d) = %.3f\n', ...
    p_ttest_beta_choiceDiff, stats_ttest_beta_choiceDiff.df, stats_ttest_beta_choiceDiff.tstat);
fprintf('paired signrank: p = %.4g\n', p_signrank_beta_choiceDiff);

fprintf('----------------------------------------------------\n');
fprintf('Pooled (session x RT-bin) correlation: slope vs mean RT\n');
for ch = 1:2
    fprintf('%s: all %d RT bins x %d sessions: r = %.4f, p = %.4g\n', ...
        choiceLabel{ch}, rt_groupno_max, nSess, r_pooled_all(ch), p_pooled_all(ch));
    if ~isempty(midBins)
        fprintf('%s: middle %d RT bins (excl. fastest/slowest) x %d sessions: r = %.4f, p = %.4g\n', ...
            choiceLabel{ch}, numel(midBins), nSess, r_pooled_mid(ch), p_pooled_mid(ch));
    end
end

fprintf('----------------------------------------------------\n');
fprintf('Pseudoreplication checks on the pooled correlation\n');
for ch = 1:2
    fprintf('%s: within-session (demeaned): r = %.4f, p = %.4g\n', ...
        choiceLabel{ch}, r_within(ch), p_within(ch));
    if ~isempty(midBins)
        fprintf('%s: within-session (demeaned), middle bins only: r = %.4f, p = %.4g\n', ...
            choiceLabel{ch}, r_within_mid(ch), p_within_mid(ch));
    end
    if ~isempty(lme_intOnly{ch})
        fprintf('%s: mixed-effects (Slope ~ RT + (1|Session)): beta_RT = %.6g, p = %.4g\n', ...
            choiceLabel{ch}, beta_RT_intOnly(ch), p_RT_intOnly(ch));
    end
    if ~isempty(lme_slope{ch})
        fprintf('%s: mixed-effects (Slope ~ RT + (RT|Session)): beta_RT = %.6g, p = %.4g\n', ...
            choiceLabel{ch}, beta_RT_slope(ch), p_RT_slope_lme(ch));
        fprintf('%s: LRT comparing random-intercept vs random-intercept+slope: p = %.4g\n', ...
            choiceLabel{ch}, p_compare_lme(ch));
    end
end

fprintf('----------------------------------------------------\n');
for ch = 1:2
    for k = 1:rt_groupno_max
        fprintf('Choice %d, Group %d: nSess = %d, meanRT = %.1f ms, meanSlope = %.5f\n', ...
            ch-1, k, n_per_group(k,ch), mean_RTgroup(k,ch), mean_slope(k,ch));
    end
end
fprintf('====================================================\n\n');

%% save
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
    'p_ttest_beta_choiceDiff', 'p_signrank_beta_choiceDiff', 'stats_ttest_beta_choiceDiff', ...
    'r_pooled_all', 'p_pooled_all', 'r_pooled_mid', 'p_pooled_mid', ...
    'r_within', 'p_within', 'r_within_mid', 'p_within_mid', ...
    'beta_RT_intOnly', 'p_RT_intOnly', 'beta_RT_slope', 'p_RT_slope_lme', 'p_compare_lme', ...
    'common_tvec', 'mean_trace', 'sem_trace');

saveas(gcf, [path, 'figs/', figfilename]);
%}
end