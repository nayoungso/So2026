function RT_project2_WhatCD_buildup_bySession_withTrace_v2(filelist, useSaccWhat, t_start, t_end, doSmooth, smoothWin)
% RT_project2_WhatCD_buildup_bySession_withTrace_v2
%
% Compute session-wise motion-consistent buildup rate of What projection
% (S_d_dec or S_d_choice, saved by RT_project2_WhatCD_FAST.m) vs
% unsigned coherence, separately by motion sign (negative/positive),
% then summarize across sessions as mean +/- SEM. Also plots the
% session-averaged projection traces themselves (left panel) alongside
% the buildup-rate summary (right panel).
%
% INPUTS
%   filelist    : cell array of saved WhatProjection filenames.
%   useSaccWhat : logical; if true, uses S_d_choice (saccade-trained
%                 CD); if false, uses S_d_dec (dots-trained CD).
%   t_start, t_end : dots-aligned time window (ms) over which each
%                 session's buildup rate (linear slope of the
%                 projection vs. time) is fit.
%   doSmooth    : 0/1, smooth the mean trace (left panel) with a
%                 Gaussian window before display.
%   smoothWin   : smoothdata window size (samples), used only if
%                 doSmooth.
%
% OUTPUT
%   None returned (figure only).
%
% REQUIRES ON PATH: fig_setting.m (included alongside this file).
%
% NOTES
%   - Also computes a within-session demeaned correlation between
%     buildup rate and |coherence|, per sign, pooling all sessions x
%     coherence levels (many more points than the coh_groupno_max-point
%     across-session group-mean regression). This avoids inflating
%     significance from between-session differences (pseudoreplication).
%   - Same demeaned correlation is also computed excluding the weakest
%     and strongest |coherence| level, as a robustness control.
%
% EXAMPLE
%   RT_project2_WhatCD_buildup_bySession_withTrace_v2(filelist, false, 200, 500, 0, [])

path = ['~/So2026/NeuralData/output/WhatProjections/'];

% rows 1:6   = negative coherence, strong -> weak
% rows 7:12  = positive coherence, weak -> strong
ColorMapCoh = [0 0.5 0; 0 0.7 0.3; 0.4 0.8 0.6; 0.4 0.8 0.6; 0.6 0.9 0.8; 0.8 1 0.8; ...
               1 1 0.8; 1 1 0.6; 1 0.9 0.5; 1 0.8 0.4; 1 0.6 0; 1 0 0];

markerBySign = {'o','^'};   % negative / positive motion coherence
lineBySign   = {'-','-'};
lineColorBySign = [0 0.5 0; 1 0 0];

nSess = length(filelist);

%% determine all |coherence| values across sessions
coh_mag_allSessions = [];

for ii = 1:nSess
    D = load([path, filelist{ii}], 'coh_all');
    coh_vals = unique(abs(D.coh_all(:))) / 10;
    coh_vals = coh_vals(~isnan(coh_vals) & coh_vals > 0);
    coh_mag_allSessions = union(coh_mag_allSessions, coh_vals);
end

coh_mag_allSessions = sort(coh_mag_allSessions(:))';   % weak -> strong
coh_groupno_max = length(coh_mag_allSessions);

if coh_groupno_max > 6
    warning('More than 6 coherence magnitudes found. Color map assumes up to 6 magnitudes.');
end

%% storage
slope_raw_sess   = nan(nSess, coh_groupno_max, 2);   % raw slope
buildup_sess     = nan(nSess, coh_groupno_max, 2);   % motion-consistent buildup
meanCoh_sess     = nan(nSess, coh_groupno_max, 2);   % signed coherence
meanAbsCoh_sess  = nan(nSess, coh_groupno_max, 2);   % unsigned coherence
trend_beta       = nan(nSess, 2);                    % trend of buildup vs |coh|

% for left-panel traces
meanTrace_sess_cell = cell(nSess,1);
tvec_sess_cell = cell(nSess,1);

%% main loop
for ii = 1:nSess

    D = load([path, filelist{ii}], 'S_d_dec','S_d_choice','coh_all','tvec_d');

    if useSaccWhat
        S_d = D.S_d_choice;
    else
        S_d = D.S_d_dec;
    end

    coh_all = D.coh_all(:) / 10;
    tvec_d  = D.tvec_d(:);

    idx_win = find(tvec_d >= t_start & tvec_d <= t_end);
    if numel(idx_win) < 2
        error('Slope window [%d, %d] has fewer than 2 bins in file %s.', t_start, t_end, filelist{ii});
    end
    t_fit = tvec_d(idx_win);

    meanTrace_this = nan(length(tvec_d), coh_groupno_max, 2);

    for k = 1:coh_groupno_max
        coh_mag = coh_mag_allSessions(k);

        for signVal = [-1, 1]

            if signVal == -1
                signIdx = 1;
            else
                signIdx = 2;
            end

            targetCoh = signVal * coh_mag;

            tol = 1e-8;
            idx = abs(coh_all - targetCoh) < tol;

            if sum(idx) < 3
                continue;
            end

            y = nanmean(S_d(:,idx), 2);
            % ---- smoothing (NEW) ----
            if doSmooth
                y = smoothdata(y, 'gaussian', smoothWin);
            end
            meanTrace_this(:,k,signIdx) = y;

            meanCoh_sess(ii,k,signIdx)    = nanmean(coh_all(idx));
            meanAbsCoh_sess(ii,k,signIdx) = nanmean(abs(coh_all(idx)));

            y_fit = y(idx_win);
            good = ~isnan(t_fit) & ~isnan(y_fit);

            if sum(good) >= 2
                X = [ones(sum(good),1), t_fit(good)];
                b = X \ y_fit(good);
                rawSlope = b(2);

                slope_raw_sess(ii,k,signIdx) = rawSlope;

                % motion-consistent buildup:
                % negative coherence -> keep raw slope
                % positive coherence -> flip raw slope
                buildup_sess(ii,k,signIdx) = -signVal * rawSlope;
            end
        end
    end

    meanTrace_sess_cell{ii} = meanTrace_this;
    tvec_sess_cell{ii} = tvec_d;

    % session-level trend of motion-consistent buildup vs unsigned coherence
    for signIdx = 1:2
        X = squeeze(meanAbsCoh_sess(ii,:,signIdx));
        Y = squeeze(buildup_sess(ii,:,signIdx));

        X = X(:);
        Y = Y(:);

        good = ~isnan(X) & ~isnan(Y);

        if sum(good) >= 2
            Xg = X(good);
            Yg = Y(good);

            Xmat = [ones(length(Xg),1), Xg];
            b = Xmat \ Yg;
            trend_beta(ii,signIdx) = b(2);
        end
    end
end

%% summary
mean_buildup = squeeze(nanmean(buildup_sess, 1));
sem_buildup  = squeeze(nanstd(buildup_sess, 0, 1) ./ sqrt(sum(~isnan(buildup_sess), 1)));

mean_abscoh = squeeze(nanmean(meanAbsCoh_sess, 1));
sem_abscoh  = squeeze(nanstd(meanAbsCoh_sess, 0, 1) ./ sqrt(sum(~isnan(meanAbsCoh_sess), 1)));

mean_coh = squeeze(nanmean(meanCoh_sess, 1));
n_per_group = squeeze(sum(~isnan(buildup_sess), 1));

%% stats
p_ttest_beta = nan(1,2);
p_signrank_beta = nan(1,2);
stats_ttest_beta = cell(1,2);

for signIdx = 1:2
    good_beta = ~isnan(trend_beta(:,signIdx));
    if sum(good_beta) >= 2
        [~, p_ttest_beta(signIdx), ~, stats_tmp] = ttest(trend_beta(good_beta,signIdx), 0);
        p_signrank_beta(signIdx) = signrank(trend_beta(good_beta,signIdx), 0);
        stats_ttest_beta{signIdx} = stats_tmp;
    else
        stats_ttest_beta{signIdx} = struct('tstat', NaN, 'df', NaN);
    end
end

good_pair = ~isnan(trend_beta(:,1)) & ~isnan(trend_beta(:,2));
if sum(good_pair) >= 2
    [~, p_ttest_beta_signDiff, ~, stats_ttest_beta_signDiff] = ...
        ttest(trend_beta(good_pair,1), trend_beta(good_pair,2));
    p_signrank_beta_signDiff = signrank(trend_beta(good_pair,1), trend_beta(good_pair,2));
else
    p_ttest_beta_signDiff = NaN;
    p_signrank_beta_signDiff = NaN;
    stats_ttest_beta_signDiff = struct('tstat', NaN, 'df', NaN);
end

%% -------------------------------
% Within-session demeaned correlation: buildup vs |coherence|, per sign
%% -------------------------------
% Subtract each session's own mean |coherence| and mean buildup (across
% coherence levels) before pooling, so the correlation reflects the
% within-session relationship rather than being inflated by
% between-session differences (pseudoreplication).
X_dm = meanAbsCoh_sess - nanmean(meanAbsCoh_sess, 2);
Y_dm = buildup_sess - nanmean(buildup_sess, 2);

if coh_groupno_max >= 3
    midCohBins = 2:(coh_groupno_max-1);
else
    midCohBins = [];
end

r_within = nan(1,2);
p_within = nan(1,2);
r_within_mid = nan(1,2);
p_within_mid = nan(1,2);

for signIdx = 1:2
    x_dm = X_dm(:,:,signIdx);
    y_dm = Y_dm(:,:,signIdx);
    x_dm = x_dm(:);
    y_dm = y_dm(:);
    goodDM = ~isnan(x_dm) & ~isnan(y_dm);
    if sum(goodDM) >= 3
        [r_within(signIdx), p_within(signIdx)] = corr(x_dm(goodDM), y_dm(goodDM), 'type', 'Pearson');
    end

    if ~isempty(midCohBins)
        x_dm_mid = X_dm(:,midCohBins,signIdx);
        y_dm_mid = Y_dm(:,midCohBins,signIdx);
        x_dm_mid = x_dm_mid(:);
        y_dm_mid = y_dm_mid(:);
        goodDMmid = ~isnan(x_dm_mid) & ~isnan(y_dm_mid);
        if sum(goodDMmid) >= 3
            [r_within_mid(signIdx), p_within_mid(signIdx)] = corr(x_dm_mid(goodDMmid), y_dm_mid(goodDMmid), 'type', 'Pearson');
        end
    end
end

%% grand-average traces across sessions for display
binwidth = 25;
t_min = -200;
t_max = 1500;
common_tvec = t_min:binwidth:t_max;

common_trace_sess = nan(length(common_tvec), coh_groupno_max, 2, nSess);

for ii = 1:nSess
    tvec_d = tvec_sess_cell{ii};
    meanTrace_this = meanTrace_sess_cell{ii};

    [~, idx_common, idx_local] = intersect(common_tvec, tvec_d);

    for k = 1:size(meanTrace_this,2)
        for signIdx = 1:2
            common_trace_sess(idx_common, k, signIdx, ii) = meanTrace_this(idx_local, k, signIdx);
        end
    end
end

keep_t = squeeze(any(any(any(~isnan(common_trace_sess),2),3),4));
common_tvec = common_tvec(keep_t);
common_trace_sess = common_trace_sess(keep_t,:,:,:);

mean_trace = nanmean(common_trace_sess, 4); % time x coh_group x sign
sem_trace = nanstd(common_trace_sess, 0, 4) ./ sqrt(sum(~isnan(common_trace_sess), 4));

%% plot
figure; clf;

% ---- Panel 1: session-averaged projections ----
subplot(1,2,1); hold on;

ymin = min(mean_trace(:) - sem_trace(:));
ymax = max(mean_trace(:) + sem_trace(:));
if isempty(ymin) || isnan(ymin), ymin = -1; end
if isempty(ymax) || isnan(ymax), ymax = 1; end

patch([t_start t_end t_end t_start], [ymin ymin ymax ymax], ...
      [0.92 0.92 0.92], 'EdgeColor', 'none', 'FaceAlpha', 0.35);
line([0 0], [ymin ymax], 'Color', 'k');

for signIdx = 1:2
    for k = 1:coh_groupno_max

        y = mean_trace(:,k,signIdx);
        e = sem_trace(:,k,signIdx);

        if all(isnan(y))
            continue;
        end

        % color choice
        if signIdx == 1
            colorIdx = coh_groupno_max - k + 1; % negative: strong -> weak
        else
            colorIdx = 6 + k;                   % positive: weak -> strong
        end
        thisColor = ColorMapCoh(colorIdx,:);

        plot(common_tvec, y, 'Color', thisColor, 'LineWidth', 1.8, 'LineStyle', lineBySign{signIdx});

        x = common_tvec(:);
        good = ~isnan(x) & ~isnan(y) & ~isnan(e);
        if any(good)
            xx = [x(good); flipud(x(good))];
            yy = [y(good)-e(good); flipud(y(good)+e(good))];
            patch(xx, yy, thisColor, 'FaceAlpha', 0.12, 'EdgeColor', 'none');
        end
    end
end

xlabel('Time from dots (ms)');
ylabel('What projection');
title('Session-averaged projections');
ylim([ymin ymax]);
box off;
fig_setting();

% ---- Panel 2: buildup summary ----
subplot(1,2,2); hold on;

for signIdx = 1:2
    for k = 1:coh_groupno_max

        if isnan(mean_buildup(k,signIdx)) || isnan(mean_abscoh(k,signIdx))
            continue;
        end

        if signIdx == 1
            colorIdx = coh_groupno_max - k + 1;
        else
            colorIdx = 6 + k;
        end
        thisColor = ColorMapCoh(colorIdx,:);

        % horizontal SEM
        line([mean_abscoh(k,signIdx)-sem_abscoh(k,signIdx), mean_abscoh(k,signIdx)+sem_abscoh(k,signIdx)], ...
             [mean_buildup(k,signIdx), mean_buildup(k,signIdx)], ...
             'Color', thisColor, 'LineWidth', 1.2, 'LineStyle', lineBySign{signIdx});

        % vertical SEM
        line([mean_abscoh(k,signIdx), mean_abscoh(k,signIdx)], ...
             [mean_buildup(k,signIdx)-sem_buildup(k,signIdx), mean_buildup(k,signIdx)+sem_buildup(k,signIdx)], ...
             'Color', thisColor, 'LineWidth', 1.2, 'LineStyle', lineBySign{signIdx});

        plot(mean_abscoh(k,signIdx), mean_buildup(k,signIdx), markerBySign{signIdx}, ...
            'MarkerFaceColor', thisColor, ...
            'MarkerEdgeColor', thisColor, ...
            'MarkerSize', 8);
    end

    % regression line through mean points
    good = ~isnan(mean_abscoh(:,signIdx)) & ~isnan(mean_buildup(:,signIdx));
    if sum(good) >= 2
        xfit = mean_abscoh(good,signIdx);
        yfit = mean_buildup(good,signIdx);

        xfit = xfit(:);
        yfit = yfit(:);

        Xmat = [ones(length(xfit),1), xfit];
        b = Xmat \ yfit;

        xline = linspace(min(xfit), max(xfit), 100)';
        yline = [ones(size(xline)), xline] * b;

        [r_tmp, p_tmp] = corr(xfit, yfit, 'type', 'Pearson');

        fprintf('Mean-point regression (sign %d): beta = %.5f, r = %.4f, p = %.4g\n', ...
            signIdx, b(2), r_tmp, p_tmp);

        plot(xline, yline, 'Color', lineColorBySign(signIdx,:), ...
            'LineStyle', lineBySign{signIdx}, 'LineWidth', 1.5);
    end
end

text(0.05, 0.95, sprintf('Demeaned: neg r=%.2f p=%.3f; pos r=%.2f p=%.3f', ...
    r_within(1), p_within(1), r_within(2), p_within(2)), ...
    'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 8);
if ~isempty(midCohBins)
    text(0.05, 0.87, sprintf('Demeaned (extremes excl.): neg r=%.2f p=%.3f; pos r=%.2f p=%.3f', ...
        r_within_mid(1), p_within_mid(1), r_within_mid(2), p_within_mid(2)), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 8);
end

xlabel('Unsigned coherence');
ylabel(sprintf('Motion-consistent buildup rate (%d to %d ms)', t_start, t_end));
title('Session-wise buildup summary');
box off;
fig_setting();

legend({'negative motion','positive motion'}, 'Location', 'best');

sgtitle('What projection slope / buildup summary');

%% print stats
fprintf('\n====================================================\n');
fprintf('What motion-consistent buildup rate across sessions\n');
fprintf('Window: %d to %d ms\n', t_start, t_end);
fprintf('Number of sessions: %d\n', nSess);
fprintf('----------------------------------------------------\n');

fprintf('Negative motion: trend beta vs 0\n');
fprintf('ttest: p = %.4g, t(%d) = %.3f\n', ...
    p_ttest_beta(1), stats_ttest_beta{1}.df, stats_ttest_beta{1}.tstat);
fprintf('signrank: p = %.4g\n', p_signrank_beta(1));

fprintf('Positive motion: trend beta vs 0\n');
fprintf('ttest: p = %.4g, t(%d) = %.3f\n', ...
    p_ttest_beta(2), stats_ttest_beta{2}.df, stats_ttest_beta{2}.tstat);
fprintf('signrank: p = %.4g\n', p_signrank_beta(2));

fprintf('----------------------------------------------------\n');
fprintf('Difference in session-level trend beta between signs\n');
fprintf('paired ttest: p = %.4g, t(%d) = %.3f\n', ...
    p_ttest_beta_signDiff, stats_ttest_beta_signDiff.df, stats_ttest_beta_signDiff.tstat);
fprintf('paired signrank: p = %.4g\n', p_signrank_beta_signDiff);

fprintf('----------------------------------------------------\n');
fprintf('Within-session (demeaned) correlation: motion-consistent buildup vs |coherence|\n');
fprintf('Negative motion: r = %.4f, p = %.4g\n', r_within(1), p_within(1));
if ~isempty(midCohBins)
    fprintf('Negative motion, weakest/strongest |coherence| excluded: r = %.4f, p = %.4g\n', ...
        r_within_mid(1), p_within_mid(1));
end
fprintf('Positive motion: r = %.4f, p = %.4g\n', r_within(2), p_within(2));
if ~isempty(midCohBins)
    fprintf('Positive motion, weakest/strongest |coherence| excluded: r = %.4f, p = %.4g\n', ...
        r_within_mid(2), p_within_mid(2));
end

fprintf('----------------------------------------------------\n');
for signIdx = 1:2
    for k = 1:coh_groupno_max
        fprintf('Sign %d, |coh| group %d: nSess = %d, meanSignedCoh = %.3f, meanAbsCoh = %.3f, meanBuildup = %.5f\n', ...
            signIdx, k, n_per_group(k,signIdx), mean_coh(k,signIdx), mean_abscoh(k,signIdx), mean_buildup(k,signIdx));
    end
end
fprintf('====================================================\n\n');

end