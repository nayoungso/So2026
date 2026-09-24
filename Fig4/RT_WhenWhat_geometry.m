function RT_WhenWhat_geometry(filelist_when, filelist_what, dec_t, choice_t, topK, nPerm)
% RT_WhenWhat_geometry
%
% Tests whether the "When" and "What" coding directions (WhenCD, saved
% by RT_decoder_when_v4_2026_FAST.m; WhatCD, saved by
% RT_decoder_what_FAST.m at a chosen dots-aligned time dec_t and
% saccade-aligned time choice_t) occupy similar or distinct
% subspaces of population activity, per session, against
% within-session shuffled controls (nPerm permutations):
%
%   (a) Cosine similarity between the norm-normalized WhenCD and each
%       WhatCD (motion-aligned and saccade-aligned), vs. the cosine
%       similarity expected when the When weights are randomly
%       permuted across neurons.
%   (b) Overlap (Jaccard index) between the top topK% of |WhenCD|
%       weights and the top topK% of |WhatCD| weights, vs. shuffled
%       controls that preserve the same number of "active" When
%       neurons but pick them at random.
%   (c) Session-wise scatter of |cosine similarity| vs. overlap, to
%       check whether the two measures agree.
%
% INPUTS
%   filelist_when : cell array of saved WhenCD filenames (as saved by
%                   RT_decoder_when_v4_2026_FAST.m), one per session.
%   filelist_what : cell array of saved WhatCD filenames (as saved by
%                   RT_decoder_what_FAST.m), same order/sessions as
%                   filelist_when.
%   dec_t         : dots-aligned time (ms, must be a value in that
%                   session's searchT1) at which to take the
%                   motion-aligned WhatCD.
%   choice_t      : saccade-aligned time (ms, must be a value in that
%                   session's searchT2) at which to take the
%                   saccade-aligned WhatCD.
%   topK          : percentile threshold (0-100) defining "active"
%                   neurons for the Jaccard overlap (e.g. 50 = top 50%
%                   of nonzero |weights|).
%   nPerm         : number of shuffles per session for both the cosine
%                   and overlap null distributions.
%
% OUTPUT
%   None returned (figure only: 2x3 panels -- cosine similarity
%   histograms (motion, saccade), overlap paired-dot plots (motion,
%   saccade), and cosine-vs-overlap scatter (motion, saccade)).
%
% REQUIRES ON PATH: fig_setting.m (included alongside this file).
%
% EXAMPLE
%   RT_WhenWhat_geometry(filelist_when, filelist_what, 450, -200, 50, 1000)

When_path = ['~/So2026/NeuralData/output/WhenCD/'];
What_path = ['~/So2026/NeuralData/output/WhatCD/'];


nSess = length(filelist_when);

cos_real_d = nan(nSess,1);
cos_real_s = nan(nSess,1);

cos_shuff_d = nan(nSess,nPerm);
cos_shuff_s = nan(nSess,nPerm);

overlap_real_d = nan(nSess,1);
overlap_real_s = nan(nSess,1);

overlap_shuff_d = nan(nSess,nPerm);
overlap_shuff_s = nan(nSess,nPerm);

for i = 1:nSess

    %% -------------------------------
    % LOAD WHEN
    %% -------------------------------
    clear w_when w_what_d w_what_s

    load([When_path, filelist_when{i}], 'whenCD_norm','coef_when');

    w_when = whenCD_norm(:);
    %w_when = coef_when(2:end);


    %% -------------------------------
    % LOAD WHAT
    %% -------------------------------
    load([What_path, filelist_what{i}], ...
        'searchT1','searchT2', 'whatCD_d_norm', 'whatCD_s_norm', ...
        'B_d', 'B_s');

    % pick time windows
    w_what_d = whatCD_d_norm{searchT1==dec_t}(:);
    w_what_s = whatCD_s_norm{searchT2==choice_t}(:);

    % choose one (you can switch or average)
    %w_what = w_what_d;  % <-- motion aligned
    % w_what = w_what_s; % <-- sacc aligned

    %% -------------------------------
    % COSINE SIMILARITY (REAL)
    %% -------------------------------
    %cos_real_d(i) = dot(w_when, w_what_d);
    %cos_real_s(i) = dot(w_when, w_what_s);

    cos_real_d(i) = 1-pdist2(w_when', w_what_d','cosine');
    cos_real_s(i) = 1-pdist2(w_when', w_what_s','cosine');


    
    %% -------------------------------
    % COSINE SHUFFLE (within session)
    %% -------------------------------
    for p = 1:nPerm
        perm = randperm(length(w_when));
        
        %cos_shuff_d(i,p) = dot(w_when(perm), w_what_d);
        %cos_shuff_s(i,p) = dot(w_when(perm), w_what_s);

        cos_shuff_d(i,p) = 1-pdist2(w_when(perm)', w_what_d','cosine');
        cos_shuff_s(i,p) = 1-pdist2(w_when(perm)', w_what_s','cosine');
    end
    
    %% -------------------------------
    % OVERLAP (Jaccard on nonzero weights)
    %% -------------------------------
  
    % ---- define sets ----
    When_nnz = find(w_when ~= 0);        % When active neurons
    What_nnz_d = find(w_what_d ~= 0);    % What (motion) active neurons
    What_nnz_s = find(w_what_s ~= 0);    % What (sacc) active neurons
    
    w_when_nz = abs(w_when(w_when~=0));
    thr_when = prctile(w_when_nz, (100-topK));   % top 50% of the nonzero when weights
    %thr_when = prctile(w_when_nz, 80);   % top 20%
    
    w_what_d_nz = abs(w_what_d(w_what_d~=0));
    w_what_s_nz = abs(w_what_s(w_what_s~=0));
    thr_what_d = prctile(w_what_d_nz, (100-topK));   % top 50% of the nonzero when weights
    thr_what_s = prctile(w_what_s_nz, (100-topK));   % top 50% of the nonzero when weights
    
    when_active = find(abs(w_when) >= thr_when);
    what_d_active = find(abs(w_what_d) >= thr_what_d);
    what_s_active = find(abs(w_what_s) >= thr_what_s);
    
    % ---- real Jaccard ----
    overlap_real_d(i) = length(intersect(when_active, what_d_active)) / length(union(when_active, what_d_active));
    overlap_real_s(i) = length(intersect(when_active, what_s_active)) / length(union(when_active, what_s_active));
    
    %% -------------------------------
    % OVERLAP SHUFFLE (preserve sparsity)
    %% -------------------------------
    
    nA = length(when_active); % active When neurons
    N  = length(w_when);   % total neurons
    
    for p = 1:nPerm
        
        % ---- shuffle When support ----
        When_shuff = randperm(N, nA);
        
        % ---- motion-aligned ----
        overlap_shuff_d(i,p) = length(intersect(When_shuff, what_d_active)) / length(union(When_shuff, what_d_active));
        
        % ---- sacc-aligned ----
        overlap_shuff_s(i,p) = length(intersect(When_shuff, what_s_active)) / length(union(When_shuff, what_s_active));
        
    end

    %{
    %% -------------------------------
    % OVERLAP (top-K)
    %% -------------------------------
    k_neurons = max(1, round(topK * length(w_when)));
    
    [~, idx_when] = sort(abs(w_when), 'descend');
    [~, idx_what_d] = sort(abs(w_what_d), 'descend');
    [~, idx_what_s] = sort(abs(w_what_s), 'descend');


    set_when = idx_when(1:k_neurons);
    set_what_d = idx_what_d(1:k_neurons);
    set_what_s = idx_what_s(1:k_neurons);

    overlap_real_d(i) = length(intersect(set_when, set_what_d)) / k_neurons;
    overlap_real_s(i) = length(intersect(set_when, set_what_s)) / k_neurons;
    
    %% -------------------------------
    % OVERLAP SHUFFLE
    %% -------------------------------
    
    for p = 1:nPerm
        
        idx_what_shuff_d = idx_what_d(randperm(length(idx_what_d)));
        set_what_shuff_d = idx_what_shuff_d(1:k_neurons);
        
        overlap_shuff_d(i,p) = ...
            length(intersect(set_when, set_what_shuff_d)) / k_neurons;
        
        idx_what_shuff_s = idx_what_s(randperm(length(idx_what_s)));
        set_what_shuff_s = idx_what_shuff_s(1:k_neurons);
        
        overlap_shuff_s(i,p) = ...
            length(intersect(set_when, set_what_shuff_s)) / k_neurons;
    end
%}
    
    
end


% mean compute for the shuffles
cos_shuff_d_mean = mean(cos_shuff_d,2);
cos_shuff_s_mean = mean(cos_shuff_s,2);

overlap_shuff_d_mean = mean(overlap_shuff_d,2);
overlap_shuff_s_mean = mean(overlap_shuff_s,2);


%mean and t-test from 0 
mean_cos_d_real = mean(cos_real_d)
mean_cos_d_shuffle = mean(cos_shuff_d_mean)

mean_cos_s_real = mean(cos_real_s)
mean_cos_s_shuffle = mean(cos_shuff_s_mean)

[~,p_d] = ttest(cos_real_d)
[~,p_d_shuffle] = ttest(cos_shuff_d_mean)
[~,p_s] = ttest(cos_real_s)
[~,p_s_shuffle] = ttest(cos_shuff_s_mean)

binwidth = 0.05;
edges = -1:binwidth:1;

%% ===============================
% PANEL (a): COSINE SIMILARITY
%% ===============================
figure(1); clf;

subplot(2,3,1)
%histogram(cos_shuff_d_mean, edges, 'FaceColor',[0.7 0.7 0.7]);
histogram(cos_shuff_d(:), edges,  'Normalization','probability','FaceColor',[0.7 0.7 0.7]); hold on;
histogram(cos_real_d, edges, 'Normalization','probability','FaceColor','k'); 
xlabel('Cosine similarity');
ylabel('Count');
legend({'Shuffled controls','Data'});
title('Cosine similarity btw WhenCD and WhatCD^{motion}');
box off; fig_setting();
line([0 0], ylim, 'Color','k','LineStyle','--');

% stats
[p,~] = signrank(cos_real_d, cos_shuff_d_mean);
disp(['Cosine data vs shuffle p = ', num2str(p)]);
text(-1, max(ylim)*0.9, ['p = ', num2str(p,'%0.3g')]);

subplot(2,3,4)
%histogram(cos_shuff_s_mean, edges, 'FaceColor',[0.7 0.7 0.7]);
histogram(cos_shuff_s(:), edges,  'Normalization','probability', 'FaceColor',[0.7 0.7 0.7]); hold on;
histogram(cos_real_s, edges, 'Normalization','probability', 'FaceColor','k');
xlabel('Cosine similarity');
ylabel('Count');
legend({'Shuffled controls','Data'});
title('Cosine similarity btw WhenCD and WhatCD^{sacc}');
box off; fig_setting();
line([0 0], ylim, 'Color','k','LineStyle','--');

% stats
[p,~] = signrank(cos_real_s, cos_shuff_s_mean);
disp(['Cosine data vs shuffle p = ', num2str(p)]);
text(-1, max(ylim)*0.9, ['p = ', num2str(p,'%0.3g')]);

%% ===============================
% PANEL (b): OVERLAP
%% ===============================
subplot(2,3,2)
hold on;
% optional: colormap (comment out if you want single color)
cmap = parula(nSess);  

for i = 1:nSess
    
    % jitter per session (shared across the pair)
    jitter = (rand-0.5)*0.1;
    x1 = 1 + jitter;
    x2 = 2 + jitter;

    % choose color
    col = cmap(i,:);   % or use: col = [0 0 0];
    %col = [0 0 0];
    
    % line connecting paired values
    plot([x1 x2], [overlap_real_d(i) overlap_shuff_d_mean(i)], ...
         '-', 'Color', col, 'LineWidth', 0.5);

    % points
    plot(x1, overlap_real_d(i), 'o', ...
        'MarkerFaceColor', col, 'MarkerEdgeColor','none', 'MarkerSize',6);

    plot(x2, overlap_shuff_d_mean(i), 'o', ...
        'MarkerFaceColor', col, 'MarkerEdgeColor','none', 'MarkerSize',6);
end

% ---- overlay shuffle mean ± SE ----
mu = mean(overlap_shuff_d_mean);
se = std(overlap_shuff_d_mean)/sqrt(nSess);

errorbar(2, mu, se, 'k', 'LineWidth',2, 'CapSize',0);
plot(2, mu, 'ks', 'MarkerFaceColor','k','MarkerSize',8);


% ---- overlay real mean (optional but recommended) ----
mu_real = mean(overlap_real_d);
se_real = std(overlap_real_d)/sqrt(nSess);

errorbar(1, mu_real, se_real, 'k', 'LineWidth',2, 'CapSize',0);
plot(1, mu_real, 'ks', 'MarkerFaceColor','k','MarkerSize',8);
line([1 2],[mu_real mu], 'Color','k','LineWidth',2);

% ---- expected overlap reference ----
%line([0.5 2.5],[topK topK],'LineStyle','--','Color',[0.5 0.5 0.5]);

% ---- formatting ----
xlim([0.5 2.5]);
set(gca,'XTick',[1 2],'XTickLabel',{'Data','Shuffle'});
ylabel('Jaccard index');
title(['Overlap in the top ',num2str(topK) '% contributing neurons']);
box off;
fig_setting();

[p2,~] = signrank(overlap_real_d, overlap_shuff_d_mean);
text(1.2, max(ylim)*0.9, ['p = ', num2str(p2,'%0.3g')]);






subplot(2,3,5)
hold on;
cmap = parula(nSess);  

for i = 1:nSess
    
    % jitter per session (shared across the pair)
    jitter = (rand-0.5)*0.1;
    x1 = 1 + jitter;
    x2 = 2 + jitter;

    % choose color
    col = cmap(i,:);   % or use: col = [0 0 0];
    %col = [0 0 0];
    
    % line connecting paired values
    plot([x1 x2], ...
         [overlap_real_s(i) overlap_shuff_s_mean(i)], ...
         '-', 'Color', col, 'LineWidth', 0.5);

    % points
    plot(x1, overlap_real_s(i), 'o', ...
        'MarkerFaceColor', col, 'MarkerEdgeColor','none', 'MarkerSize',6);

    plot(x2, overlap_shuff_s_mean(i), 'o', ...
        'MarkerFaceColor', col, 'MarkerEdgeColor','none', 'MarkerSize',6);
end

% ---- overlay shuffle mean ± SE ----
mu = mean(overlap_shuff_s_mean);
se = std(overlap_shuff_s_mean)/sqrt(nSess);

errorbar(2, mu, se, 'k', 'LineWidth',2, 'CapSize',0);
plot(2, mu, 'ks', 'MarkerFaceColor','k','MarkerSize',8);


% ---- overlay real mean (optional but recommended) ----
mu_real = mean(overlap_real_s);
se_real = std(overlap_real_s)/sqrt(nSess);

errorbar(1, mu_real, se_real, 'k', 'LineWidth',2, 'CapSize',0);
plot(1, mu_real, 'ks', 'MarkerFaceColor','k','MarkerSize',8);
line([1 2],[mu_real mu], 'Color','k','LineWidth',2);

% ---- expected overlap reference ----
%line([0.5 2.5],[topK topK],'LineStyle','--','Color',[0.5 0.5 0.5]);

% ---- formatting ----
xlim([0.5 2.5]);
set(gca,'XTick',[1 2],'XTickLabel',{'Data','Shuffle'});
ylabel('Jaccard index');
title(['Overlap in the top ',num2str(topK) '% contributing neurons']);
box off;
fig_setting();

[p2,~] = signrank(overlap_real_s, overlap_shuff_s_mean);
text(1.2, max(ylim)*0.9, ['p = ', num2str(p2,'%0.3g')]);

% quick outlier check
%{
idx_outlier = overlap_real_s > 0.2;

overlap_real_s_no = overlap_real_s(~idx_outlier);
overlap_shuff_s_no = overlap_shuff_s_mean(~idx_outlier);

[p_no,~] = signrank(overlap_real_s_no, overlap_shuff_s_no);

disp(['p without outlier = ', num2str(p_no)]);
%}

%% ===============================
% PANEL (c): cosine vs overlap
%% ===============================

% ---- motion ----
subplot(2,3,3); hold on;

scatter(abs(cos_real_d), overlap_real_d, 40, 'k', 'filled', ...
    'MarkerFaceAlpha',0.6);

xlabel('|Cosine|');
ylabel('Overlap');
box off;
fig_setting();

line([0 0], ylim, 'Color',[0.5 0.5 0.5],'LineStyle','--');
%line(xlim, [topK topK], 'Color',[0.5 0.5 0.5],'LineStyle','--');

[rho, pval] = corr(abs(cos_real_d), overlap_real_d, 'type','Spearman');
text(-0.1, max(ylim)*0.9, ['Spearman rho = ', num2str(rho), 'p = ', num2str(pval,'%0.3g')]);

% ---- sacc ----
subplot(2,3,6); hold on;

scatter(abs(cos_real_s), overlap_real_s, 40, 'k', 'filled', ...
    'MarkerFaceAlpha',0.6);

xlabel('|Cosine|');
ylabel('Overlap');
box off;
fig_setting();

line([0 0], ylim, 'Color',[0.5 0.5 0.5],'LineStyle','--');
%line(xlim, [topK topK], 'Color',[0.5 0.5 0.5],'LineStyle','--');

[rho, pval] = corr(abs(cos_real_s), overlap_real_s, 'type','Spearman');
text(-0.1, max(ylim)*0.9, ['Spearman rho = ', num2str(rho), 'p = ', num2str(pval,'%0.3g')]);

sgtitle('Figure 3: WhenCD vs WhatCD geometry');

end



