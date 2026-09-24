function [sigMask, clusters, tobs, threshold] = clusterPermTest(dataMat, nPerm, alpha)
%CLUSTERPERMTEST Sign-flip cluster-based permutation test (one-sample vs 0).
%   dataMat: sessions x time. Tests each column's mean against zero,
%   correcting for multiple comparisons across time via cluster-mass
%   permutation (Maris & Oostenveld, 2007). Sessions (rows) are the
%   exchangeable unit under H0, so signs are flipped per session across
%   the whole time course at once, preserving temporal autocorrelation
%   within a session's timecourse.
%
%   For correlation coefficients, Fisher-z transform (atanh) dataMat
%   before calling this so the per-session values are approximately
%   normal and comparable across sessions.
%
%   [sigMask, clusters, tobs, threshold] = clusterPermTest(dataMat, nPerm, alpha)
%     sigMask   : 1 x nT logical, true at time bins in a significant cluster
%     clusters  : struct array (idx, mass, pval, sig) for every candidate cluster
%     tobs      : 1 x nT observed t-statistic
%     threshold : 1 x nT cluster-forming t threshold (per-column df)

if nargin < 2 || isempty(nPerm),  nPerm = 5000; end
if nargin < 3 || isempty(alpha),  alpha = 0.05; end

[nSess, nT] = size(dataMat);

n_valid = sum(~isnan(dataMat),1);
m       = nanmean(dataMat,1);
s       = nanstd(dataMat,[],1);
tobs    = m ./ (s./sqrt(n_valid));
tobs(n_valid < 3) = NaN;

df        = max(n_valid - 1, 1);
threshold = tinv(1-alpha/2, df);

clusters = findClusters(abs(tobs) > threshold, tobs);

nullMax = zeros(1,nPerm);
for p = 1:nPerm
    flip = sign(rand(nSess,1) - 0.5);
    flip(flip == 0) = 1;
    permMat = dataMat .* flip;

    mp = nanmean(permMat,1);
    sp = nanstd(permMat,[],1);
    tp = mp ./ (sp./sqrt(n_valid));
    tp(n_valid < 3) = NaN;

    cl_p = findClusters(abs(tp) > threshold, tp);
    if isempty(cl_p)
        nullMax(p) = 0;
    else
        nullMax(p) = max([cl_p.mass]);
    end
end

sigMask = false(1,nT);
for c = 1:numel(clusters)
    clusters(c).pval = (1 + sum(nullMax >= clusters(c).mass)) / (nPerm + 1);
    clusters(c).sig  = clusters(c).pval < alpha;
    if clusters(c).sig
        sigMask(clusters(c).idx) = true;
    end
end

end

function clusters = findClusters(candidate, tvals)
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
