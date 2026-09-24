function [sig, adj_p] = fdr_bh(pvals, q)
%FDR_BH Benjamini-Hochberg FDR correction (step-up procedure).
%   pvals may contain NaN (skipped; returned as NaN/false).
%
%   [sig, adj_p] = fdr_bh(pvals, q)
%     sig   : logical, same size as pvals, true where adj_p <= q
%     adj_p : BH-adjusted p-values, same size as pvals

if nargin < 2 || isempty(q), q = 0.05; end

sig   = false(size(pvals));
adj_p = nan(size(pvals));

valid = ~isnan(pvals);
p = pvals(valid);
n = numel(p);
if n == 0, return; end

[psorted, order] = sort(p);
ranks      = 1:n;
adj_sorted = psorted .* n ./ ranks;

% enforce monotonicity (step-up): adj_sorted(i) <= adj_sorted(i+1)
for i = n-1:-1:1
    adj_sorted(i) = min(adj_sorted(i), adj_sorted(i+1));
end
adj_sorted = min(adj_sorted, 1);

adj = nan(1,n);
adj(order) = adj_sorted;

adj_p(valid) = adj;
sig(valid)   = adj <= q;

end
