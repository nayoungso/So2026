function nlogl = fitJointCost_generic(theta, data, calcFun)
% fitJointCost_generic
%
% Generic joint psychometric+chronometric cost function, shared across
% all three termination-mechanism models (imposed deadline / magnitude-
% timing / collapsing-bound DDM). Computes exactly the same cost as your
% existing fitDiff5_bias.m -- weighted least-squares on mean RT (T1 and
% T2 choices, per coherence) plus binomial log-likelihood on choice
% proportions -- but delegates the model predictions themselves to
% calcFun, so this one function serves all three models rather than
% duplicating the cost logic three times.
%
% INPUTS
%   theta   : parameter vector for whichever model calcFun implements
%   data    : [coh_set, RT1_mean, RT1_se, RT2_mean, RT2_se, nT1, ntotal]
%             -- identical layout to data1 in
%             RT_behav_fitPsychChronoData_v3.m
%   calcFun : function handle, [t1,t2,p] = calcFun(cohs, theta)
%
% OUTPUT
%   nlogl : total negative log-likelihood (Gaussian RT terms + binomial
%           choice term). This already includes the Gaussian normalizing
%           constants (not just a bare SSE), so it can be used directly
%           for AIC = 2*nParams + 2*nlogl to compare the three models,
%           which have different numbers of free parameters.
%
% EXAMPLE
%   nlogl = fitJointCost_generic(theta, data1, @calcModel3_collapseBound_asymTnd)

cohs    = data(:,1);
t1_obs  = data(:,2);
t1_se   = data(:,3);
t2_obs  = data(:,4);
t2_se   = data(:,5);
n1_obs  = data(:,6);
n_total = data(:,7);

[t1_pred, t2_pred, p_pred] = calcFun(cohs, theta);

N = size(data,1);
nlogl = 0;

t1_ind = ~isnan(t1_pred) & ~isnan(t1_obs) & ~isnan(t1_se);
t2_ind = ~isnan(t2_pred) & ~isnan(t2_obs) & ~isnan(t2_se);

% ---- Cost of RT (Gaussian negative log-likelihood; same convention as
% your existing fitDiff5_bias.m) ----
nlogl = nlogl + sum((t1_obs(t1_ind) - t1_pred(t1_ind)).^2 ./ (2.*t1_se(t1_ind).^2)) ...
              + N./2.*log(2.*pi) + sum(log(t1_se(t1_ind)));
nlogl = nlogl + sum((t2_obs(t2_ind) - t2_pred(t2_ind)).^2 ./ (2.*t2_se(t2_ind).^2)) ...
              + N./2.*log(2.*pi) + sum(log(t2_se(t2_ind)));

% ---- Cost of choice probability (binomial negative log-likelihood) ----
p_pred = min(max(p_pred, eps), 1-eps);
nlogl = nlogl - sum(gammaln(n_total+1) - gammaln(n1_obs+1) - gammaln(n_total-n1_obs+1) ...
    + n1_obs.*log(p_pred) + (n_total-n1_obs).*log(1-p_pred));

end