function [t1, t2, p] = calcModel1_deadline_asymTnd(cohs, theta)
% calcModel1_deadline_asymTnd
%
% MODEL 1 (matched-flexibility variant): imposed deadline, IDENTICAL to
% calcModel1_deadline.m except tnd is now split into separate tnd_up/
% tnd_dn, matching Model 3's non-decision-time flexibility exactly (6
% params here vs. calcModel1_deadline.m's 5).
%
% Purpose: calcModel1_deadline.m deliberately used a single, shared tnd
% to keep Model 1 the strictest, most conservative "termination is
% independent of accumulated evidence" model. This variant tests how
% robust that model's loss (vs. Model 3) is: if Model 1 still loses even
% after being given the SAME choice-dependent non-decision-time
% flexibility Model 3 has, that's a stronger result than winning only
% because Model 3 had an extra free parameter Model 1 lacked. Used by
% RT_behav_compareModels_v2.m alongside AIC *and* BIC, so you can see
% whether the conclusion holds both when the alternative models are given
% matched flexibility and under a complexity penalty that scales with
% sample size.
%
% Evidence accumulates as a standard Wiener process (drift v = k*C + b,
% unit diffusion variance) but never terminates on its own -- there is no
% bound. Instead, on each trial the process is simply read out at an
% externally-imposed stopping time T, drawn INDEPENDENTLY of coherence:
% T ~ Gamma(mean=mu_T, sd=sigma_T). choice = sign of the accumulated
% evidence at time T.
%
% theta = [k, b, mu_T, sigma_T, tnd_up, tnd_dn]
%   k       : evidence-accumulation drift scale (drift = k*C + b)
%   b       : drift/choice bias (constant added to k*C)
%   mu_T    : mean of the imposed deadline distribution (ms)
%   sigma_T : SD of the imposed deadline distribution (ms)
%   tnd_up  : non-decision time added for T1 (up) choices (ms)
%   tnd_dn  : non-decision time added for T2 (down) choices (ms)
%             -- a coherence-independent constant, so an asymmetry here
%             can only add a flat offset between T1/T2 RTs (consistent
%             with a peripheral motor-execution asymmetry), not the
%             coherence-dependent RT-vs-evidence coupling that would
%             actually violate this model's core "independent of
%             evidence" assumption. See calcModel1_deadline.m for the
%             single-tnd (more conservative) version.
%
% NOTES -- see calcModel1_deadline.m for the full numerical-integration
% notes (Gamma deadline grid, etc.); identical here except for the split
% tnd.
%
% EXAMPLE
%   [t1,t2,p] = calcModel1_deadline_asymTnd(coh_set', theta)

k       = theta(1);
b       = theta(2);
mu_T    = theta(3);
sigma_T = theta(4);
tnd_up  = theta(5);
tnd_dn  = theta(6);

nC = length(cohs);
t1 = nan(nC,1);
t2 = nan(nC,1);
p  = nan(nC,1);

% ---- Gamma(mean=mu_T, sd=sigma_T) deadline distribution ----
shape_T = (mu_T./sigma_T).^2;
scale_T = (sigma_T.^2)./mu_T;

% ---- T grid for numerical integration ----
T_max = mu_T + 8*sigma_T;
nGrid = 2000;
Tgrid = linspace(max(T_max/nGrid, 1), T_max, nGrid)';  % keep T>0 for gampdf
dT = Tgrid(2) - Tgrid(1);

pT = gampdf(Tgrid, shape_T, scale_T);
pT = pT ./ (sum(pT)*dT + eps);   % renormalize (grid truncation)

for i = 1:nC
    C = cohs(i);
    v = k.*C + b;

    Pup_givenT = normcdf(v .* sqrt(Tgrid));   % P(X(T)>0 | T) = Phi(v*sqrt(T))

    p(i,1) = sum(Pup_givenT .* pT) * dT;

    ET_up = sum(Tgrid .* Pup_givenT       .* pT) * dT / max(p(i,1), eps);
    ET_dn = sum(Tgrid .* (1-Pup_givenT)   .* pT) * dT / max(1-p(i,1), eps);

    t1(i,1) = ET_up + tnd_up;
    t2(i,1) = ET_dn + tnd_dn;
end

end