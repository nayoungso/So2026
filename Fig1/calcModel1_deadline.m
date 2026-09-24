function [t1, t2, p] = calcModel1_deadline(cohs, theta)
% calcModel1_deadline
%
% MODEL 1: imposed deadline. Evidence accumulates as a standard Wiener
% process (drift v = k*C + b, unit diffusion variance) but never
% terminates on its own -- there is no bound. Instead, on each trial the
% process is simply read out at an externally-imposed stopping time T,
% drawn INDEPENDENTLY of coherence: T ~ Gamma(mean=mu_T, sd=sigma_T).
% choice = sign of the accumulated evidence at time T.
%
% This is the direct test of "termination is independent of the
% accumulated evidence": T's distribution does not depend on C at all.
% Choice accuracy still improves with |C| (the readout still reflects
% however much evidence has accumulated by time T), but mean RT is flat
% across coherence (up to the deadline's own variability) -- no coupling
% between motion strength and response time. That flatness is the
% pattern this model is built to represent, for comparison against
% Model 3's actual coupling.
%
% theta = [k, b, mu_T, sigma_T, tnd]
%   k       : evidence-accumulation drift scale (drift = k*C + b)
%   b       : drift/choice bias (constant added to k*C)
%   mu_T    : mean of the imposed deadline distribution (ms)
%   sigma_T : SD of the imposed deadline distribution (ms)
%   tnd     : non-decision time, SHARED across T1 (up) and T2 (down)
%             choices (ms). Kept symmetric rather than separate tnd_up/
%             tnd_dn so this model has NO choice-dependent asymmetry at
%             all -- a coherence-independent constant offset wouldn't
%             actually have violated the independence claim either way,
%             but a single shared tnd is the more conservative, strictly
%             direction-independent version to start with. Model 3 keeps
%             asymmetric tnd_up/tnd_dn, since realistic motor asymmetry
%             is a plausible feature of the coupled account, not
%             something that model is meant to rule out.
%
% Choice probability and choice-conditional mean deadline are computed by
% numerically integrating over the Gamma-distributed deadline (Gamma
% rather than Gaussian so T can't go negative, with a simple mean/SD
% parameterization: shape=mu^2/sigma^2, scale=sigma^2/mu).
%
% NOTES
%   - No closed form exists for E_T[Phi(v*sqrt(T))] under a Gamma T, so
%     this integrates numerically over a fine T grid (trapezoidal-style
%     summation). This is exact up to grid resolution, NOT a Monte Carlo
%     approximation.
%   - All times (T, tnd) are in ms, matching RT_behav_fitPsychChronoData_v3.m.
%
% EXAMPLE
%   [t1,t2,p] = calcModel1_deadline(coh_set', theta)

k       = theta(1);
b       = theta(2);
mu_T    = theta(3);
sigma_T = theta(4);
tnd     = theta(5);

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

    t1(i,1) = ET_up + tnd;
    t2(i,1) = ET_dn + tnd;
end

end