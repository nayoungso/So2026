function [t1, t2, p] = calcModel3_collapseBound_asymTnd(cohs, theta)
% calcModel3_collapseBound_asymTnd
%
% MODEL 3: standard drift-diffusion, but with a COLLAPSING bound (instead
% of the flat bound in your existing calcDiff5_bias.m) and asymmetric
% non-decision time (already present in calcDiff5_bias.m as t1_res/
% t2_res; kept here under the names tnd_up/tnd_dn). Uses your lab's
% existing spectral_dtb.m (Wolpert-style Fourier/Fokker-Planck solver,
% handles arbitrary time-varying bounds) instead of the closed-form tanh
% formula in calcDiff5_bias.m, which only holds for a FLAT bound.
%
% This is the model in which decision termination and decision content
% are governed by the SAME evidence stream -- the alternative that
% Models 1 (imposed deadline) and 2 (magnitude-driven independent timing)
% are meant to be ruled out against.
%
% Bound shape matches your library's cost_dtb_collapse.m convention: flat
% at height B until time Balpha, then quadratic decay at rate Bbeta:
%   Bup(t) = B,                        t <= Balpha
%   Bup(t) = B - Bbeta*(t-Balpha)^2,   t >  Balpha
% (floored above 0 so the bound can't cross zero/invert)
%
% theta = [k, b, B, Balpha, Bbeta, tnd_up, tnd_dn]
%   k       : drift scale (drift = k*C + b)
%   b       : drift/choice bias
%   B       : initial (flat) bound height
%   Balpha  : time (ms) at which the bound begins to collapse
%   Bbeta   : rate of the post-Balpha quadratic collapse
%   tnd_up  : non-decision time, T1 (up) choices (ms)
%   tnd_dn  : non-decision time, T2 (down) choices (ms)
%
% NOTES
%   - UNITS: spectral_dtb.m's docstring says "SI units" (seconds); your
%     RT data / Balpha / tnd are in ms elsewhere in this codebase, so
%     this function converts to seconds internally for the spectral_dtb
%     call and converts back to ms for the returned t1/t2. Double-check
%     this conversion against real data before trusting fits -- this is
%     the single most likely place for a units bug, and it hasn't been
%     run against your actual trial data yet.
%   - dt (solver's internal time step) trades accuracy against speed;
%     5 ms (0.005 s) matches the value already used in your library's own
%     cost_dtb_collapse.m.
%   - Whether "cohs" here are on the same numeric scale as k in your
%     existing calcDiff5_bias.m (i.e. raw signed coherence, not x1000 or
%     percent) needs to match -- check against how coh_set is built in
%     RT_behav_fitPsychChronoData_v3.m (coh = dot_coh*dir_sign/1000).
%
% EXAMPLE
%   [t1,t2,p] = calcModel3_collapseBound_asymTnd(coh_set', theta)

k      = theta(1);
b      = theta(2);
B      = theta(3);
Balpha = theta(4);
Bbeta  = theta(5);
tnd_up = theta(6);
tnd_dn = theta(7);

dt_s     = 0.005;              % seconds; matches cost_dtb_collapse.m
maxRT_ms = 3000;                % generous upper bound on RT (ms) for the time grid
tmax_s   = maxRT_ms/1000 + 0.3;
t_s      = (0:dt_s:tmax_s)';
nt       = length(t_s);

Balpha_s = Balpha/1000;

Bup = B*ones(nt,1);
collapse = t_s > Balpha_s;
Bup(collapse) = B - Bbeta.*(t_s(collapse) - Balpha_s).^2;
Bup = max(Bup, 0.001);          % floor so the bound never fully closes/crosses zero
Blo = -Bup;

ustrength = cohs(:)';            % one drift level per requested coherence
uv = k.*ustrength + b;            % per-second drift

md = max(abs(uv));
sm = md*dt_s + sqrt(dt_s)*4;
y  = linspace(min(Blo)-sm, max(Bup)+sm, 512)';
y0 = zeros(size(y));
[~,i0] = min(abs(y-0));
y0(i0) = 1;

D = spectral_dtb(uv, t_s, Bup, Blo, y, y0);

p     = D.up.p(:);                     % P(up) per coherence
Td_up = D.up.mean_t(:) * 1000;          % seconds -> ms
Td_dn = D.lo.mean_t(:) * 1000;

t1 = Td_up + tnd_up;
t2 = Td_dn + tnd_dn;

end