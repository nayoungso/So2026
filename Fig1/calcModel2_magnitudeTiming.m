function [t1, t2, p] = calcModel2_magnitudeTiming(cohs, theta)
% calcModel2_magnitudeTiming
%
% MODEL 2: parallel/independent magnitude-driven timing, WITH a
% collapsing bound in the timing channel (added per discussion: giving
% Model 2 the same urgency-like flexibility as Model 3 isolates the
% actual thing under test -- whether termination depends on the full
% signed accumulator vs. on stimulus magnitude alone -- rather than
% Model 2 losing simply for lacking a parameter Model 3 has).
%
% Two simultaneous, INDEPENDENT processes per trial:
%   (a) "When": a one-boundary accumulator with drift k_time*(|C|+floor)
%       -- always positive, driven by UNSIGNED motion strength, not
%       signed evidence -- racing to a bound that collapses over time
%       (same B/Balpha/Bbeta shape as Model 3's bound, but here governing
%       only the termination-time channel). First-passage time to that
%       bound is the termination time T.
%   (b) "What": a completely separate standard signed accumulator (drift
%       k_choice*C + b, unit diffusion variance) that is simply READ OUT
%       (its sign taken) at time T -- it never has its own bound.
%       choice = sign(X_choice(T)), X_choice(T) ~ Normal((k_choice*C+b)*T, T).
%
% This decouples "how fast you respond" (driven purely by |C| and
% elapsed time, via the timing channel) from "what you decide" (driven
% by the signed evidence accumulator, evaluated at a time that has
% NOTHING to do with that accumulator's own state).
%
% theta = [k_choice, b, k_time, B_time, Balpha_time, Bbeta_time, tnd]
%   k_choice    : choice-accumulator drift scale (drift = k_choice*C + b)
%   b           : choice-accumulator bias
%   k_time      : timing-accumulator drift scale on |C| (always >=0 drift)
%   B_time      : timing-accumulator's initial (flat) bound height
%   Balpha_time : time (ms) at which the timing bound begins to collapse
%   Bbeta_time  : rate of the post-Balpha_time quadratic collapse
%   tnd         : non-decision time, SHARED across T1 (up) and T2 (down)
%                 choices (ms) -- symmetric, not tnd_up/tnd_dn, so this
%                 model has no choice-dependent asymmetry at all; see
%                 calcModel1_deadline.m's header for the same reasoning.
%
% With a collapsing bound, the timing channel's first-passage time no
% longer has a closed-form (Wald) distribution, so it's computed
% numerically with the SAME spectral_dtb.m solver Model 3 uses: one-sided
% absorption, where only the upper (collapsing) bound is real and the
% lower "bound" is just a numerically-convenient stand-in for "never
% reached" (see COH_FLOOR/Blo_mult below), not a genuine second boundary.
% The resulting per-coherence termination-time distribution is then fed
% into the SAME semi-analytic choice-readout integral used in
% calcModel1_deadline.m: P(up|C) = integral of Phi((k_choice*C+b)*sqrt(T))
% over the timing channel's T-distribution, rather than the Gamma
% deadline distribution used there. This keeps Model 2 fully
% deterministic/smooth in theta -- no Monte Carlo, no simulation noise
% for fminsearch to fight.
%
% NOTES
%   - COH_FLOOR is a small, fixed (non-fitted) additive floor on |C| so
%     the timing drift is never exactly 0 at C=0 (which would make the
%     one-sided first-passage time degenerate/never-terminating within a
%     finite grid) -- a numerical necessity, not a scientific claim.
%   - Blo_mult sets the (finite, moderately negative) "practically never
%     reached" lower boundary passed to spectral_dtb, since the y-grid
%     spectral_dtb's caller builds must stay finite (literal -Inf breaks
%     linspace) -- see calcModel3_collapseBound_asymTnd.m's Bup/Blo
%     construction for the same pattern.
%   - Units: seconds internally (matching spectral_dtb.m and Model 3),
%     converted to ms for the returned t1/t2 and for Balpha_time's input
%     units, matching the rest of this codebase.
%
% EXAMPLE
%   [t1,t2,p] = calcModel2_magnitudeTiming(coh_set', theta)

k_choice    = theta(1);
b           = theta(2);
k_time      = theta(3);
B_time      = theta(4);
Balpha_time = theta(5);
Bbeta_time  = theta(6);
tnd         = theta(7);

COH_FLOOR = 0.01;   % see NOTES -- not a free parameter
Blo_mult  = 3;       % lower "bound" fixed at -Blo_mult*B_time; practically unreachable

dt_s     = 0.005;
maxRT_ms = 3000;
tmax_s   = maxRT_ms/1000 + 0.3;
t_s      = (0:dt_s:tmax_s)';
nt       = length(t_s);

Balpha_time_s = Balpha_time/1000;

Bup = B_time*ones(nt,1);
collapse = t_s > Balpha_time_s;
Bup(collapse) = B_time - Bbeta_time.*(t_s(collapse) - Balpha_time_s).^2;
Bup = max(Bup, 0.001);
Blo = -Blo_mult*B_time*ones(nt,1);   % practically-never-reached lower "bound"

cohs = cohs(:);
nC = length(cohs);

v_time = k_time .* (abs(cohs) + COH_FLOOR);  % one drift level per coherence, always >0

md = max(abs(v_time));
sm = md*dt_s + sqrt(dt_s)*4;
y  = linspace(min(Blo)-sm, max(Bup)+sm, 512)';
y0 = zeros(size(y));
[~,i0] = min(abs(y-0));
y0(i0) = 1;

Dtime = spectral_dtb(v_time, t_s, Bup, Blo, y, y0);
% Dtime.up.pdf_t is (nt x nC): probability density, over the time grid,
% of the TIMING channel terminating at each time, for each coherence.

t1 = nan(nC,1);
t2 = nan(nC,1);
p  = nan(nC,1);

for i = 1:nC
    C = cohs(i);
    v_choice = k_choice.*C + b;

    pdfT = Dtime.up.pdf_t(:,i);
    pTotal = sum(pdfT);
    pdfT = pdfT ./ max(pTotal, eps);   % normalize over the finite grid

    Pup_givenT = normcdf(v_choice .* sqrt(t_s));

    p(i,1) = sum(Pup_givenT .* pdfT);

    ET_up = sum(t_s .* Pup_givenT       .* pdfT) / max(p(i,1), eps);
    ET_dn = sum(t_s .* (1-Pup_givenT)   .* pdfT) / max(1-p(i,1), eps);

    t1(i,1) = ET_up*1000 + tnd;
    t2(i,1) = ET_dn*1000 + tnd;
end

end