function [t1, t2, p] = calcModel2_magnitudeTiming_asymTnd(cohs, theta)
% calcModel2_magnitudeTiming_asymTnd
%
% MODEL 2 (matched-flexibility variant): parallel/independent magnitude-
% driven timing, IDENTICAL to calcModel2_magnitudeTiming.m except tnd is
% now split into separate tnd_up/tnd_dn, matching Model 3's non-decision-
% time flexibility exactly (8 params here vs. calcModel2_magnitudeTiming.m's 7).
%
% Purpose: same robustness-check rationale as
% calcModel1_deadline_asymTnd.m -- see that file's header. Used by
% RT_behav_compareModels_v2.m alongside AIC *and* BIC.
%
% Two simultaneous, INDEPENDENT processes per trial:
%   (a) "When": a one-boundary accumulator with drift k_time*(|C|+floor)
%       -- always positive, driven by UNSIGNED motion strength, not
%       signed evidence -- racing to a bound that collapses over time.
%       First-passage time to that bound is the termination time T.
%   (b) "What": a completely separate standard signed accumulator (drift
%       k_choice*C + b, unit diffusion variance) that is simply READ OUT
%       (its sign taken) at time T -- it never has its own bound.
%
% theta = [k_choice, b, k_time, B_time, Balpha_time, Bbeta_time, tnd_up, tnd_dn]
%   k_choice    : choice-accumulator drift scale (drift = k_choice*C + b)
%   b           : choice-accumulator bias
%   k_time      : timing-accumulator drift scale on |C| (always >=0 drift)
%   B_time      : timing-accumulator's initial (flat) bound height
%   Balpha_time : time (ms) at which the timing bound begins to collapse
%   Bbeta_time  : rate of the post-Balpha_time quadratic collapse
%   tnd_up      : non-decision time, T1 (up) choices (ms)
%   tnd_dn      : non-decision time, T2 (down) choices (ms)
%             -- see calcModel1_deadline_asymTnd.m's header for why an
%             asymmetric, coherence-independent tnd doesn't compromise
%             this model's independence claim. See
%             calcModel2_magnitudeTiming.m for the single-tnd (more
%             conservative) version.
%
% NOTES -- see calcModel2_magnitudeTiming.m for the full numerical
% details (spectral_dtb one-sided absorption, COH_FLOOR, Blo_mult, units);
% identical here except for the split tnd.
%
% EXAMPLE
%   [t1,t2,p] = calcModel2_magnitudeTiming_asymTnd(coh_set', theta)

k_choice    = theta(1);
b           = theta(2);
k_time      = theta(3);
B_time      = theta(4);
Balpha_time = theta(5);
Bbeta_time  = theta(6);
tnd_up      = theta(7);
tnd_dn      = theta(8);

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

    t1(i,1) = ET_up*1000 + tnd_up;
    t2(i,1) = ET_dn*1000 + tnd_dn;
end

end