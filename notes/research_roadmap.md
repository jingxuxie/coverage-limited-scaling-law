# Research Roadmap and Current Status

## Completed theorem stack

The original roadmap asked whether ordinary replay SGD on genuinely multi-active sparse examples follows the coverage-limited law. The following pieces are now rigorous.

1. An algorithm-independent unseen-feature lower bound and its sharp power-law constant.
2. An exact finite-epoch one-hot replay formula with capacity, optimization, and coverage frontiers.
3. Statistical achievability for multi-active data using singleton-filtered replay.
4. A new **all-row coactive replay theorem** for example-norm-clipped, with-replacement SGD with Polyak averaging.
5. An all-data saturation corollary:
   \[
   K_{\rm sat}=\widetilde O(\eta^{-1}n^{a/s-1}).
   \]
6. A progressive-data schedule attaining
   \[
   \widetilde O\!\left(
   m^{-(b-1)}+(\eta T)^{-(b-1)/a}+n^{-(b-1)/s}
   \right).
   \]

The noncommuting rank-one update obstruction is handled by two observations: clipping makes every coactive update a Euclidean contraction, and naturally occurring singleton rows certify coordinate-wise curvature without being filtered out by the optimizer.

## Completed empirical package

The release sweep contains 670 sparse training runs and 4,285 risk checkpoints. It tests:

- optimization-to-coverage transitions over `n` and `K`;
- the three-frontier scaling collapse;
- the progressive compute schedule;
- fixed-compute fresh-data versus reuse allocation;
- four source exponents;
- three spectral configurations;
- coactivation density and sampling protocol;
- clipped versus unclipped stability.

The fixed-exponent two-term model explains 98.1% of averaged-risk variation, the progressive schedule has slope `-0.5016`, and all source-exponent slopes are within `0.02` of their predictions.

## Submission-ready claim boundary

The paper claims a theorem for:

- independent Bernoulli sparse coordinates;
- `s>1`, `a>=s`, and source range `a-s+1<b<a+1`;
- noiseless realizable labels on the observed dictionary;
- with-replacement replay;
- example-norm clipping;
- Polyak averaging;
- an observable support certificate, with zero fallback on certificate failure.

The experiments additionally test unguarded last iterates, cyclic replay, and random reshuffling. These are reported as empirical robustness, not theorem-level guarantees.

## Strong follow-up directions

1. Remove the logarithm by replacing simultaneous singleton coverage with a weighted occupancy argument.
2. Prove the pre-saturation `K^{-(b-1)/a}` last-iterate rate for ordinary all-data replay.
3. Extend the contraction proof to independently reshuffled and cyclic replay.
4. Add label noise and characterize the interaction between coverage and variance floors.
5. Replace coordinate-addressable features by a Gaussian random sketch or nonlinear random features.
6. Derive and validate an unlabeled estimator of the stopping epoch from empirical occupancy and covariance statistics.
