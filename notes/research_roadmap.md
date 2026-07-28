# Research Roadmap and Current Status

## Completed for the AAAI submission

The submission now contains a coherent theorem–estimator–experiment stack.

1. **Coverage lower bound:** an algorithm-independent unseen-feature floor and its sharp power-law constant.
2. **Occupancy calculus:** asymptotics for distinct observed features, observed Euclidean teacher energy, and missing predictive mass.
3. **Exact benchmark:** a finite-sample one-hot replay formula with capacity, optimization, and coverage frontiers.
4. **Sharp coactive theorem:** example-norm-clipped, with-replacement replay uses every coactive row; a label-free anchor readout reaches the sharp coverage rate.
5. **Plain output theorem:** the unmasked Polyak average needs no singleton-support abort and attains the same exponents up to logarithms.
6. **Replay horizon:**
   \[
   K_{\rm sat}=O(\eta^{-1}n^{a/s-1}).
   \]
7. **Full frontier schedule:**
   \[
   O\!\left(m^{-(b-1)}+(\eta T)^{-(b-1)/a}+n^{-(b-1)/s}\right)
   \]
   for the anchor output.
8. **Input-only estimator:** empirical activation frequencies and conditional amplitudes predict the saturation epoch without labels.
9. **Direct empirical coverage of all three frontiers:** dataset-size/epoch sweeps, a width sweep, progressive compute scaling, source exponents, fixed-compute allocation, coactivation protocols, clipping stress, and stopping-rule calibration.

The expanded release contains 795 training runs and 5,400 population-risk checkpoints.

## Submission claim boundary

The theorem applies to:

- independent Bernoulli sparse coordinates;
- `s>1`, `a>=s`, and `a-s+1<b<a+1`;
- noiseless realizable labels on the observed dictionary;
- with-replacement replay;
- example-norm clipping;
- Polyak averaging;
- a label-free, distribution-aware anchor mask for the sharp rate.

The ordinary unmasked output is also proved, with a logarithmic loss and no support-abort event. Cyclic replay, random reshuffling, last-iterate behavior, and rougher sources appear only as empirical robustness checks.

## Post-submission extensions, ranked

1. **Noisy labels and a variance frontier.** Determine how the coverage floor interacts with stochastic-gradient variance and early stopping.
2. **Empirical anchor thresholds.** Replace population `p_j` in the sharp mask by confidence-adjusted empirical frequencies.
3. **Last-iterate and reshuffling theory.** Prove the sharp pre-saturation exponent without Polyak averaging or with independent reshuffling.
4. **Correlated sparse supports.** Identify when coactivation clusters change only constants and when they create a new interference frontier.
5. **Random sketches and nonlinear features.** Determine whether mixing preserves, hides, or improves rare-feature coverage.
6. **Real sparse representations.** Test the input-only stopping rule on bag-of-words, routed-expert, hashed, or recommender-system features.

These are intentionally framed as future work rather than additional claims in the AAAI manuscript.
