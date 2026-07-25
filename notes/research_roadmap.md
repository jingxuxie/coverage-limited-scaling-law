# Research Roadmap

## Central claim under investigation

Repeated optimization can only reveal directions present in the unique data. In sparse power-law regression, model width, optimization time, and unique-feature occupancy create distinct frontiers:

\[
J_{\mathrm{cap}}=m,
\qquad
J_{\mathrm{opt}}=(\eta nK)^{1/a},
\qquad
J_{\mathrm{cov}}=n^{1/s}.
\]

The working law is

\[
\mathcal E(m,n,K)
\asymp
\min\{J_{\mathrm{cap}},J_{\mathrm{opt}},J_{\mathrm{cov}}\}^{1-b}.
\]

## Completed in proof version 1

- Universal unseen-feature lower bound for arbitrary learners.
- Power-law occupancy asymptotics.
- Capacity-plus-coverage lower bound.
- Exact finite-epoch one-hot theorem.
- Matching one-hot three-frontier rate.
- Exact multi-active theorem for singleton-filtered replay SGD.
- Exact numerical pilot confirming the optimization and coverage exponents.

## Main theorem still required

Prove a finite-time upper bound for ordinary multi-pass SGD that uses all coactive examples. Candidate variants, in increasing difficulty:

1. With-replacement sampling from a fixed dataset, clipped updates.
2. Independent random reshuffling each epoch.
3. Same-order cyclic replay.
4. Random Gaussian sketch followed by trainable linear readout.

## Experiment sequence

1. Implement sparse multi-active data generation without dense matrices.
2. Compare ordinary, clipped, and singleton-filtered SGD.
3. Sweep unique data \(n\), epochs \(K\), width \(m\), and sparsity mass \(\mu=\sum_jp_j\).
4. Measure pre-saturation slope, plateau height, and saturation epoch.
5. Test collapse against the three-frontier prediction.
6. Hold \(nK\) fixed and vary the unique/reuse split.
7. Add random sketches only after the direct model is understood.

## Go/no-go criteria

Continue with the simple three-frontier theory if ordinary SGD shows:

- pre-plateau slope near \(-(b-1)/a\),
- large-epoch floor near \(n^{-(b-1)/s}\),
- saturation near \(\min(m^a,n^{a/s})/(\eta n)\), and
- stable collapse across at least two choices of \((a,s,b)\).

If these fail systematically as coactivation density increases, characterize the deviation as an interference frontier rather than hiding it.

## Paper framing

The one-hot result is a benchmark and should not be presented as the principal novelty. The strongest defensible story is:

1. unique-data coverage gives an information-theoretic floor;
2. that floor is achievable for genuinely multi-active sparse data;
3. ordinary replay SGD either matches the three-frontier law or exhibits a new coactivation-dependent regime;
4. the resulting saturation estimator decides when another epoch is less useful than another unique example.
