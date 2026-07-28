# Coverage-Limited Scaling Laws

This repository studies when repeated passes over a fixed sparse dataset stop helping because predictive features are absent from the unique examples.

For feature frequency, covariance, and target-energy profiles

\[
p_j\asymp j^{-s},\qquad
\lambda_j\asymp j^{-a},\qquad
g_j\asymp j^{-b},
\]

the theory separates three learned-rank frontiers,

\[
J_{\rm cap}=m,\qquad
J_{\rm opt}=(\eta nK)^{1/a},\qquad
J_{\rm cov}=n^{1/s}.
\]

The smallest frontier determines the leading risk. Capacity is improved by model width, optimization by additional stable updates, and coverage only by additional unique examples.

## Main results

1. **Information-theoretic coverage floor.** Every learner trained on `n` unique examples has Bayes—and therefore minimax—risk at least
   \[
   \sum_j g_j(1-p_j)^n\asymp n^{-(b-1)/s}.
   \]
   The proof also gives the sharp Gamma-function constant for exact power laws.

2. **Exact replay benchmark.** In a one-hot model, finite-epoch SGD obeys
   \[
   \mathcal R\asymp
   m^{-(b-1)}+(\eta nK)^{-(b-1)/a}+n^{-(b-1)/s}.
   \]

3. **Sharp multi-active achievability.** Example-norm-clipped, with-replacement SGD uses every coactive row during training. Natural singleton rows certify coordinate-wise curvature and clipping makes each rank-one update contractive. For
   \[
   \alpha=(a-b+1)/s,\qquad \beta=(b-1)/s,
   \]
   a label-free, distribution-aware anchor readout satisfies
   \[
   \mathbb E\mathcal R(\widehat w_T)
   \lesssim \frac{N^\alpha}{\eta T}+N^{-\beta}+e^{-cm}.
   \]
   The ordinary unmasked Polyak average requires no support-abort event and achieves the same power-law frontiers up to a logarithmic factor.

4. **Useful replay horizon.** With all data and `T=nK`, the sharp output reaches the coverage rate after
   \[
   K_{\rm sat}=O\!\left(\eta^{-1}n^{a/s-1}\right)
   \]
   epochs. A progressive schedule with
   \[
   N_\star\asymp\min\{n,m^s,(\eta T)^{s/a}\}
   \]
   attains the full capacity–optimization–coverage law.

5. **Input-only stopping rule.** Since conditional feature amplitude obeys
   \[
   q_j^2=\lambda_j/p_j\asymp p_j^{a/s-1},
   \]
   an unlabeled regression of empirical amplitudes against feature frequencies estimates the replay horizon before an expensive epoch sweep.

## Paper and proof files

- [`paper/main.tex`](paper/main.tex): expanded AAAI manuscript.
- [`paper/supplement.tex`](paper/supplement.tex): detailed proofs, constants, and experiment protocol.
- [`paper/references.bib`](paper/references.bib): verified bibliography.
- [`notes/coactive_sgd_proof.md`](notes/coactive_sgd_proof.md): readable proof derivation.
- [`notes/initial_proofs.md`](notes/initial_proofs.md): initial lower bounds and exact benchmark.
- [`notes/research_roadmap.md`](notes/research_roadmap.md): completed milestones and defensible open extensions.

## Reproduce and validate

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run a quick smoke test:

```bash
make quick
```

Regenerate the complete laptop-scale experiment release:

```bash
make full
make validate
make figures
```

Validate the checked-in tables, figures, and Python sources:

```bash
make check
```

The release contains **795 training runs** and **5,400 population-risk checkpoints**. The main numerical checks are:

- fixed-exponent two-term fit: `R^2 = 0.9805`, median relative error `0.0954`;
- optimization slope: `-0.4587` versus `-0.5` predicted;
- coverage slope: `-0.6548` versus `-0.6667` predicted;
- capacity slope: `-0.9911` versus `-1.0` predicted;
- progressive-schedule slope: `-0.5016` versus `-0.5` predicted;
- input-only saturation prediction: median factor error below `1.7x` across three spectral regimes;
- fixed-compute fresh-data advantage: `12.84x` in the primary sweep.

Checked-in outputs live under [`experiments/results`](experiments/results). `experiments/validate_results.py` verifies schemas, run counts, fitted exponents, and headline values. Figure PDFs use embedded Type 1 fonts.

## Claim boundary

The proved multi-active result assumes independent Bernoulli sparse coordinates, noiseless realizability, `s>1`, `a>=s`, source range `a-s+1<b<a+1`, with-replacement replay, example-norm clipping, and Polyak averaging. The sharp output mask is label-free but uses the population activation probabilities. Cyclic replay, random reshuffling, unaveraged last iterates, noisy labels, empirical anchor thresholds, and rotated or learned representations are reported only as empirical checks or future directions.
