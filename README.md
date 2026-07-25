# Coverage-Limited Scaling Laws

This project studies when repeated passes over a fixed sparse dataset stop helping because predictive features are absent from the unique examples.

For feature activation probabilities \(p_j\asymp j^{-s}\), covariance eigenvalues \(\lambda_j\asymp j^{-a}\), and target energy \(g_j\asymp j^{-b}\), the proposed learned frontier is

\[
J_\star=\min\{m,(\eta nK)^{1/a},n^{1/s}\},
\]

with excess risk \(\mathcal E\asymp J_\star^{1-b}\).

## Proved in the initial stack

[`notes/initial_proofs.md`](notes/initial_proofs.md) proves:

1. an algorithm-independent unseen-feature lower bound
   \[
   \mathbb E\mathcal R\ge\sum_jg_j(1-p_j)^n;
   \]
2. the coverage floor \(n^{-(b-1)/s}\), including its sharp Gamma-function constant;
3. an exact finite-epoch one-hot replay formula and matching capacity/optimization/coverage law; and
4. the same exponents for genuinely multi-active Bernoulli data under label-independent singleton-filtered replay SGD.

The central open theorem is the finite-time upper bound for ordinary multi-pass SGD using every coactive example. The proof plan is in [`notes/research_roadmap.md`](notes/research_roadmap.md).

## Exact pilots

```bash
python -m pip install -r requirements.txt
python experiments/exact_pilots.py
```

The evaluator uses exact expectations rather than Monte Carlo. With \((s,a,b)=(1.5,2,2)\):

- predicted optimization slope: `-0.5`;
- predicted coverage slope: `-0.6667`;
- measured one-hot coverage slope: `-0.6666`;
- measured singleton-filtered coverage slope: `-0.6657`;
- predicted/measured one-hot plateau constants: `1.040975` / `1.040958`;
- predicted/measured singleton plateau constants: `2.667413` / `2.666811`.

Outputs are written below `experiments/results/`.
