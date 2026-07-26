# Coactive Replay Proof

This note completes the main rigorous step left open in
[`initial_proofs.md`](initial_proofs.md): an upper bound for replay SGD that
**uses rather than filters out coactive examples**.  The result has two parts.

1. With all \(n\) unique examples, norm-clipped with-replacement replay reaches
   the coverage floor after
   \(K=\widetilde O(\eta^{-1}n^{a/s-1})\) epochs.
2. A frontier-matched progressive-data schedule achieves the complete
   capacity--optimization--coverage law, up to logarithms.

The proof is intentionally explicit about its scope.  It applies to clipped
SGD with Polyak averaging and an observable technical guard.  The experiments
also show the same exponents for unguarded last-iterate SGD throughout its
stable step-size range, but that stronger statement is empirical rather than
claimed as a theorem.

## 1. Sparse power-law model

For \(j\ge 1\), let

\[
X_j=q_j Z_j E_j,
\qquad
Z_j\sim\operatorname{Bernoulli}(p_j),
\qquad
E_j\sim\operatorname{Rad},
\]

independently over coordinates and examples.  Let

\[
\lambda_j=\mathbb E X_j^2=p_jq_j^2,
\qquad
Y=\langle \theta,X\rangle,
\qquad
g_j=\lambda_j\theta_j^2.
\]

Assume that for positive constants independent of \(j\),

\[
p_j\asymp j^{-s},
\qquad
\lambda_j\asymp j^{-a},
\qquad
g_j\asymp j^{-b},
\tag{A1}
\]

with

\[
s>1,\qquad a\ge s,\qquad b>1.
\tag{A2}
\]

We also assume \(\sup_jp_j<1\), that \(\lambda_j\) is nonincreasing up to a
constant factor, and, for the Euclidean source bound below,

\[
a-s+1<b<a+1.
\tag{A3}
\]

Condition (A3) includes the isotropic-teacher case \(b=a\), because \(s>1\).
It is the range in which the Euclidean norm of the teacher restricted to the
observed feature dictionary grows polynomially but remains integrable.

Since \(s>1\), \(\sum_jp_j<\infty\).  Hence every example has finite support
almost surely.  For \(N\) selected unique examples \(D_N=\{(X_i,Y_i)\}_{i=1}^N\),
write

\[
\mathcal S_N=\bigcup_{i=1}^N\operatorname{supp}(X_i),
\qquad
V_N=|\mathcal S_N|,
\qquad
A_N=\|\theta_{\mathcal S_N}\|_2^2.
\]

The sparse predictor stores the identities and weights of all features in
\(\mathcal S_N\), provided \(V_N\le m\).  This is a dictionary-memory
interpretation of width \(m\).  On this event, the selected training problem
is exactly realizable because every coordinate contributing to a selected
label belongs to \(\mathcal S_N\).

Population excess risk is

\[
\mathcal R(w)
=
\mathbb E_X\bigl[\langle w-\theta,X\rangle^2\bigr]
=
\sum_{j\ge1}\lambda_j(w_j-\theta_j)^2.
\tag{1}
\]

## 2. The coactive replay algorithm

Fix a base step size \(\eta>0\) and a clipping radius
\(\rho\in(0,1]\).  Assume

\[
\eta\sup_j q_j^2\le \rho.
\tag{A4}
\]

The assumption is compatible with a fixed step size because \(a\ge s\)
implies \(q_j^2\asymp j^{-(a-s)}\) is uniformly bounded.

At update \(t\), sample \(I_t\) uniformly from \(\{1,\ldots,N\}\) and set

\[
\gamma_i
=
\min\left\{\eta,\frac{\rho}{\|X_i\|_2^2}\right\},
\tag{2}
\]

with the second argument interpreted as \(+\infty\) when \(X_i=0\).  Starting
from \(w_0=0\), run

\[
w_{t+1}
=
w_t-\gamma_{I_t}X_{I_t}
\bigl(\langle X_{I_t},w_t\rangle-Y_{I_t}\bigr)
\tag{3}
\]

for \(T\) sample updates, and output the Polyak average

\[
\bar w_T=\frac1T\sum_{t=0}^{T-1}w_t.
\tag{4}
\]

Equation (3) is an ordinary squared-loss SGD update on every sampled row.
Clipping changes only the step size; it does not discard examples with
multiple active features.

### Observable guard

For a high-probability theorem, define the number of singleton-\(j\) rows

\[
C_{N,j}
=
\#\{i:\operatorname{supp}(X_i)=\{j\}\}.
\]

Let

\[
J_N
=
\left\lfloor
c_J\left(\frac{N}{\log(eN)}\right)^{1/s}
\right\rfloor
\tag{5}
\]

for a sufficiently small constant \(c_J\).  The good event is

\[
\mathcal G_N
=
\{V_N\le m\}
\cap
\left\{
C_{N,j}\ge c_0Np_j\ \text{for every }j\le J_N
\right\}.
\tag{6}
\]

Both conditions depend only on input supports and are observable before
training.  The guarded algorithm returns the zero predictor if
\(\mathcal G_N\) fails.  This guard is a technical device that makes the
expectation bound uniform over the heavy-tailed design.  Its failure
probability can be made smaller than any prescribed polynomial in \(N\);
the experiments run the unguarded optimizer.

## 3. Occupancy lemmas

### Lemma 1: singleton anchors

Let

\[
\pi_0=\prod_{k\ge1}(1-p_k).
\]

Then \(\pi_0>0\), and the probability that one example has support exactly
\(\{j\}\) is

\[
\nu_j
=
p_j\prod_{k\ne j}(1-p_k)
=
\frac{\pi_0p_j}{1-p_j}
\asymp p_j.
\tag{7}
\]

For every \(r>0\), the constant \(c_J\) in (5) can be chosen so that

\[
\Pr\left(
C_{N,j}\ge c_0Np_j
\text{ for all }j\le J_N
\right)
\ge 1-N^{-r}.
\tag{8}
\]

#### Proof

The infinite product is positive because \(\sum_jp_j<\infty\) and
\(\sup_jp_j<1\).  Equation (7) follows directly.

For each fixed \(j\),

\[
C_{N,j}\sim\operatorname{Binomial}(N,\nu_j).
\]

When \(j\le J_N\),

\[
N\nu_j
\ge cNp_j
\ge cc_J^{-s}\log(eN).
\]

A multiplicative Chernoff bound gives

\[
\Pr(C_{N,j}<N\nu_j/2)
\le \exp(-N\nu_j/8).
\]

Taking \(c_J\) sufficiently small makes the right-hand side at most
\(N^{-(r+2)}\).  A union bound over
\(J_N\le N^{1/s}\le N\) coordinates proves (8), after reducing the constant
in the lower bound from \(\nu_j\) to \(p_j\). \(\square\)

### Lemma 2: number of observed features

Let \(I_{N,j}=\mathbf 1\{j\in\mathcal S_N\}\).  The variables
\(\{I_{N,j}\}_{j\ge1}\) are independent, with

\[
\Pr(I_{N,j}=1)=1-(1-p_j)^N.
\]

Moreover,

\[
\mathbb EV_N
=
\sum_j[1-(1-p_j)^N]
\asymp N^{1/s}.
\tag{9}
\]

For exact \(p_j=c_pj^{-s}\),

\[
\mathbb EV_N
\sim
\Gamma(1-1/s)(c_pN)^{1/s}.
\tag{10}
\]

Consequently, there is a constant \(c_m>0\) such that, whenever

\[
N\le c_m m^s,
\tag{11}
\]

\[
\Pr(V_N>m)\le e^{-c m}.
\tag{12}
\]

#### Proof

Independence follows because the complete activation sequences
\(\{Z_{ij}:1\le i\le N\}\) are independent across \(j\).  Using
\(1-(1-p)^N\asymp\min\{1,Np\}\), split the sum at \(N^{1/s}\) to obtain (9).

For (10), rescale \(j=(c_pN)^{1/s}u\) and use the Riemann-sum limit

\[
\int_0^\infty(1-e^{-u^{-s}})\,du
=
\Gamma(1-1/s).
\]

Finally, choose \(c_m\) so that (11) implies
\(\mathbb EV_N\le m/2\), and apply a Chernoff bound for a sum of independent
Bernoulli variables.  The countable sum is justified by first truncating the
coordinates and then taking a monotone limit. \(\square\)

### Lemma 3: observed source norm

Set

\[
\alpha=\frac{a-b+1}{s},
\qquad
\beta=\frac{b-1}{s}.
\tag{13}
\]

Under (A3), \(0<\alpha<1\), \(\beta>0\), and

\[
\alpha+\beta=\frac as.
\tag{14}
\]

The observed Euclidean teacher norm satisfies

\[
\mathbb EA_N
=
\sum_j\theta_j^2[1-(1-p_j)^N]
\asymp N^\alpha.
\tag{15}
\]

For exact profiles
\(p_j=c_pj^{-s}\), \(\lambda_j=c_\lambda j^{-a}\), and
\(g_j=c_gj^{-b}\),

\[
\mathbb EA_N
\sim
\frac{c_g}{c_\lambda}
\frac{\Gamma(1-\alpha)}{a-b+1}
(c_pN)^\alpha.
\tag{16}
\]

#### Proof

Since \(\theta_j^2=g_j/\lambda_j\asymp j^{a-b}\), split at
\(J=N^{1/s}\).  The head is

\[
\sum_{j\le J}j^{a-b}\asymp J^{a-b+1},
\]

where \(b<a+1\) is used.  The tail is

\[
N\sum_{j>J}j^{a-b-s}
\asymp NJ^{a-b-s+1},
\]

where \(b>a-s+1\) is used.  Both expressions equal \(N^\alpha\).

For exact profiles, the rescaled limiting integral is

\[
\int_0^\infty
u^{a-b}(1-e^{-u^{-s}})\,du
=
\frac{\Gamma(1-\alpha)}{a-b+1}.
\]

This gives (16). \(\square\)

### Lemma 4: unseen-feature risk

For any \(b>1\),

\[
\mathbb E\sum_{j\notin\mathcal S_N}g_j
=
\sum_jg_j(1-p_j)^N
\asymp N^{-\beta}.
\tag{17}
\]

For exact profiles,

\[
\sum_jg_j(1-p_j)^N
\sim
\frac{c_g}{s}
\Gamma(\beta)
(c_pN)^{-\beta}.
\tag{18}
\]

This is Lemma 2 of `initial_proofs.md`.  It is also an algorithm-independent
lower bound under the random-sign teacher prior.

## 4. The contraction argument

### Lemma 5: one clipped update is a contraction

Let \(\delta_t=w_t-\theta_{\mathcal S_N}\).  For a sampled row \(x\),

\[
\delta_{t+1}
=
(I-\gamma xx^\top)\delta_t.
\]

If \(\gamma\|x\|_2^2\le\rho\le1\), then

\[
\|\delta_{t+1}\|_2^2
=
\|\delta_t\|_2^2
-
\gamma(2-\gamma\|x\|_2^2)
\langle x,\delta_t\rangle^2
\le
\|\delta_t\|_2^2
-
\gamma\langle x,\delta_t\rangle^2.
\tag{19}
\]

In particular,

\[
\|\delta_t\|_2\le\|\delta_0\|_2=\sqrt{A_N}
\quad\text{for every update path}.
\tag{20}
\]

### Lemma 6: singleton rows provide low-rank curvature

Condition on a dataset in \(\mathcal G_N\).  Then, for every current residual,

\[
\mathbb E_t\|\delta_{t+1}\|_2^2
\le
\|\delta_t\|_2^2
-
c\eta\sum_{j\le J_N}\lambda_j\delta_{t,j}^2,
\tag{21}
\]

where the expectation is over the uniform with-replacement choice of the next
training row.

#### Proof

Average (19) over the \(N\) rows.  Discarding all non-singleton contributions
can only weaken the decrease.  A singleton-\(j\) row equals
\(\pm q_je_j\).  By (A4), it is not clipped, so its contribution is
\(\eta q_j^2\delta_{t,j}^2\).  Hence

\[
\frac1N\sum_{i=1}^N
\gamma_i\langle X_i,\delta_t\rangle^2
\ge
\frac\eta N
\sum_{j\le J_N}C_{N,j}q_j^2\delta_{t,j}^2.
\]

On \(\mathcal G_N\), \(C_{N,j}\ge c_0Np_j\).  Since
\(p_jq_j^2=\lambda_j\), (21) follows. \(\square\)

The singleton rows are used only as a lower-bound certificate.  The optimizer
still trains on all sampled coactive rows, whose contributions are
nonnegative in (19).

## 5. Main conditional and expected bounds

### Theorem 1: conditional coactive replay bound

Condition on \(D_N\in\mathcal G_N\).  The Polyak average of clipped
with-replacement replay satisfies

\[
\boxed{
\mathbb E_{\mathrm{sgd}}\mathcal R(\bar w_T)
\le
\frac{A_N}{c\eta T}
+
C\lambda_{J_N+1}A_N
+
\sum_{j\notin\mathcal S_N}g_j.
}
\tag{22}
\]

#### Proof

Sum (21) from \(t=0\) to \(T-1\) and take expectations over the SGD indices:

\[
c\eta\sum_{t=0}^{T-1}
\mathbb E_{\mathrm{sgd}}
\sum_{j\le J_N}\lambda_j\delta_{t,j}^2
\le
\|\delta_0\|_2^2
=
A_N.
\tag{23}
\]

Convexity of the squared weighted norm gives

\[
\mathbb E_{\mathrm{sgd}}
\sum_{j\le J_N}\lambda_j\bar\delta_{T,j}^2
\le
\frac{A_N}{c\eta T}.
\tag{24}
\]

By (20),

\[
\|\bar\delta_T\|_2
\le
\frac1T\sum_{t=0}^{T-1}\|\delta_t\|_2
\le
\sqrt{A_N}.
\]

Monotonicity of the spectrum therefore yields

\[
\sum_{\substack{j>J_N\\j\in\mathcal S_N}}
\lambda_j\bar\delta_{T,j}^2
\le
C\lambda_{J_N+1}A_N.
\tag{25}
\]

Finally, the predictor is zero outside \(\mathcal S_N\), contributing exactly
\(\sum_{j\notin\mathcal S_N}g_j\).  Combining (24)--(25) proves (22).
\(\square\)

### Theorem 2: expected coactive replay bound

Fix \(r>\beta\).  Choose the anchor constant \(c_J\) for failure exponent
\(r\), and suppose \(N\le c_m m^s\).  The guarded algorithm satisfies

\[
\boxed{
\mathbb E\mathcal R(\bar w_T)
\le
C\left[
\frac{N^\alpha}{\eta T}
+
\{\log(eN)\}^{a/s}N^{-\beta}
+
N^{-r}
+
e^{-cm}
\right].
}
\tag{26}
\]

In particular, for \(m\gtrsim N^{1/s}\),

\[
\mathbb E\mathcal R(\bar w_T)
\le
C\left[
\frac{N^\alpha}{\eta T}
+
\{\log(eN)\}^{a/s}N^{-\beta}
\right].
\tag{27}
\]

#### Proof

On \(\mathcal G_N\), apply Theorem 1 and average over the dataset.  Lemma 3
controls the first term.  Since

\[
J_N\asymp
\left(\frac{N}{\log(eN)}\right)^{1/s},
\]

Lemmas 3 and the eigenvalue law give

\[
\lambda_{J_N+1}\mathbb EA_N
\lesssim
\left(\frac{N}{\log(eN)}\right)^{-a/s}
N^\alpha
=
\{\log(eN)\}^{a/s}N^{-\beta},
\]

using (14).  Lemma 4 controls the unseen term.

On \(\mathcal G_N^c\), the guarded algorithm outputs zero, whose risk is
\(\sum_jg_j<\infty\).  Lemmas 1--2 bound the failure probability by
\(N^{-r}+e^{-cm}\). \(\square\)

## 6. Consequences

### Corollary 1: all-data saturation

Use all \(n\) unique examples, assume \(m\gtrsim n^{1/s}\), and run
\(T=nK\) updates.  Then

\[
\boxed{
\mathbb E\mathcal R(\bar w_{nK})
\le
C\left[
\frac{n^{\alpha-1}}{\eta K}
+
\{\log(en)\}^{a/s}n^{-\beta}
\right].
}
\tag{28}
\]

Thus

\[
K_{\mathrm{sat}}
=
\widetilde O\left(
\eta^{-1}n^{a/s-1}
\right)
\tag{29}
\]

epochs suffice to reach the algorithm-independent coverage floor
\(n^{-\beta}\), up to logarithmic factors.  The power \(a/s-1\) matches the
exact one-hot replay law.  The theorem is a sufficiency result for coactive
clipped SGD; the universal lower bound concerns the final coverage floor, not
the required optimization time.

### Corollary 2: frontier-matched progressive schedule

Given at most \(n\) unique examples, dictionary budget \(m\), and update budget
\(T\), define

\[
J_\star
=
\min\left\{
m,\ n^{1/s},\ (\eta T)^{1/a}
\right\},
\tag{30}
\]

and select

\[
N_\star
=
\left\lfloor
c\min\left\{
n,\ m^s,\ (\eta T)^{s/a}
\right\}
\right\rfloor
\tag{31}
\]

unique examples, for a sufficiently small constant \(c\).  Run the guarded
clipped replay algorithm on this selected subset.  Then, for
\(J_\star\) above a fixed constant,

\[
\boxed{
\mathbb E\mathcal R(\bar w_T)
\le
C\{\log(eJ_\star)\}^{a/s}
J_\star^{\,1-b}.
}
\tag{32}
\]

Equivalently,

\[
\mathbb E\mathcal R(\bar w_T)
=
\widetilde O\left(
m^{-(b-1)}
+
(\eta T)^{-(b-1)/a}
+
n^{-(b-1)/s}
\right).
\tag{33}
\]

#### Proof

Equation (31) ensures \(N_\star\le c_mm^s\), so the dictionary condition
holds with high probability.  It also ensures

\[
\eta T\gtrsim N_\star^{a/s}.
\]

Therefore the first term in (27) satisfies

\[
\frac{N_\star^\alpha}{\eta T}
\lesssim
N_\star^{\alpha-a/s}
=
N_\star^{-\beta}.
\]

The second term is logarithmically larger than the same power.  Finally,
\(N_\star^{1/s}\asymp J_\star\), and
\(N_\star^{-\beta}=J_\star^{1-b}\).  For a negative power, the maximum of the
three frontier terms is equivalent within a factor three to their sum.
\(\square\)

The progressive schedule is a conservative achievability construction, not
a claim that discarding available data is empirically optimal.  In the
simulations, using more unique examples at fixed update count is consistently
beneficial.  Its role is to turn the all-data saturation theorem into a clean
uniform capacity--data--compute guarantee.

## 7. What is and is not resolved

The proof resolves the noncommuting-update obstruction for a concrete,
stable algorithm:

- every coactive row is retained;
- example-norm clipping makes each rank-one update a contraction;
- singleton rows that occur naturally in the same multi-active distribution
  certify coordinate-wise curvature;
- Polyak averaging converts cumulative contraction into a population-risk
  bound;
- occupancy and source asymptotics yield the scaling exponents.

The following stronger statements remain open:

1. the same finite-time theorem for a fixed unmodified step size throughout
   the heavy-tailed regime;
2. a sharp no-log theorem for all-data last-iterate SGD before saturation;
3. the corresponding theorem for random reshuffling or same-order cyclic
   replay;
4. random-feature sketches where the sparse coordinates are not directly
   addressable.

The experiments test all four boundaries.  Stable unclipped and last-iterate
runs follow the same exponents, while aggressive steps can diverge; clipping
removes that instability.
