# Sharp Coactive Replay Proof: Readable Derivation

This note records the proof architecture behind the main multi-active theorem. The full constant-level argument appears in `paper/supplement.tex`.

## 1. Model and the three occupancy quantities

Let

\[
X_j=q_jZ_jE_j,
\qquad Z_j\sim\mathrm{Bernoulli}(p_j),
\qquad E_j\sim\mathrm{Rad},
\]

independently, with

\[
p_j\asymp j^{-s},\qquad
\lambda_j=p_jq_j^2\asymp j^{-a},\qquad
g_j=\lambda_j\theta_j^2\asymp j^{-b}.
\]

Assume

\[
s>1,\qquad a\ge s,\qquad a-s+1<b<a+1.
\]

For `N` unique examples, write

\[
\mathcal S_N=\bigcup_{i\le N}\operatorname{supp}(X_i),
\quad V_N=|\mathcal S_N|,
\quad A_N=\|\theta_{\mathcal S_N}\|_2^2,
\quad U_N=\sum_{j\notin\mathcal S_N}g_j.
\]

Define

\[
\alpha=\frac{a-b+1}{s},
\qquad
\beta=\frac{b-1}{s}.
\]

Regular-variation calculations give

\[
\mathbb EV_N\asymp N^{1/s},
\qquad
\mathbb EA_N\asymp N^\alpha,
\qquad
\mathbb EU_N\asymp N^{-\beta},
\]

and the key identity

\[
\alpha+\beta=\frac as.
\]

`V_N` controls capacity, `A_N` is the Euclidean energy that finite-time SGD must dissipate, and `U_N` is the information-theoretic missing-feature floor.

## 2. Algorithm

Store the identities of all features observed in the selected `N` examples. At update `t`, sample one stored example uniformly with replacement and use

\[
\gamma_i=\min\left\{\eta,\frac{\rho}{\|X_i\|_2^2}\right\},
\qquad
w_{t+1}=w_t-\gamma_iX_i(X_i^\top w_t-Y_i).
\]

Every coactive row is used. Let

\[
\bar w_T=\frac1T\sum_{t<T}w_t.
\]

The base-step condition `eta sup_j q_j^2 <= rho <= 1` ensures that singleton examples use the base step.

## 3. Pathwise contraction resolves noncommutativity

Let `delta_t=w_t-theta_{S_N}`. Realizability gives

\[
\delta_{t+1}=(I-\gamma_iX_iX_i^\top)\delta_t.
\]

Although these rank-one matrices do not commute, clipping gives

\[
\gamma_i\|X_i\|^2\le1.
\]

Expanding the norm therefore yields the pathwise inequality

\[
\|\delta_{t+1}\|^2
\le
\|\delta_t\|^2-
\gamma_i(X_i^\top\delta_t)^2.
\]

The important point is qualitative: a rare, high-norm coactive example may reduce the effective step size, but it cannot increase Euclidean error. The proof never needs to diagonalize the epoch product.

## 4. Natural singleton rows provide curvature

Let `C_{N,j}` count examples whose support is exactly `{j}`. Because `s>1`,

\[
\pi_0=\prod_k(1-p_k)>0,
\qquad
\Pr(\operatorname{supp}(X)=\{j\})
=\frac{\pi_0p_j}{1-p_j}\asymp p_j.
\]

Average the contraction inequality over the next replay index and retain only the nonnegative singleton contributions. After summing over time,

\[
\frac{\eta}{N}
\sum_{t<T}\mathbb E_{\rm sgd}
\sum_jC_{N,j}q_j^2\delta_{t,j}^2
\le A_N.
\]

This is the core energy inequality. Singleton rows are not filtered by the algorithm; they are only a lower bound on the curvature already present in the full empirical objective.

## 5. Sharp anchor output

For a fixed `kappa <= pi_0/2`, define the label-free anchor set

\[
\mathcal A_N=\{j:C_{N,j}\ge\kappa Np_j\},
\qquad
\widehat w_T=P_{\mathcal A_N}\bar w_T.
\]

On `A_N`,

\[
\frac{C_{N,j}q_j^2}{N}
\ge\kappa p_jq_j^2
=\kappa\lambda_j.
\]

Jensen’s inequality converts the energy bound into

\[
\mathbb E_{\rm sgd}
\sum_{j\in\mathcal A_N}\lambda_j
(\bar w_{T,j}-\theta_j)^2
\le\frac{A_N}{\kappa\eta T}.
\]

A Chernoff bound gives

\[
\Pr(j\notin\mathcal A_N)\le e^{-cNp_j},
\]

so the expected target energy removed by the mask is

\[
\sum_jg_je^{-cNp_j}\asymp N^{-\beta}.
\]

Taking expectation over `A_N` yields

\[
\boxed{
\mathbb E\mathcal R(\widehat w_T)
\lesssim
\frac{N^\alpha}{\eta T}+N^{-\beta}+e^{-cm}.
}
\]

The final exponential term is the probability that the observed dictionary exceeds width `m` when `N` is below a constant multiple of `m^s`.

## 6. Plain Polyak output without a support abort

The ordinary output uses no mask. Choose a simultaneous-coverage core

\[
J_N\asymp\left(\frac{N}{\log(eN)}\right)^{1/s}.
\]

A union bound shows that the coordinates up to `J_N` have enough singleton anchors with overwhelming probability. The core is controlled by the same energy inequality. On the represented tail, Euclidean contraction gives

\[
\sum_{j>J_N}\lambda_j\bar\delta_j^2
\le\lambda_{J_N+1}A_N.
\]

Using `E A_N ~ N^alpha` and `alpha+beta=a/s`,

\[
\lambda_{J_N+1}\mathbb EA_N
\lesssim
\{\log(eN)\}^{a/s}N^{-\beta}.
\]

A weighted occupancy moment bound and Hölder’s inequality control the rare datasets where the core certificate fails; the algorithm itself never aborts on this event. Hence

\[
\boxed{
\mathbb E\mathcal R(\bar w_T)
\lesssim
\frac{N^\alpha}{\eta T}
+\{\log(eN)\}^{a/s}N^{-\beta}
+e^{-cm}.
}
\]

## 7. Saturation and the complete frontier

With all data, `N=n` and `T=nK`, the sharp bound becomes

\[
\mathbb E\mathcal R(\widehat w_{nK})
\lesssim
\frac{n^{\alpha-1}}{\eta K}+n^{-\beta}.
\]

Balancing the two terms and using `alpha+beta=a/s` gives

\[
K_{\rm sat}=O\left(\eta^{-1}n^{a/s-1}\right).
\]

For a total budget `T`, select

\[
N_\star\asymp\min\{n,m^s,(\eta T)^{s/a}\}.
\]

Substitution yields

\[
\mathbb E\mathcal R(\widehat w_T)
=O\left(
m^{-(b-1)}+(\eta T)^{-(b-1)/a}
+n^{-(b-1)/s}+e^{-cm}
\right).
\]

This is the capacity–optimization–coverage law. The progressive subset is an achievability construction, not a claim that discarding available data is optimal.

## 8. What remains open

The proof does not yet cover noisy labels, last-iterate sharp rates, cyclic or reshuffled replay, unknown-frequency plug-in anchor masks, correlated supports, or arbitrary rotations of the sparse representation. These are natural extensions; they should not be stated as established results in the paper.
