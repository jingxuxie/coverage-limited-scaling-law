# Initial proof stack

This note proves the information-theoretic coverage floor, its sharp power-law constant, an exact finite-epoch one-hot law, and a matching result for genuinely multi-active inputs under singleton-filtered replay. The ordinary-SGD theorem using every coactive example remains open.

## 1. Model

Let

\[
X_j=q_jZ_jE_j,
\quad Z_j\sim\mathrm{Bernoulli}(p_j),
\quad E_j\sim\mathrm{Rad},
\]

independently over coordinates and examples. Then \(\lambda_j=\mathbb E X_j^2=p_jq_j^2\). Let the teacher be \(\Theta_j=\tau_j\Omega_j\), where \(\Omega_j\) are independent Rademacher signs, and set

\[
Y=\langle\Theta,X\rangle,
\qquad g_j=\lambda_j\tau_j^2.
\]

Assume \(\sum_j\lambda_j<\infty\), \(\sum_jg_j<\infty\), and

\[
p_j\asymp j^{-s},\ s>1;
\qquad
\lambda_j\asymp j^{-a};
\qquad
g_j\asymp j^{-b},\ b>1.
\]

For fixed-step results assume \(a\ge s\), so \(q_j^2\asymp j^{-(a-s)}\) does not grow with rank.

## 2. Universal coverage lower bound

**Theorem 1.** For every measurable, possibly randomized learner trained on \(n\) unique examples,

\[
\boxed{
\mathbb E_{\Omega,D_n}\mathcal R(\widehat f;\Theta)
\ge \sum_{j\ge1}g_j(1-p_j)^n.
}
\]

**Proof.** Let \(U(D_n)=\{j:Z_{1j}=\cdots=Z_{nj}=0\}\). Conditional on all training inputs, all teacher signs outside \(U\), and the learner's randomness, the labels do not depend on \(\{\Omega_j:j\in U\}\). These signs remain independent and symmetric. For a test input, write

\[
H(X)=\sum_{j\notin U}\tau_j\Omega_jX_j.
\]

Averaging over the hidden signs gives

\[
\begin{aligned}
&\mathbb E_{\Omega_U}
\left(\widehat f(X)-H(X)-\sum_{j\in U}\tau_j\Omega_jX_j\right)^2\\
&=(\widehat f(X)-H(X))^2+\sum_{j\in U}\tau_j^2X_j^2
\ge\sum_{j\in U}\tau_j^2X_j^2.
\end{aligned}
\]

Apply the identity first to finitely many unseen coordinates and pass to the limit by monotone convergence. Averaging over the test input gives \(\sum_{j\in U}g_j\), and \(\Pr(j\in U)=(1-p_j)^n\). \(\square\)

This is independent of optimizer, representation, compute, and epochs. Averaging over the sign prior also implies a deterministic worst-case teacher lower bound.

**Lemma 2.** If \(p_j\asymp j^{-s}\) and \(g_j\asymp j^{-b}\), then

\[
\boxed{\sum_jg_j(1-p_j)^n\asymp n^{-(b-1)/s}.}
\]

**Proof.** Let \(J=n^{1/s}\). For the upper bound, use \((1-p_j)^n\le e^{-np_j}\). The tail \(j>J\) is \(O(J^{1-b})\). For \(j\le J\),

\[
e^{-np_j}\le e^{-c(J/j)^s}.
\]

On dyadic shells \(j\in(J/2^{\ell+1},J/2^\ell]\), the sum is at most

\[
CJ^{1-b}2^{(b-1)\ell}e^{-c2^{s\ell}},
\]

whose series is finite. For the lower bound, choose fixed \(L\) so that \(np_j\le1/2\) on \([LJ,2LJ]\); then \((1-p_j)^n\ge1-np_j\ge1/2\), and that shell contributes \(\Omega(J^{1-b})\). \(\square\)

For exact profiles \(p_j=c_pj^{-s}\), \(g_j=c_gj^{-b}\), with \(0<c_p<1\), a rescaled Riemann sum gives the sharp constant

\[
\boxed{
\sum_jg_j(1-p_j)^n
\sim
\frac{c_g}{s}\Gamma\!\left(\frac{b-1}{s}\right)
(c_pn)^{-(b-1)/s}.
}
\]

Indeed, with \(J=(c_pn)^{1/s}\), the limiting integral is

\[
\int_0^\infty u^{-b}e^{-u^{-s}}du
=\frac1s\Gamma\!\left(\frac{b-1}{s}\right).
\]

For direct width \(m\), orthogonality yields

\[
\mathbb E\mathcal R(\widehat f)
\ge \sum_{j>m}g_j+\sum_{j\le m}g_j(1-p_j)^n
\asymp \min\{m,n^{1/s}\}^{1-b}.
\]

## 3. Exact one-hot replay law

In the one-hot benchmark, \(J_i\sim(p_j)\) and \(X_i=E_iq_{J_i}e_{J_i}\). Train a width-\(m\) linear model from zero by squared-loss SGD for \(K\) passes over the same \(n\) examples. Let \(C_j\sim\mathrm{Binomial}(n,p_j)\).

**Proposition 3.**

\[
\boxed{
\mathcal R_D
=\sum_{j\le m}g_j(1-\eta q_j^2)^{2KC_j}+\sum_{j>m}g_j,
}
\]

and

\[
\boxed{
\mathbb E_D\mathcal R_D
=\sum_{j\le m}g_j
[1-p_j+p_j(1-\eta q_j^2)^{2K}]^n
+\sum_{j>m}g_j.
}
\]

**Proof.** For residual \(\delta_j=w_j-\Theta_j\), an example on coordinate \(j\) gives \(\delta_j^+=(1-\eta q_j^2)\delta_j\). Coordinate updates commute, so coordinate \(j\) receives \(KC_j\) updates. The expectation follows from the binomial probability-generating function. \(\square\)

Assume \(\sup_jp_j\le\bar p<1\), \(0<\eta q_j^2\le\rho<1\), and the power laws above. Define

\[
J_\star=\min\{m,(\eta nK)^{1/a},n^{1/s}\}.
\]

**Theorem 4.**

\[
\boxed{
\mathbb E_D\mathcal R_D
\asymp J_\star^{1-b}
\asymp
m^{-(b-1)}+(\eta nK)^{-(b-1)/a}+n^{-(b-1)/s}.
}
\]

**Proof.** Put \(\alpha_j=\eta q_j^2\) and \(r_j=1-(1-\alpha_j)^{2K}\). Uniformly under the stability assumption,

\[
r_j\asymp\min\{1,K\alpha_j\}.
\]

Hence

\[
np_jr_j\asymp
\min\{np_j,\eta nK\lambda_j\}
\asymp\min\{nj^{-s},\eta nKj^{-a}\}.
\]

Because \(p_jr_j\le\bar p\), \((1-p_jr_j)^n\) is bounded above and below by exponentials with constant multiples of \(np_jr_j\) in the exponent. It remains to estimate

\[
S=\sum_{j\le m}j^{-b}e^{-c\min\{Aj^{-a},Bj^{-s}\}}+
\sum_{j>m}j^{-b},
\quad A=\eta nK,\ B=n.
\]

Let \(J=\min\{m,A^{1/a},B^{1/s}\}\). For \(j\le J\), since \(a\ge s\),

\[
\min\{Aj^{-a},Bj^{-s}\}\ge(J/j)^s.
\]

The dyadic-shell bound from Lemma 2 gives an \(O(J^{1-b})\) core, and all indices above \(J\) contribute at most \(O(J^{1-b})\). For the lower bound: if \(J=m\), the omitted tail is \(\Omega(J^{1-b})\); otherwise, either \(m<2J\) and the same is true, or \([J,2J]\) is represented and has exponent bounded by a constant, contributing \(\Omega(J^{1-b})\). \(\square\)

The saturation epoch is therefore

\[
\boxed{
K_{\mathrm{sat}}\asymp
\frac{\min\{m^a,n^{a/s}\}}{\eta n}.
}
\]

## 4. Genuine multi-active achievability

Return to independent Bernoulli masks. Since \(s>1\), \(\sum_jp_j<\infty\). If \(\sup_jp_j<1\), then

\[
\pi_0=\prod_{k\ge1}(1-p_k)>0.
\]

The probability that an example activates exactly coordinate \(j\) is

\[
\nu_j=p_j\prod_{k\ne j}(1-p_k)
=\frac{\pi_0p_j}{1-p_j}\asymp p_j.
\]

Consider **singleton-filtered replay SGD**: apply the usual update only when the observed support has size one; otherwise skip the sample. The filter is label-independent.

**Theorem 5.** If \(S_j\sim\mathrm{Binomial}(n,\nu_j)\) counts singleton-\(j\) examples, then

\[
\mathbb E_D\mathcal R_D^{\mathrm{sing}}
=\sum_{j\le m}g_j
[1-\nu_j+\nu_j(1-\eta q_j^2)^{2K}]^n
+\sum_{j>m}g_j,
\]

and

\[
\boxed{
\mathbb E_D\mathcal R_D^{\mathrm{sing}}
\asymp
m^{-(b-1)}+(\eta nK)^{-(b-1)/a}+n^{-(b-1)/s}.
}
\]

**Proof.** On a singleton-\(j\) sample, \(X=E_jq_je_j\) and \(Y=\Theta_jE_jq_j\), so the update is exactly the one-hot update. The binomial formula follows. Since \(\nu_j\asymp p_j\) and \(\nu_jq_j^2\asymp\lambda_j\), Theorem 4 applies with \(p_j\) replaced by \(\nu_j\). \(\square\)

As \(K\to\infty\), this matches the universal capacity/coverage lower bound up to constants. Thus the coverage exponent is statistically tight for genuinely multi-active data.

## 5. Remaining theorem

Ordinary SGD on a coactive sample obeys

\[
\delta_{t+1}=(I-\eta X_tX_t^\top)\delta_t,
\]

and the rank-one factors do not commute. The next target is the same three-term upper bound, possibly with logarithmic factors, for ordinary clipped or reshuffled replay SGD.

A concrete route is to truncate at \(J_c\asymp(n/\log n)^{1/s}\), concentrate the empirical diagonal there, control the whitened off-diagonal covariance, compare an epoch product with an empirical spectral filter, and charge the remaining target energy to the coverage tail. A systematic empirical dependence on coactivation density would instead indicate a fourth interference frontier.
