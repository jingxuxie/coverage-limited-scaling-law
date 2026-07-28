from __future__ import annotations

from pathlib import Path

MAIN_BUDGET_HEADING = r"\subsection{Budgeted Data Acquisition and Replay}"
MAIN_ELASTICITY_HEADING = r"\subsection{Diagnosing the Active Frontier from Local Slopes}"
SUPP_BUDGET_HEADING = r"\section{Budgeted Acquisition and Replay}"
SUPP_ELASTICITY_HEADING = r"\section{Frontier Elasticities}"


def update_main() -> None:
    path = Path("paper/main.tex")
    text = path.read_text(encoding="utf-8")

    style = r"\bibliographystyle{../AAAI_AuthorKit27/aaai2027}"
    if style in text:
        text = text.replace(
            style,
            "% aaai2027.sty selects the required aaai2027 bibliography style.",
            1,
        )

    marker = (
        "The progressive subset is a conservative achievability construction rather\n"
        "than a claim that discarding data is optimal. At fixed update compute, the\n"
        "experiments consistently favor more unique examples.\n\n"
    )
    if MAIN_BUDGET_HEADING not in text:
        if marker not in text:
            raise SystemExit("main-paper budget insertion marker not found")
        budget = Path(".agent_expand/budget_main.tex").read_text(encoding="utf-8")
        text = text.replace(marker, marker + budget + "\n", 1)

    if MAIN_ELASTICITY_HEADING not in text:
        discussion = "\\section{Discussion and Limitations}\n"
        if discussion not in text:
            raise SystemExit("discussion insertion marker not found")
        elasticity = r'''
\subsection{Diagnosing the Active Frontier from Local Slopes}

The frontier law can be used as a diagnostic, not only as an asymptotic
rate. Write the three-term proxy as
\begin{equation}
L(m,T,N)=C_m m^{-\delta}+C_T(\eta T)^{-\gamma}+C_NN^{-\beta},
\label{eq:elasticity-proxy}
\end{equation}
where $\delta=b-1$, $\gamma=(b-1)/a$, and
$\beta=(b-1)/s$. Define the positive local elasticities
\[
e_m=-\partial_{\log m}\log L,\quad
e_T=-\partial_{\log T}\log L,\quad
e_N=-\partial_{\log N}\log L.
\]

\begin{proposition}[Frontier weights]
\label{prop:elasticity}
For the proxy in Eq.~\eqref{eq:elasticity-proxy},
\begin{equation}
\frac{e_m}{\delta}+\frac{e_T}{\gamma}+\frac{e_N}{\beta}=1.
\label{eq:elasticity-simplex}
\end{equation}
Moreover, each normalized elasticity equals the fraction of predicted
risk contributed by its corresponding term; for example,
\[
\frac{e_T}{\gamma}=
\frac{C_T(\eta T)^{-\gamma}}{L(m,T,N)}.
\]
\end{proposition}

\begin{proof}
Differentiate Eq.~\eqref{eq:elasticity-proxy} with respect to the three
logarithmic resources and divide by $L$. The three resulting risk
fractions sum to one.
\end{proof}

Equation~\eqref{eq:elasticity-simplex} gives a scale-local phase map
that does not require knowing the constants $C_m,C_T,C_N$. A
normalized width elasticity near one identifies a capacity bottleneck;
a normalized update elasticity near one indicates that replay is still
useful; and a normalized unique-data elasticity near one identifies a
coverage bottleneck. At a pairwise transition, the two corresponding
weights are comparable. In the optimization--coverage slice this
occurs when
$C_T(\eta T)^{-\gamma}\asymp C_NN^{-\beta}$, hence
$T\asymp\eta^{-1}N^{a/s}$ and
$K=T/N\asymp\eta^{-1}N^{a/s-1}$, recovering the saturation law.

The experiments probe all three corners of this simplex. The measured
small-width slope $-0.991$ is close to $-\delta=-1$; the early-epoch
slope $-0.459$ approaches $-\gamma=-1/2$; and the large-epoch data
slope $-0.655$ approaches $-\beta=-2/3$. Thus the observed changes in
slope are not separate empirical laws: they are different faces of one
resource decomposition. In practice the derivatives can be estimated
from neighboring logarithmic scale points. This suggests a simple
allocation loop: expand the resource with the largest normalized
elasticity, re-estimate after the next scale increase, and stop replay
when its weight becomes small relative to the data or capacity weight.
The rule is exact for the proxy and a testable heuristic for more
general sparse representations.

'''
        text = text.replace(discussion, elasticity + discussion, 1)

    path.write_text(text, encoding="utf-8")


def update_supplement() -> None:
    path = Path("paper/supplement.tex")
    text = path.read_text(encoding="utf-8")
    marker = "\\section{Input-Only Saturation Estimation}\n"
    if marker not in text:
        raise SystemExit("supplement insertion marker not found")

    insertion = ""
    if SUPP_BUDGET_HEADING not in text:
        insertion += Path(".agent_expand/budget_supp.tex").read_text(encoding="utf-8") + "\n"
    if SUPP_ELASTICITY_HEADING not in text:
        insertion += r'''
\section{Frontier Elasticities}

For completeness, let
\[
L=C_m m^{-\delta}+C_T(\eta T)^{-\gamma}+C_NN^{-\beta}.
\]
Direct differentiation gives
\[
-\frac{\partial\log L}{\partial\log m}
=\delta\frac{C_m m^{-\delta}}{L},\quad
-\frac{\partial\log L}{\partial\log T}
=\gamma\frac{C_T(\eta T)^{-\gamma}}{L},
\]
and the analogous identity for $N$. Dividing by the respective
exponents proves Eq.~\eqref{eq:elasticity-simplex}. Centered finite
differences on a logarithmic grid consistently estimate these
elasticities whenever the scaling proxy is locally accurate. The
identity also extends immediately to additional additive bottlenecks:
the normalized local elasticities form a simplex whose coordinates are
the corresponding fractions of predicted risk.

'''
    if insertion:
        text = text.replace(marker, insertion + marker, 1)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    update_main()
    update_supplement()
