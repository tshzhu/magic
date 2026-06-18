# Proofs

<style>
.theorem, .definition {
  border-left: 4px solid #4f6f9f;
  padding: 0.75rem 1rem;
  margin: 1rem 0;
  background: #f7f9fc;
}
.definition {
  border-left-color: #608b4e;
  background: #f8fbf6;
}
.proof {
  border-left: 3px solid #aaa;
  padding: 0.75rem 1rem;
  margin: 1rem 0;
  background: #fafafa;
}
.figure {
  margin: 1rem 0;
  text-align: center;
}
.figure img {
  max-width: 100%;
  height: auto;
  display: inline-block;
}
.caption {
  margin-top: -0.35rem;
  margin-bottom: 1.25rem;
  text-align: center;
  color: #555;
  font-size: 0.95rem;
}
.figure-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 0.75rem;
  align-items: center;
  margin: 1rem 0;
}
.figure-grid img {
  width: 100%;
  height: auto;
}
img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 1rem auto;
}
table {
  border-collapse: collapse;
  margin: 1rem auto;
}
th, td {
  border: 1px solid #ddd;
  padding: 0.35rem 0.55rem;
  vertical-align: top;
}
</style>


<div id="sec:proof of thm:ess retrieval"></div>
## Proofs of Theorem 1 and Corollary 1
We first introduce some additional notation. Define the unnormalized attention weights as $a_i = \exp(s_i)$ for $i \in \left\{1, \dots, n+m\right\}$. Let


```math align=center
Z_{\beta, n} = \sum_{i=1}^{n} a_i^{\beta},
    \ 
    \tilde{Z}_{\beta, m} = \sum_{j=1}^{m} a_{n+j}^{\beta},
    \ 
    Z_{\beta, n+m} = \sum_{i=1}^{n+m} a_i^{\beta}.
```

Thus $\alpha_i = \frac{a_i}{Z_{1, n+m}}$. The attention outputs for the signal, noise, and combined sequences are


```math align=center
\boldsymbol{o}_{n} = \sum_{i=1}^{n} \frac{a_i}{Z_{1, n}} \boldsymbol{v}_i,
    \ 
    \tilde{\boldsymbol{o}}_{m} = \sum_{j=1}^{m} \frac{a_{n+j}}{\tilde{Z}_{1, m}} \boldsymbol{v}_{n+j},
    \ 
    \boldsymbol{o}_{n+m} = \sum_{i=1}^{n+m} \frac{a_i}{Z_{1, n+m}} \boldsymbol{v}_i.
```

By Definition 3, the corresponding effective sequence lengths for $\beta \neq 1$ are


```math align=center
\mathcal{E}_{\beta, n} = \frac{Z_{1, n}^{\frac{\beta}{\beta - 1}}}{Z_{\beta, n}^{\frac{1}{\beta - 1}}},
    \ 
    \tilde{\mathcal{E}}_{\beta, m} = \frac{\tilde{Z}_{1, m}^{\frac{\beta}{\beta - 1}}}{\tilde{Z}_{\beta, m}^{\frac{1}{\beta - 1}}},
    \ 
    \mathcal{E}_{\beta, n+m} = \frac{Z_{1, n+m}^{\frac{\beta}{\beta - 1}}}{Z_{\beta, n+m}^{\frac{1}{\beta - 1}}}.
```

Since $Z_{\beta, n+m} = Z_{\beta, n} + \tilde{Z}_{\beta, m}$ for all $\beta > 0$, we can decompose


```math align=center
\begin{aligned}
        \boldsymbol{o}_{n+m} 
        &= \sum_{i=1}^{n} \frac{a_i}{Z_{1, n} + \tilde{Z}_{1, m}} \boldsymbol{v}_i + \sum_{j=1}^{m} \frac{a_{n+j}}{Z_{1, n} + \tilde{Z}_{1, m}} \boldsymbol{v}_{n+j} \\
        &= \frac{Z_{1, n}}{Z_{1, n} + \tilde{Z}_{1, m}} \sum_{i=1}^{n} \frac{a_i}{Z_{1, n}} \boldsymbol{v}_i + \frac{\tilde{Z}_{1, m}}{Z_{1, n} + \tilde{Z}_{1, m}} \sum_{j=1}^{m} \frac{a_{n+j}}{\tilde{Z}_{1, m}} \boldsymbol{v}_{n+j} \\
    \end{aligned}
```

Let $\eta \coloneqq \frac{\tilde{Z}_{1, m}}{Z_{1, n} + \tilde{Z}_{1, m}} = 1 - \frac{Z_{1, n}}{Z_{1, n} + \tilde{Z}_{1, m}} \in (0, 1)$. Then


```math align=center
\boldsymbol{o}_{n+m} - \boldsymbol{o}_n 
    = \eta \left( \sum_{j=1}^{m} \frac{a_{n+j}}{\tilde{Z}_{1, m}} \boldsymbol{v}_{n+j} - \sum_{i=1}^{n} \frac{a_i}{Z_{1, n}} \boldsymbol{v}_i \right)
    = \eta (\tilde{\boldsymbol{o}}_m - \boldsymbol{o}_n).
```

Therefore,


```math align=center
\| \boldsymbol{o}_{n+m} - \boldsymbol{o}_n \| \le \eta \| \tilde{\boldsymbol{o}}_{m} - \boldsymbol{o}_n \|.
```

Since $\tilde{\boldsymbol{o}}_{m}$ and $\boldsymbol{o}_n$ are convex combinations of the corresponding value vectors, we have


```math align=center
\| \tilde{\boldsymbol{o}}_{m} - \boldsymbol{o}_n \| 
    \le \| \tilde{\boldsymbol{o}}_{m} \| + \| \boldsymbol{o}_n \|
    \le \max_{1 \le i \le n} \| \boldsymbol{v}_i \| + \max_{1 \le j \le m} \| \boldsymbol{v}_{n+j} \| 
    < \infty.
```

Thus, it remains to bound $\eta$ in terms of $\Delta_{\beta}$. Moreover,


```math align=center
\begin{aligned}
        \mathcal{E}_{\beta, n+m}^{1-\beta} 
        &= \frac{Z_{\beta, n+m}}{Z_{1, n+m}^{\beta}}
        = \frac{Z_{\beta, n} + \tilde{Z}_{\beta, m}}{\left( Z_{1, n} + \tilde{Z}_{1, m} \right)^{\beta}} \\
        &= \left( \frac{Z_{1, n}}{Z_{1, n} + \tilde{Z}_{1, m}} \right)^{\beta} \frac{Z_{\beta, n}}{Z_{1, n}^{\beta}} + \left( \frac{\tilde{Z}_{1, m}}{Z_{1, n} + \tilde{Z}_{1, m}} \right)^{\beta} \frac{\tilde{Z}_{\beta, m}}{\tilde{Z}_{1, m}^{\beta}} \\
        &= (1 - \eta)^{\beta} \mathcal{E}_{\beta, n}^{1 - \beta} + \eta^{\beta} \tilde{\mathcal{E}}_{\beta, m}^{1 - \beta},
    \end{aligned}
```

Dividing by $\mathcal{E}_{\beta, n}^{1-\beta}$ gives


<div id="eqn:retrieval identity"></div>

```math align=center
(1 + \Delta_{\beta})^{1 - \beta} 
    = (1 - \eta)^{\beta} + \eta^{\beta} \left( \frac{\tilde{\mathcal{E}}_{\beta, m}}{\mathcal{E}_{\beta, n}} \right)^{1 - \beta}
    = (1 - \eta)^{\beta} + \kappa_{\beta} \eta^{\beta},
```

where $\kappa_{\beta} \coloneqq \left( \tilde{\mathcal{E}}_{\beta, m} / \mathcal{E}_{\beta, n} \right)^{1 - \beta} > 0$ for $\beta \neq 1$, since the generalized ESS satisfies $\mathcal{E}_{\beta, \bullet} \ge 1$. Since $\Delta_{\beta} > 0$, Bernoulli's inequality yields


```math align=center
\begin{aligned}
        (1 + \Delta_{\beta})^{1 - \beta} &\ge 1 + (1 - \beta) \Delta_{\beta}, \  & \mathrm{ if } &   \beta > 1, \\
        (1 + \Delta_{\beta})^{1 - \beta} &\le 1 + (1 - \beta) \Delta_{\beta}, \  & \mathrm{ if } &   0 < \beta < 1.
    \end{aligned}
```

For the right-hand side, since $\eta \in (0, 1)$, Taylor's theorem gives


```math align=center
(1 - \eta)^{\beta} = 1 - \beta \eta + \frac{\beta (\beta - 1)}{2} (1 - \xi)^{\beta - 2} \eta^2,
```

for some $\xi \in (0, \eta) \subset (0, 1)$. Thus, for the three ranges of $\beta$, we obtain


```math align=center
\begin{aligned}
        (1 - \eta)^{\beta} &\le 1 - \beta \eta + \frac{\beta (\beta - 1)}{2} \eta^2, \  & \mathrm{ if } &   \beta \ge 2, \\
        (1 - \eta)^{\beta} &\le 1 - \eta, \  & \mathrm{ if } &   1 < \beta < 2, \\
        (1 - \eta)^{\beta} &\ge 1 - \eta, \  & \mathrm{ if } &   0 < \beta < 1.
    \end{aligned}
```

Substituting these bounds into Eq. (9) yields


```math align=center
\begin{aligned}
        \eta &\le \frac{\beta - 1}{\beta} \Delta_{\beta} + \left( \frac{\beta - 1}{2} + \frac{\kappa_{\beta}}{\beta} \right) \eta^2, \  & \mathrm{ if } &   \beta \ge 2, \\
        \eta &\le (\beta - 1) \Delta_{\beta} + \kappa_{\beta} \eta^{\beta}, \  & \mathrm{ if } &   1 < \beta < 2, \\
        \eta^{\beta} &\le \frac{1 - \beta}{\kappa_{\beta}} \Delta_{\beta} + \frac{1}{\kappa_{\beta}} \eta, \  & \mathrm{ if } &   0 < \beta < 1.
    \end{aligned}
```

The following Lemma 1 unifies these three cases and provides the basis for the proof of Theorem 1.


<div class="theorem" id="lem:connectivity">

**Lemma 1.** For constants $\gamma > 1$ and $A, B > 0$, suppose $x \in [0, 1]$ satisfies $x \le A + B x^{\gamma}$. If $A < \left( \gamma B 2^{\gamma} \right)^{\frac{1}{1 - \gamma}}$, then $0 \le x \le 2 A$.

</div>

<div class="proof">

**Proof.**

Let $\Psi(x, t) = x - B x^{\gamma} - t A$, so $\partial_x \Psi(x, t) = 1 - \gamma B x^{\gamma - 1}$. Since $\Psi(0, 0) = 0$ and $\partial_x \Psi(0, 0) = 1 \neq 0$, the implicit function theorem yields a unique $C^1$ function $x(t)$ defined in a neighborhood of $0$ with $\Psi(x(t), t) = 0$ and $x(0) = 0$. Define


```math align=center
I = \left\{\tau \in [0, 1] : \forall t \in [0, \tau],   x(t) \mathrm{ exists, is continuous, and } x(t) \le 2 t A\right\}.
```

Clearly $0 \in I$, so $I \neq \varnothing$. We show $I = [0, 1]$ by the continuity method.


**1. Monotonicity of $x(t)$.** For any $\tau \in I$ and $t \in [0, \tau]$, differentiate $\Psi(x(t), t) = 0$ to obtain


```math align=center
x'(t) \left( 1 - \gamma B x(t)^{\gamma - 1} \right) = A.
```

Since $x(t) \le 2 t A \le 2 A$ and by the hypothesis $0 < A < \left( \gamma B 2^{\gamma} \right)^{\frac{1}{1 - \gamma}}$ we have


```math align=center
\begin{aligned}
        1 - \gamma B x(t)^{\gamma - 1} 
        \ge 1 - \gamma B (2 A)^{\gamma - 1}
        > 1 - \gamma B (2 \gamma B)^{-1}
        = \frac{1}{2} > 0,
    \end{aligned}
```

hence $x'(t) > 0$. Thus $x(t)$ is strictly increasing on $[0, \tau]$.


**2. Closedness of $I$.** Let $\tau_n \in I$ with $\tau_n \uparrow \tau^*$. Then $x(\tau_n)$ is increasing and bounded by $2 A$, so by the monotone convergence theorem the limit $x^* \coloneqq \lim_{n \to \infty} x(\tau_n)$ exists. Clearly $x^* \le 2 \tau^* A$. Hence


```math align=center
\begin{aligned}
        \partial_x \Psi(x^*, \tau^*) 
        = 1 - \gamma B (x^*)^{\gamma - 1}
        \ge 1 - \gamma B (2 A)^{\gamma - 1} > 0.
    \end{aligned}
```

By the implicit function theorem the solution extends uniquely to $x(\tau^*) \coloneqq x^*$. Therefore the property holds at $\tau^*$, and $I$ is closed.


**3. Openness of $I$.** For any $t \in I$, substituting $x(t) \le 2 t A$ into $x \le A + B x^{\gamma}$ gives


```math align=center
\begin{aligned}
        x(t) &= t A + B (x(t))^{\gamma} \\
        &\le t A + B (2 t A)^{\gamma} \\
        &= t A \left( 1 + B 2^{\gamma} t^{\gamma - 1} A^{\gamma - 1} \right) \\
        &\le t A \left( 1 + \gamma B 2^{\gamma} A^{\gamma - 1} \right) \\
        &< 2 t A,
    \end{aligned}
```

where the second inequality uses $\gamma > 1$ and $t \in [0, 1]$, and the last inequality follows from $0 < A < \left( \gamma B 2^{\gamma} \right)^{\frac{1}{1 - \gamma}}$. By the continuity of $x(t)$, $t$ is an interior point of $I$, so $I$ is open.


**4.** Since $I$ is nonempty, open and closed in the connected set $[0, 1]$, we have $I = [0, 1]$. In particular at $t = 1$ there is a solution $x(1)$ with $x(1)\le 2A$, completing the proof.

</div>
<div class="theorem">

**Restatement of Theorem 1.** The statement is given above.

</div>


<div class="proof">

**Proof.**

Let $B \coloneqq \max_{1 \le i \le n+m} \| \boldsymbol{v}_i \| < \infty$. Then the change in the output is bounded by


<div id="eqn:retrieval ub"></div>

```math align=center
\| \boldsymbol{o}_{n+m} - \boldsymbol{o}_n \| 
    \le \eta \| \tilde{\boldsymbol{o}}_{m} - \boldsymbol{o}_n \|
    \le 2 B \eta.
```

Consider four regimes depending on the range of $\beta$.


**1.** If $\beta \ge 2$, applying Lemma 1 with $\gamma = 2$ to


```math align=center
\eta \le \frac{\beta - 1}{\beta} \Delta_{\beta} + \left( \frac{\beta - 1}{2} + \frac{\kappa_{\beta}}{\beta} \right) \eta^2
```

gives


<div id="eqn:retrieval ub beta>=2"></div>

```math align=center
\eta \le \frac{2(\beta - 1)}{\beta} \Delta_{\beta}
    \  \mathrm{ if }
    \Delta_{\beta} 
    < \frac{\beta}{\beta - 1} \left( 8 \left( \frac{\beta - 1}{2} + \frac{\kappa_{\beta}}{\beta} \right) \right)^{-1}.
```

Combined with Eq. (10), the change in the output satisfies


```math align=center
\| \boldsymbol{o}_{n+m} - \boldsymbol{o}_n \| \lesssim \Delta_{\beta}.
```

**2.** If $1 < \beta < 2$, applying Lemma 1 with $\gamma = \beta$ to


```math align=center
\eta \le (\beta - 1) \Delta_{\beta} + \kappa_{\beta} \eta^{\beta},
```

we obtain


<div id="eqn:retrieval ub 1<beta<2"></div>

```math align=center
\eta \le 2 (\beta - 1) \Delta_{\beta}
    \  \mathrm{ if }
    \Delta_{\beta} 
    < \frac{1}{\beta - 1} \left( \kappa_{\beta} \beta 2^{\beta} \right)^{\frac{1}{1 - \beta}}.
```

Combined with Eq. (10), this implies


```math align=center
\| \boldsymbol{o}_{n+m} - \boldsymbol{o}_n \| \lesssim \Delta_{\beta}.
```

**3.** If $0 < \beta < 1$, applying Lemma 1 with $\gamma = 1 / \beta$ to


```math align=center
\eta^{\beta} \le \frac{1 - \beta}{\kappa_{\beta}} \Delta_{\beta} + \frac{1}{\kappa_{\beta}} \left( \eta^{\beta} \right)^{\frac{1}{\beta}}
```

yields


<div id="eqn:retrieval ub 0<beta<1"></div>

```math align=center
\eta \le (2 \Delta_{\beta})^{\frac{1}{\beta}}
    \  \mathrm{ if }
    \Delta_{\beta} 
    < \frac{\kappa_{\beta}}{1 - \beta} \left( \frac{2^{\frac{1}{\beta}}}{\kappa_{\beta} \beta} \right)^{\frac{\beta}{\beta - 1}}.
```

Combining this with Eq. (10) gives


```math align=center
\| \boldsymbol{o}_{n+m} - \boldsymbol{o}_n \| \lesssim \Delta_{\beta}^{\frac{1}{\beta}}.
```

**4.** If $\beta = 1$, note that $\mathcal{E}_1 = \exp(H_1)$, where $H_1$ is the Shannon entropy. Define


```math align=center
\begin{aligned}
    &H_{1, n} = -\sum_{i=1}^{n} \frac{a_i}{Z_{1, n}} \ln \frac{a_i}{Z_{1, n}},
    \ 
    \tilde{H}_{1, m} = -\sum_{j=1}^{m} \frac{a_{n+j}}{\tilde{Z}_{1, m}} \ln \frac{a_{n+j}}{\tilde{Z}_{1, m}},
    \\
    &H_{1, n+m} = -\sum_{i=1}^{n+m} \frac{a_i}{Z_{1, n+m}} \ln \frac{a_i}{Z_{1, n+m}}.
    \end{aligned}
```

By direct decomposition,


```math align=center
\begin{aligned}
        H_{1, n+m}
        &= -\sum_{i=1}^{n} \frac{a_i}{Z_{1, n+m}} \ln \frac{a_i}{Z_{1, n+m}} -\sum_{j=1}^{m} \frac{a_{n+j}}{Z_{1, n+m}} \ln \frac{a_{n+j}}{Z_{1, n+m}} \\
        &= -\sum_{i=1}^{n} \frac{(1 - \eta) a_i}{Z_{1, n}} \ln \frac{(1 - \eta) a_i}{Z_{1, n}} -\sum_{j=1}^{m} \frac{\eta a_{n+j}}{\tilde{Z}_{1, m}} \ln \frac{\eta a_{n+j}}{\tilde{Z}_{1, m}} \\
        &= - (1 - \eta) \sum_{i=1}^{n} \frac{a_i}{Z_{1, n}} \ln \frac{a_i}{Z_{1, n}} - \eta \sum_{j=1}^{m} \frac{a_{n+j}}{\tilde{Z}_{1, m}} \ln \frac{a_{n+j}}{\tilde{Z}_{1, m}} - (1 - \eta) \ln (1 - \eta) - \eta \ln \eta \\
        &\ge (1 - \eta) H_{1, n} + \eta \tilde{H}_{1, m} - \eta \ln \eta,
    \end{aligned}
```

where the last inequality uses $(1 - \eta) \ln (1 - \eta) \le 0$. Hence


```math align=center
\ln(\mathcal{E}_{1, n+m}) \ge (1 - \eta) \ln(\mathcal{E}_{1, n}) + \eta \ln(\tilde{\mathcal{E}}_{1, m}) - \eta \ln \eta.
```

Subtracting $\ln(\mathcal{E}_{1, n})$ gives


```math align=center
\ln(1 + \Delta_1) \ge \eta \ln \left( \frac{\tilde{\mathcal{E}}_{1, m}}{\mathcal{E}_{1, n}} \cdot \frac{1}{\eta} \right).
```

Let $\kappa_1 \coloneqq \tilde{\mathcal{E}}_{1, m} / \mathcal{E}_{1, n} > 0$. In analogy with the proof of Lemma 1, set $A \coloneqq \ln(1 + \Delta_1)$ and define $\Psi(x, t) = x \ln (\kappa_{\beta}/x) - t A$. By the implicit function theorem there exists a unique continuous solution $x(t)$ in a neighborhood of $0$ with $x(0) = 0$. Define


```math align=center
I = \left\{\tau \in [0, 1] : \forall t \in [0, \tau],   x(t) \mathrm{ exists, is continuous, and } x(t) \le \frac{t A}{\ln (\kappa_1 / t A)}\right\}.
```

Assume $A < \kappa_1 / e$. Then for any $t \in I \subseteq [0, 1]$,


```math align=center
\begin{aligned}
        \partial_x \Psi(x(t), t) 
        = \ln \frac{\kappa_1}{x(t)} - 1 
        \ge \ln \left( \frac{\kappa_1}{t A} \ln \frac{\kappa_1}{t A} \right) - 1 
        > \ln \left( \frac{e}{t} \ln \frac{e}{t} \right) - 1 
        \ge 0,
    \end{aligned}
```

and


```math align=center
\begin{aligned}
        x(t) 
        = \frac{t A}{\ln (\kappa_1 / x(t))} 
        \le \frac{t A}{\ln \left( \frac{\kappa_1}{t A} \ln \frac{\kappa_1}{t A} \right)} 
        < \frac{t A}{\ln \left( \frac{\kappa_1}{t A} \ln \frac{e}{t} \right)} 
        \le \frac{t A}{\ln (\kappa_1 / t A)}. 
    \end{aligned}
```

The same connectivity argument as in Lemma 1 implies $I = [0, 1]$. Hence,


<div id="eqn:retrieval ub beta=1"></div>

```math align=center
\eta \le \frac{\ln(1 + \Delta_1)}{\ln (\kappa_1 / \ln(1 + \Delta_1))}
    \  \mathrm{ if }
    \Delta_{\beta} 
    < e^{\kappa_1 / e} - 1.
```

Combined with Eq. (10), the change in the output is bounded by


```math align=center
\| \boldsymbol{o}_{n+m} - \boldsymbol{o}_n \| \lesssim \frac{\Delta_1}{\ln(1/\Delta_1)}.
```

**5.** Finally, since $1 \le \mathcal{E}_{\beta, n} \le n$ and $1 \le \tilde{\mathcal{E}}_{\beta, m} \le m$, we obtain


```math align=center
\begin{aligned}
        & n^{\frac{1}{\beta - 1}} \le \kappa_{\beta} \le n^{\frac{1}{\beta - 1}}, \  & \mathrm{ if } &   \beta > 1, \\
        & \frac{1}{n} \le \kappa_1 \le m, \  & \mathrm{ if } &   \beta = 1, \\
        & n^{\frac{1}{\beta - 1}} \le \kappa_{\beta} \le n^{\frac{1}{\beta - 1}}, \  & \mathrm{ if } &   0 < \beta < 1.
    \end{aligned}
```

The proof is completed by substituting these bounds into Eq. (11), Eq. (12), Eq. (13), Eq. (14).

</div>
<div class="theorem">

**Restatement of Corollary 1.** The statement is given above.

</div>


<div class="proof">

**Proof.**

For $\beta > 1$, define


```math align=center
R_1 \coloneqq \frac{\tilde{Z}_{1, m}}{Z_{1, n}} = \frac{\sum_{j=1}^{m} e^{s_{n+j}}}{\sum_{i=1}^{n} e^{s_i}},
    \ 
    R_{\beta} \coloneqq \frac{\tilde{Z}_{\beta, m}}{Z_{\beta, n}} = \frac{\sum_{j=1}^{m} e^{\beta s_{n+j}}}{\sum_{i=1}^{n} e^{\beta s_i}}.
```

By $\tilde{\mathcal{E}}_{\beta, m} \ge \mathcal{E}_{\beta, n}$ and Definition 3, we have


```math align=center
1 
    \le \left( \frac{\tilde{\mathcal{E}}_{\beta, m}}{\mathcal{E}_{\beta, n}}\right)^{\beta - 1} 
    = \left. \left( \frac{\sum_{j=1}^{m} e^{s_{n+j}}}{\sum_{i=1}^{n} e^{s_i}} \right)^{\beta} \right/ \left( \frac{\sum_{j=1}^{m} e^{\beta s_{n+j}}}{\sum_{i=1}^{n} e^{\beta s_i}} \right)
    = \frac{R_1^{\beta}}{R_{\beta}}.
```

Thus,


```math align=center
\begin{aligned}
        (1 + \Delta_{\beta})^{\beta - 1}
        &= \left( \frac{\mathcal{E}_{\beta, n+m}}{\mathcal{E}_{\beta, n}}\right)^{\beta - 1}
        = \left. \left( \frac{\sum_{i=1}^{n+m} e^{s_i}}{\sum_{i=1}^{n} e^{s_i}} \right)^{\beta} \right/ \left( \frac{\sum_{i=1}^{n+m} e^{\beta s_i}}{\sum_{i=1}^{n} e^{\beta s_i}} \right) \\
        &= \frac{(1 + R_1)^{\beta}}{1 + R_{\beta}}
        \ge \frac{(1 + R_1)^{\beta}}{1 + R_1^{\beta}}
        > \frac{1 + R_1^{\beta}}{1 + R_1^{\beta}} = 1,
    \end{aligned}
```

which implies $\Delta_{\beta} > 0$. The cases $0 < \beta < 1$ and $\beta = 1$ follow analogously.

</div>

<div id="sec:proof of thm:ess aggregation"></div>
## Proof of Theorem 2
<div class="theorem">

**Restatement of Theorem 2.** The statement is given above.

</div>


<div class="proof">

**Proof.**

Since $p$ is Riemann integrable and $\int_{0}^{1} p(x) \mathop{}\!\mathrm{d} x = 1$, the Riemann sums $R_n \coloneqq \frac{1}{n} \sum_{i=1}^{n} p(i / n)$ converge to $1$ as $n \to \infty$. Hence there exists $N \in \mathbb{N}$ such that for all $n > N$ we have $R_n \ge 1/2$, and therefore


```math align=center
R_n \ge \min\left\{R_1, \dots, R_N, 1/2\right\} \coloneqq K > 0.
```

For every $n$ and $1 \le i \le n$,


```math align=center
\frac{p(i / n)}{\sum_{j=1}^{n} p(j / n)} = \frac{p(i / n)}{n R_n} \le \frac{\sup_{x \in [0, 1]} p(x)}{n K}.
```

Since a Riemann integrable function on $[0, 1]$ is bounded, set $C \coloneqq \sup_{x \in [0, 1]} p(x) / K > 0$. Then the delocalization condition


<div id="eqn:aggregation delocalize"></div>

```math align=center
\max_{1 \le i \le n} \pi_i \le \frac{C}{n}
```

holds uniformly in $n$. Because $1 \le \mathcal{E}_{\beta}(\boldsymbol{\alpha}) \le n$, it suffices to prove $\mathcal{E}_{\beta}(\boldsymbol{\alpha}) = \Omega(n)$. We consider three cases depending on the value of $\beta$.


**1.** If $\beta > 1$, the condition $D_{\beta}(\boldsymbol{\alpha} \parallel \boldsymbol{\pi}) \le \varepsilon$ implies


```math align=center
\frac{1}{\beta - 1} \ln \left( \sum_{i=1}^n \alpha_i^{\beta} \pi_i^{1-\beta} \right) \le \varepsilon,
```

and since $\beta - 1 > 0$, exponentiating both sides gives


```math align=center
\sum_{i=1}^n \alpha_i^{\beta} \pi_i^{1-\beta} \le e^{\varepsilon(\beta-1)}.
```

By Eq. (15), $\pi_i^{1-\beta} \ge \left( C / n \right)^{1-\beta}$ for all $i \in \left\{1, \dots, n\right\}$. Substituting yields


```math align=center
C^{1-\beta} n^{\beta-1} \sum_{i=1}^n \alpha_i^{\beta} \le \sum_{i=1}^n \alpha_i^{\beta} \pi_i^{1-\beta} \le e^{\varepsilon(\beta-1)}.
```

Hence,


```math align=center
\sum_{i=1}^n \alpha_i^{\beta} \le e^{\varepsilon(\beta-1)} C^{\beta-1} n^{1-\beta}.
```

By the definition of $\mathcal{E}_{\beta}$,


```math align=center
\mathcal{E}_{\beta}(\boldsymbol{\alpha}) 
    = \left( \sum_{i=1}^n \alpha_i^{\beta} \right)^{\frac{1}{1-\beta}}
    \ge \left( e^{\varepsilon(\beta-1)} C^{\beta-1} n^{1-\beta} \right)^{\frac{1}{1-\beta}}
    = \frac{n}{C e^{\varepsilon}}.
```

**2.** If $0 < \beta < 1$, then from $D_{\beta}(\boldsymbol{\alpha} \parallel \boldsymbol{\pi}) \le \varepsilon$ we obtain


```math align=center
\sum_{i=1}^n \alpha_i^{\beta} \pi_i^{1-\beta} \ge e^{\varepsilon(\beta-1)}.
```

By Eq. (15), $\pi_i^{1 - \beta} \le \left( C / n \right)^{1 - \beta}$ for all $i \in \left\{1, \dots, n\right\}$. Substituting this bound gives


```math align=center
C^{1-\beta} n^{\beta-1} \sum_{i=1}^n \alpha_i^{\beta} \ge \sum_{i=1}^n \alpha_i^{\beta} \pi_i^{1-\beta} \ge e^{\varepsilon(\beta-1)}.
```

Thus,


```math align=center
\sum_{i=1}^n \alpha_i^{\beta} \ge e^{\varepsilon(\beta-1)} C^{\beta-1} n^{1-\beta}.
```

Raising both sides to the power $1 / (1-\beta)$ preserves the inequality, so


```math align=center
\mathcal{E}_{\beta}(\boldsymbol{\alpha}) \ge \left( e^{\varepsilon(\beta-1)} C^{\beta-1} n^{1-\beta} \right)^{\frac{1}{1-\beta}} = \frac{n}{C e^{\varepsilon}}.
```

**3.** For $\beta = 1$, the Rényi divergence equals the KL divergence,


```math align=center
D_{1}(\boldsymbol{\alpha} \parallel \boldsymbol{\pi}) = \sum_{i=1}^n \alpha_i \ln \frac{\alpha_i}{\pi_i} \le \varepsilon.
```

With the Shannon entropy $H_1(\boldsymbol{\alpha}) = - \sum_i \alpha_i \ln \alpha_i$, expanding the KL divergence gives


```math align=center
H_1(\boldsymbol{\alpha}) \ge -\sum_{i=1}^n \alpha_i \ln \pi_i - \varepsilon.
```

By Eq. (15), $-\ln \pi_i \ge \ln n - \ln C$. Therefore,


```math align=center
H_1(\boldsymbol{\alpha}) \ge \sum_{i=1}^n \alpha_i (\ln n - \ln C) - \varepsilon = \ln n - \ln C - \varepsilon.
```

Exponentiating both sides yields


```math align=center
\mathcal{E}_1(\boldsymbol{\alpha}) = \exp(H_1(\boldsymbol{\alpha})) \ge \exp(\ln n - \ln C - \varepsilon) = \frac{n}{C e^{\varepsilon}}.
```

</div>

<div id="sec:proof of pro:clt logits"></div>
## Proof of Proposition 1
<div class="theorem">

**Restatement of Proposition 1.** The statement is given above.

</div>


<div class="proof">

**Proof.**

Given the rotation matrix $\boldsymbol{R}$, the attention logit between $\boldsymbol{q}$ and $\boldsymbol{k}_i$ for $i \in \left\{1, \dots, L\right\}$ is


```math align=center
\begin{aligned}
        s_i 
        = \frac{1}{\sqrt{d}} \boldsymbol{q}^{\mathsf{T}} \boldsymbol{R}_{i-L} \boldsymbol{k}_i
        = \frac{1}{\sqrt{d}} \boldsymbol{q}^{\mathsf{T}} \boldsymbol{R}_{i-L} \left( \frac{1}{\sqrt{d}} \rho \boldsymbol{q} + \sigma \boldsymbol{z}_i \right)
        = \frac{\rho}{d} \boldsymbol{q}^{\mathsf{T}} \boldsymbol{R}_{i-L} \boldsymbol{q} + \frac{\sigma}{\sqrt{d}} \boldsymbol{q}^{\mathsf{T}} \boldsymbol{R}_{i-L} \boldsymbol{z}_i.
    \end{aligned}
```

Since $\boldsymbol{q} \sim \mathcal{N}(\boldsymbol{0}_d, \boldsymbol{I}_d)$, define


```math align=center
\begin{aligned}
        Y_d \coloneqq \frac{1}{d} \boldsymbol{q}^{\mathsf{T}} \boldsymbol{R}_{i-L} \boldsymbol{q},
        \ 
        \mathbb{E} Y_d = \frac{1}{d} \operatorname{Tr}(\boldsymbol{R}_{i-L})
        = \frac{2}{d} \sum_{f=0}^{d/2 - 1} \cos((i-L) \theta_f).
    \end{aligned}
```

Since $\boldsymbol{R}$ is orthogonal,


```math align=center
\| \boldsymbol{R} \|_F^2 = \operatorname{Tr}(\boldsymbol{R}^{\mathsf{T}} \boldsymbol{R}) = \operatorname{Tr}(\boldsymbol{I}_d) = d,
    \ 
    \| \boldsymbol{R} \| = \sqrt{\lambda_{\max}(\boldsymbol{R}^{\mathsf{T}} \boldsymbol{R})} = \sqrt{\lambda_{\max}(\boldsymbol{I}_d)} = 1.
```

By the Hanson–Wright inequality (see, e.g., Section 6.2: Vershynin (2026)[^vershynin2026high]), there exists a constant $c > 0$ such that, for any $\varepsilon \in (0, 1)$,


```math align=center
\mathbb{P}\left\{ \left| Y_d - \mathbb{E} Y_d \right| > \varepsilon\right\}
    \le 2 \exp\left( -c \min\left\{ \varepsilon^2 d, \varepsilon d \right\} \right)
    \le 2 \exp\left( -c \varepsilon^2 d \right).
```

Therefore,


```math align=center
\sum_{d=1}^{\infty} \mathbb{P}\left\{ \left| Y_d - \mathbb{E} Y_d \right| > \varepsilon\right\}
    \le \sum_{d=1}^{\infty} 2 \exp\left( -c \varepsilon^2 d \right)
    = 2 \sum_{d=1}^{\infty} \left( e^{-c \varepsilon^2} \right)^d < \infty.
```

By the Borel–Cantelli lemma, as $d \to \infty$,


```math align=center
Y_d
    \overset{\mathrm{a.s.}}{\longrightarrow} \lim_{d \to \infty} \mathbb{E} Y_d 
    = \lim_{d \to \infty} \frac{2}{d} \sum_{f=0}^{d/2 - 1} \cos\left( (i-L) \theta_f \right).
```

Let $\boldsymbol{b} = (b_1, \dots, b_L)^{\mathsf{T}}$, where $b_i \coloneqq \frac{\sigma}{\sqrt{d}} \boldsymbol{q}^{\mathsf{T}} \boldsymbol{R}_{i-L} \boldsymbol{z}_i$. Then


```math align=center
\begin{aligned}
        \boldsymbol{b} 
        = \frac{\sigma}{\sqrt{d}} \begin{pmatrix}
            \vdots \\ \boldsymbol{q}^{\mathsf{T}} \boldsymbol{R}_{i-L} \boldsymbol{z}_i \\ \vdots
        \end{pmatrix}
        = \frac{\sigma}{\sqrt{d}} \begin{pmatrix}
            \vdots \\ \sum_{j=1}^{d} q_j (\boldsymbol{R}_{i-L} \boldsymbol{z}_i)_j \\ \vdots
        \end{pmatrix}
        = \frac{\sigma}{\sqrt{d}} \sum_{j=1}^{d} q_j \begin{pmatrix}
            \vdots \\ (\boldsymbol{R}_{i-L} \boldsymbol{z}_i)_j \\ \vdots
        \end{pmatrix}
        \coloneqq \frac{1}{\sqrt{d}} \sum_{j=1}^{d} \boldsymbol{c}_j.
    \end{aligned}
```

Since $\tilde{\boldsymbol{z}}_i \coloneqq \boldsymbol{R}_{i-L} \boldsymbol{z}_i \mathrel{\overset{\mathrm{iid}}{\sim}} \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}_d)$, the vectors $\boldsymbol{c}_j \in \mathbb{R}^{L}$ are i.i.d. and satisfy


```math align=center
\mathbb{E}(\boldsymbol{c}_1) = \boldsymbol{0}_L, 
    \ 
    \mathrm{Cov}(\boldsymbol{c}_1) = \mathbb{E}\left( \sigma^2 q_1^2 \left( \tilde{z}_{i 1} \tilde{z}_{j 1} \right)_{i, j} \right) = \sigma^2 \boldsymbol{I}_L.
```

By the multivariate central limit theorem,


```math align=center
\boldsymbol{b} \overset{\mathcal{D}}{\longrightarrow} \mathcal{N}(\boldsymbol{0}_L, \sigma^2 \boldsymbol{I}_L) \  \mathrm{ as } d \to \infty.
```

By Slutsky's theorem,


```math align=center
(s_1, \dots, s_L) \overset{\mathcal{D}}{\longrightarrow} \mathcal{N}\left( (\mu_1, \dots, \mu_L), \sigma^2 \boldsymbol{I}_L \right) \  \mathrm{ as } d \to \infty,
```

where


```math align=center
\mu_i = \rho \lim_{d \to \infty} \frac{2}{d} \sum_{f=0}^{d/2 - 1} \cos\left( (i-L) \theta_f \right),
    \  \forall i \in \left\{1, \dots, L\right\}.
```

For NoPE, $\boldsymbol{R} = \boldsymbol{I}_d$, so $\theta_f = 0$ for all $f \in \left\{0, \dots, d/2-1\right\}$. Hence


```math align=center
\mu_i = \rho,
    \  \forall i \in \left\{1, \dots, L\right\}.
```

For RoPE, $\theta_f = b^{-2f/d}$ for $f \in \left\{0, \dots, d/2-1\right\}$. Thus


```math align=center
\mu_i = \rho \lim_{d \to \infty} \frac{2}{d} \sum_{f=0}^{d/2 - 1} \cos\left( (i-L) b^{-2f/d} \right)
    = \rho \int_{0}^{1} \cos\left( (i-L) b^{-x} \right) \mathop{}\!\mathrm{d} x,
    \  \forall i \in \left\{1, \dots, L\right\}.
```

</div>

<div id="sec:proof of pro:lln ess"></div>
## Proof of Proposition 2
We first analyze the limiting behavior of the partition functions $Z(\boldsymbol{s}; \lambda)$ and $Z(\boldsymbol{s}; \beta \lambda)$, as stated in Lemma 2.


<div class="theorem" id="lem:lln partition fucntion">

**Lemma 2.** Let the logits $s_1, \dots, s_L$ be independent Gaussian random variables with common variance $\sigma^2$, i.e., $s_i \sim \mathcal{N}(\mu_i, \sigma^2)$. Write $\boldsymbol{\mu} \coloneqq (\mu_1, \dots, \mu_L)$ and assume $\|\boldsymbol{\mu}\|_{\infty} \coloneqq \max_{1 \le i \le L} | \mu_i | < \infty$. Let the inverse temperature $\tau = \tau(L)$ possibly depend on $L$ (e.g., $\tau \in \left\{\lambda, \beta \lambda\right\}$), and define the scaling parameter


```math align=center
\Lambda \coloneqq \limsup_{L \to \infty} \frac{\tau(L) \sigma}{\sqrt{\ln L}}.
```

Let


```math align=center
S_L(\tau) = \sum_{i=1}^{L} e^{\tau s_i}.
```

Then:


1. If $0 \le \Lambda < \sqrt{2}$, then
  
  

```math align=center
\frac{S_L(\tau)}{\mathbb{E} S_L(\tau)} \overset{\mathbb{P}}{\longrightarrow} 1 \  \mathrm{ as } L \to \infty.
```

2. If $\Lambda = \sqrt{2}$ and
  
  

```math align=center
\lim_{L \to \infty} \frac{\mathcal{E}_{2}(\mathring{\boldsymbol{\alpha}})}{L / \sqrt{\ln L}} \to \infty,
              \  \mathrm{ where }  
              \mathcal{E}_{2}(\mathring{\boldsymbol{\alpha}}) \coloneqq \frac{\left( \sum_{i=1}^{L} e^{\tau \mu_i} \right)^2}{\sum_{i=1}^{L} e^{2 \tau \mu_i}},
```

  then
  
  

```math align=center
\frac{S_L(\tau)}{\mathbb{E} S_L(\tau)} \overset{\mathbb{P}}{\longrightarrow} \frac{1}{2} \  \mathrm{ as } L \to \infty.
```

3. If $\sqrt{2} < \Lambda < \infty$, the law of large numbers fails: the normalized sum $S_L(\tau) / \mathbb{E} S_L(\tau)$ can converge to a nondegenerate, unbounded random variable on $[0, \infty)$.
4. If $\Lambda = \infty$, then
  
  

```math align=center
\frac{S_L(\tau)}{M_L(\tau)} \overset{\mathbb{P}}{\longrightarrow} 1   \mathrm{ as } L \to \infty,
```

  where $M_L(\tau) \coloneqq \max_{1 \le i \le L} e^{\tau s_i}$.

</div>

<div class="proof">

**Proof.**

For each $s_i \sim \mathcal{N}(\mu_i, \sigma^2)$, $e^{\tau s_i}$ is log-normal with expectation $\mathbb{E} e^{\tau s_i} = e^{\tau \mu_i + \frac{1}{2} \tau^2 \sigma^2}$.


**(i)** If $0 \le \Lambda < \sqrt{2}$, let


```math align=center
\bar{S}_L(\tau) \coloneqq \frac{S_L(\tau)}{\mathbb{E} S_L(\tau)}.
```

It suffices to show that for some $r > 1$, $\lim_{L \to \infty} \mathbb{E} |\bar{S}_L(\tau) - 1|^r = 0$, which implies $\bar{S}_L(\tau) \overset{\mathbb{P}}{\longrightarrow} 1$ as $L \to \infty$. By the von Bahr–Esseen inequality [Theorem 2: von Bahr and Esseen, 1965][^von1965inequalities], for any $r \in [1, 2]$,


```math align=center
\begin{aligned}
        \mathbb{E} |\bar{S}_L(\tau) - 1|^r 
        = \mathbb{E} \left| \sum_{i=1}^{L} \frac{e^{\tau s_i} - \mathbb{E} e^{\tau s_i}}{\mathbb{E} S_L(\tau)} \right|^r
        \le 2 \sum_{i=1}^{L} \mathbb{E} \left| \frac{e^{\tau s_i} - \mathbb{E} e^{\tau s_i}}{\mathbb{E} S_L(\tau)} \right|^r
        = 2 \frac{\sum_{i=1}^{L} \mathbb{E} | e^{\tau s_i} - \mathbb{E} e^{\tau s_i} |^r}{ \left| \sum_{i=1}^{L} \mathbb{E} e^{\tau s_i} \right|^r }.
    \end{aligned}
```

For the denominator, since $\mu_i \ge - \|\boldsymbol{\mu}\|_{\infty}$,


```math align=center
\sum_{i=1}^{L} \mathbb{E} e^{\tau s_i}
    = \sum_{i=1}^{L} e^{\tau \mu_i + \frac{1}{2} \tau^2 \sigma^2}
    \ge L e^{- \tau \|\boldsymbol{\mu}\|_{\infty} + \frac{1}{2} \tau^2 \sigma^2}.
```

For the numerator, using the power mean inequality $(x + y)^r \le 2^{r-1} (x^r + y^r)$ for $r \ge 1$ and $x, y \ge 0$,


```math align=center
\begin{aligned}
        \sum_{i=1}^{L} \mathbb{E} | e^{\tau s_i} - \mathbb{E} e^{\tau s_i} |^r
        &\le \sum_{i=1}^{L} \mathbb{E} ( e^{\tau s_i} + \mathbb{E} e^{\tau s_i} )^r \\
        &\le 2^{r-1} \sum_{i=1}^{L} \left[ \mathbb{E} e^{r \tau s_i} + (\mathbb{E} e^{\tau s_i})^r \right] \\
        &\le 2^{r-1} L ( e^{r \tau \|\boldsymbol{\mu}\|_{\infty} + \frac{1}{2} r^2 \tau^2 \sigma^2} + e^{r \tau \|\boldsymbol{\mu}\|_{\infty} + \frac{1}{2} r \tau^2 \sigma^2} ).
    \end{aligned}
```

Hence,


```math align=center
\begin{aligned}
        \mathbb{E} |\bar{S}_L(\tau) - 1|^r 
        &\le 2^r L^{1-r} \frac{e^{r \tau \|\boldsymbol{\mu}\|_{\infty} + \frac{1}{2} r^2 \tau^2 \sigma^2} + e^{r \tau \|\boldsymbol{\mu}\|_{\infty} + \frac{1}{2} r \tau^2 \sigma^2}}{e^{- r \tau \|\boldsymbol{\mu}\|_{\infty} + \frac{1}{2} r \tau^2 \sigma^2}} \\
        &= 2^r L^{1-r} \left( e^{2 r \tau \|\boldsymbol{\mu}\|_{\infty} + \frac{1}{2} r(r-1) \tau^2 \sigma^2} + e^{2 r \tau \|\boldsymbol{\mu}\|_{\infty}} \right) \\
        &= 2^r \exp\left( -(r-1) \ln L + 2 r \tau \|\boldsymbol{\mu}\|_{\infty} + \frac{1}{2} r(r-1) \tau^2 \sigma^2 \right) (1 + o(1)).
    \end{aligned}
```

Thus $\lim_{L \to \infty} \mathbb{E} |\bar{S}_L(\tau) - 1|^r = 0$ provided the exponent


```math align=center
\frac{1}{2} r(r-1) \tau^2 \sigma^2 -(r-1) \ln L 
    = \frac{r(r-1)}{2} \ln L \cdot \left( \frac{\tau^2 \sigma^2}{\ln L} - \frac{2}{r} \right)
    \to -\infty
    \  \mathrm{ as } L \to \infty.
```

Since $\Lambda < \sqrt{2}$, there exists $\varepsilon > 0$ such that for sufficiently large $L$, $0 \le \tau^2 \sigma^2 / \ln L < 2 - \varepsilon$. Choose any $r \in (1, 2/(2-\varepsilon)) \subset (1, 2)$; then for large $L$ we have $\tau^2 \sigma^2 / \ln L < 2 / r$, so the exponent tends to $-\infty$. Hence some $r > 1$ satisfies $\lim_{L \to \infty} \mathbb{E}|\bar{S}_L(\tau) - 1|^r = 0$, which completes the proof.


**(ii)** If $\Lambda = \sqrt{2}$, write $s_i = \mu_i + \sigma z_i$ with $z_i \mathrel{\overset{\mathrm{iid}}{\sim}} \mathcal{N}(0, 1)$. Decompose $S_L(\tau)$ as $S_L(\tau) = S_L^{\le}(\tau) + S_L^{>}(\tau)$, where


```math align=center
S_L^{\le}(\tau) = \sum_{i=1}^{L} e^{\tau s_i} \mathbb{1}_{\left\{z_i \le \tau \sigma\right\}},
    \ 
    S_L^{>}(\tau) = \sum_{i=1}^{L} e^{\tau s_i} \mathbb{1}_{\left\{z_i > \tau \sigma\right\}}.
```

**1. Bounding $S_L^{>}(\tau)$.** For any $x > 0$, the Mills' ratio gives $\Phi(-x) \le \phi(x) / x$, where $\Phi$ and $\phi$ denote the CDF and PDF of the standard normal distribution, respectively (see, e.g., Proposition 2.1.2: Vershynin (2026)[^vershynin2026high]). By the union bound,


```math align=center
\begin{aligned}
        \mathbb{P}\left( \max_{1 \le i \le L} z_i > \tau \sigma \right)
        \le \sum_{i=1}^{L} \mathbb{P}(z_i > \tau \sigma)
        = L \Phi(- \tau \sigma)
        \le L \cdot \frac{1}{\tau \sigma \sqrt{2 \pi}} e^{-\frac{1}{2} \tau^2 \sigma^2}.
    \end{aligned}
```

Substituting $\tau \sigma = \sqrt{2 \ln L}$, the right-hand side equals $1 / (2 \sqrt{\pi \ln L}) \to 0$ as $L \to \infty$. It follows that


```math align=center
\mathbb{P}\left( S_L^{>}(\tau) = 0 \right) 
    \ge \mathbb{P}\left( \max_{1 \le i \le L} z_i \le \tau \sigma \right)
    = 1 - \mathbb{P}\left( \max_{1 \le i \le L} z_i > \tau \sigma \right) 
    \to 1 \  \mathrm{ as } L \to \infty.
```

Therefore $S_L^{>}(\tau) \overset{\mathbb{P}}{\longrightarrow} 0$ as $L \to \infty$.


**2. Bounding $S_L^{\le}(\tau)$.** Set $Y_i \coloneqq e^{\tau \sigma z_i} \mathbb{1}_{\left\{z_i \le \tau \sigma\right\}}$. Then


```math align=center
\begin{aligned}
        \mathbb{E} Y_i
        = \int_{-\infty}^{\tau \sigma} e^{\tau \sigma x} \frac{1}{\sqrt{2 \pi}} e^{-\frac{x^2}{2}} \mathop{}\!\mathrm{d} x
        = e^{\frac{1}{2} \tau^2 \sigma^2} \int_{-\infty}^{\tau \sigma} \frac{1}{\sqrt{2 \pi}} e^{-\frac{(x - \tau \sigma)^2}{2}} \mathop{}\!\mathrm{d} x
        = \frac{1}{2} e^{\frac{1}{2} \tau^2 \sigma^2}.
    \end{aligned}
```

Hence


```math align=center
\mathbb{E} S_L^{\le}(\tau)
    = \sum_{i=1}^{L} e^{\tau \mu_i} \cdot \frac{1}{2} \mathbb{E} e^{\tau \sigma z_i}
    = \frac{1}{2} \sum_{i=1}^{L} \mathbb{E} e^{\tau \sigma s_i}
    = \frac{1}{2} \mathbb{E} S_L(\tau).
```

Moreover,


```math align=center
\begin{aligned}
        \mathbb{E} Y_i^2
        = \int_{-\infty}^{\tau \sigma} e^{2 \tau \sigma x} \frac{1}{\sqrt{2 \pi}} e^{-\frac{x^2}{2}} \mathop{}\!\mathrm{d} x
        = e^{2 \tau^2 \sigma^2} \int_{-\infty}^{\tau \sigma} \frac{1}{\sqrt{2 \pi}} e^{-\frac{(x - 2 \tau \sigma)^2}{2}} \mathop{}\!\mathrm{d} x
        = e^{2 \tau^2 \sigma^2} \Phi(- \tau \sigma),
    \end{aligned}
```

By Mills' ratio,


```math align=center
\mathbb{E} Y_i^2 
    \le e^{2 \tau^2 \sigma^2} \frac{1}{\tau \sigma \sqrt{2 \pi}} e^{- \frac{\tau^2 \sigma^2}{2}} 
    = \frac{1}{\tau \sigma \sqrt{2 \pi}} e^{\frac{3}{2} \tau^2 \sigma^2}
```

Therefore,


```math align=center
\begin{aligned}
        \frac{\mathrm{Var}(S_L^{\le}(\tau))}{\mathbb{E} S_L^{\le}(\tau)^2} 
        &= \frac{\sum_{i=1}^{L} e^{2 \tau \mu_i} \mathrm{Var}(Y_i)}{( \frac{1}{2} \sum_{i=1}^{L} e^{\tau \mu_i + \frac{1}{2} \tau^2 \sigma^2} )^2}
        \le \frac{\sum_{i=1}^{L} e^{2 \tau \mu_i} \mathbb{E}(Y_i^2)}{( \frac{1}{2} \sum_{i=1}^{L} e^{\tau \mu_i + \frac{1}{2} \tau^2 \sigma^2} )^2} \\
        &= \frac{4}{\tau \sigma \sqrt{2 \pi}} e^{\frac{1}{2} \tau^2 \sigma^2} \frac{\sum_{i=1}^{L} e^{2 \tau \mu_i}}{(\sum_{i=1}^{L} e^{\tau \mu_i})^2}
        = \frac{2 L}{\sqrt{\pi \ln L}} \cdot \frac{1}{\mathcal{E}_2(\mathring{\boldsymbol{\alpha}})} \to 0 \  \mathrm{ as } L \to \infty,
    \end{aligned}
```

where we used $\tau \sigma = \sqrt{2\ln L}$ in the last step and $\mathcal{E}_2(\mathring{\boldsymbol{\alpha}})$ denotes the effective sample size defined by $( \sum_{i=1}^{L} e^{\tau\mu_i} )^2 / ( \sum_{i=1}^{L} e^{2\tau\mu_i} )$. By Chebyshev's inequality, for any $\delta > 0$,


```math align=center
\mathbb{P}\left( \left| \frac{S_L^{\le}(\tau)}{\mathbb{E} S_L^{\le}(\tau)} - 1 \right| > \delta \right) 
    \le \frac{1}{\delta^2} \frac{\mathrm{Var}(S_L^{\le}(\tau))}{\mathbb{E} S_L^{\le}(\tau)^2} \to 0 \  \mathrm{ as } L \to \infty.
```

Finally,


```math align=center
\frac{S_L(\tau)}{\mathbb{E} S_L(\tau)} 
    = \frac{S_L^{\le}(\tau)}{\mathbb{E} S_L(\tau)} + \frac{S_L^{>}(\tau)}{\mathbb{E} S_L(\tau)}
    = \frac{S_L^{\le}(\tau)}{2 \mathbb{E} S_L^{\le}(\tau)} + \frac{S_L^{>}(\tau)}{\mathbb{E} S_L(\tau)}
    \overset{\mathbb{P}}{\longrightarrow} \frac{1}{2}
    \  \mathrm{ as } L \to \infty.
```

**(iii)** Consider the case $\sqrt{2} < \Lambda < \infty$. In the simplest setting, where the $s_i$ are i.i.d. standard Gaussian (i.e., $\mu_i \equiv 0$ and $\sigma = 1$), $S_L(\tau)$ converges in distribution to a nondegenerate stable law on $[0, \infty)$ (see, e.g., Proposition 3.1: Molchanov and Panov (2019)[^molchanov2019limit] and Theorem 3: Ben Arous et al. (2005)[^ben2005limit]). Hence the law of large numbers fails.


**(iv)** If $\Lambda = \infty$, define $\zeta_L \coloneqq \frac{S_L(\tau) - M_L(\tau)}{M_L(\tau)}$; it suffices to show $\zeta_L \overset{\mathbb{P}}{\longrightarrow} 0$ as $L \to \infty$. Let $s_{(L)} \coloneqq \max_{1 \le i \le L} s_i$ denote the largest logit, so that $M_L(\tau) = \max_{1 \le i \le L} e^{\tau s_i} = e^{\tau s_{(L)}}$. Denote by $F_i(z)$ and $f_i(z)$ the CDF and PDF of $s_i$, respectively.


**1. Bounding $s_{(L)}.$.** Let $F_{(L)}(z) \coloneqq \prod_{i=1}^{L} F_i(z)$ be the CDF of $s_{(L)}$. Since $F_{(L)}(\cdot)$ is non-decreasing, choose $\gamma_L \coloneqq \min\left\{\ln L, \left( \tau / \sqrt{\ln L} \right)^{1/2}\right\}$ and define $s_* \coloneqq \inf\left\{z : F_{(L)}(z) = e^{-\gamma_L}\right\}$. Then $\mathbb{P}(s_{(L)} < s_*) = e^{-\gamma_L} \to 0$ as $L \to \infty$. Moreover,


```math align=center
\gamma_L 
    = - \sum_{i=1}^{L} \ln F_i(s_*)
    = - \sum_{i=1}^{L} \ln \Phi\left(\frac{s_* - \mu_i}{\sigma}\right)
    \ge -L \ln \Phi\left(\frac{s_* + \|\boldsymbol{\mu}\|_{\infty}}{\sigma}\right).
```

Hence


```math align=center
- \frac{\gamma_L}{L} \le \ln \Phi\left(\frac{s_* + \|\boldsymbol{\mu}\|_{\infty}}{\sigma}\right) \le 0.
```

Letting $L \to \infty$ implies $\Phi\left(\frac{s_* + \|\boldsymbol{\mu}\|_{\infty}}{\sigma}\right) \to 1$, and therefore $s_* \to \infty$. On the other hand,


```math align=center
\gamma_L 
    = - \sum_{i=1}^{L} \ln \Phi\left(\frac{s_* - \mu_i}{\sigma}\right)
    \le -L \ln \Phi\left(\frac{s_* - \|\boldsymbol{\mu}\|_{\infty}}{\sigma}\right).
```

If $\frac{s_* - \|\boldsymbol{\mu}\|_{\infty}}{\sigma} \ge 1$, then $\Phi\left(\frac{s_* - \|\boldsymbol{\mu}\|_{\infty}}{\sigma}\right) \ge \frac{1}{2}$. For $x \in \left[ \frac{1}{2}, 1 \right)$, we have $\ln x \ge 1 - \frac{1}{x} = -\frac{1-x}{x} \ge -2 (1 - x)$. Thus, by Mills' ratio,


```math align=center
\begin{aligned}
        \ln \Phi\left(\frac{s_* - \|\boldsymbol{\mu}\|_{\infty}}{\sigma}\right)
        &\ge -2 \left( 1 - \Phi\left(\frac{s_* - \|\boldsymbol{\mu}\|_{\infty}}{\sigma}\right) \right) \\
        &\ge -2 \frac{1}{\sqrt{2 \pi} \left(\frac{s_* - \|\boldsymbol{\mu}\|_{\infty}}{\sigma}\right)} \exp\left( -\frac{1}{2} \left(\frac{s_* - \|\boldsymbol{\mu}\|_{\infty}}{\sigma}\right)^2 \right) \\
        &\ge - \exp\left( -\frac{1}{2} \left(\frac{s_* - \|\boldsymbol{\mu}\|_{\infty}}{\sigma}\right)^2 \right).
    \end{aligned}
```

Therefore, for $L \ge 3$,


```math align=center
1 \le \gamma_L \le L \exp\left( -\frac{1}{2} \left(\frac{s_* - \|\boldsymbol{\mu}\|_{\infty}}{\sigma}\right)^2 \right),
```

which implies $s_* \le \sigma \sqrt{2 \ln L} + \|\boldsymbol{\mu}\|_{\infty}$. To obtain an upper bound for $s_{(L)}$, set $s^* \coloneqq 2 \sigma \sqrt{\ln L} + \|\boldsymbol{\mu}\|_{\infty}$. By the union bound,


```math align=center
\begin{aligned}
        \mathbb{P}(s_{(L)} > s^*) 
        &= \mathbb{P}\left( \bigcup_{i=1}^{L} \left\{s_i > s^*\right\} \right)
        \le \sum_{i=1}^{L} \mathbb{P}(s_i > s^*) \\
        &= \sum_{i=1}^{L} \Phi\left(\frac{\mu_i - s^*}{\sigma}\right)
        \le L \Phi\left(\frac{\|\boldsymbol{\mu}\|_{\infty} - s^*}{\sigma}\right) \\
        &= L \Phi\left( -2 \sqrt{\ln L} \right)
        \le L e^{-2 \ln L} 
        = \frac{1}{L} 
        \to 0 \  \mathrm{ as } L \to \infty,
    \end{aligned}
```

where the last inequality follows from $\Phi(-x) \le \frac{1}{2} e^{-x^2 / 2} < e^{-x^2 / 2}$ for $x > 0$. Combining the two bounds yields


<div id="eqn:bound s(L)"></div>

```math align=center
\mathbb{P}\left( s_{(L)} \notin [s_*, s^*] \right) 
    \le \mathbb{P}(s_{(L)} < s_*) + \mathbb{P}(s_{(L)} > s^*) 
    \to 0 \  \mathrm{ as } L \to \infty.
```

**2.** Suppose $z \in [s_*, s^*]$. Since $s_* \to \infty$ as $L \to \infty$, for sufficiently large $L$ we have $s_* > \|\boldsymbol{\mu}\|_{\infty}$. For each $i \in \left\{1, \dots, L\right\}$, define


```math align=center
\begin{aligned}
        R_i(z) 
        &\coloneqq e^{-\tau z} \mathbb{E}\left[e^{\tau s_i} \;\middle|\; s_i \le z \right] \\
        &= \frac{e^{-\tau z}}{\Phi\left( \frac{z - \mu_i}{\sigma} \right)} \int_{-\infty}^{z} e^{\tau u} \frac{1}{\sqrt{2 \pi} \sigma} e^{-\frac{(u - \mu_i)^2}{2 \sigma^2}} \mathop{}\!\mathrm{d} u \\
        &\xlongequal{v = \frac{u - \mu_i}{\sigma}} \frac{e^{-\tau (z - \mu_i)}}{\Phi\left( \frac{z - \mu_i}{\sigma} \right)} \int_{-\infty}^{\frac{z - \mu_i}{\sigma}} e^{\tau \sigma v} \frac{1}{\sqrt{2 \pi}} e^{-\frac{v^2}{2}} \mathop{}\!\mathrm{d} v \\
        &= \exp\left( \frac{1}{2} \tau^2 \sigma^2 - \tau (z - \mu_i) \right) \frac{\Phi\left( \frac{z - \mu_i}{\sigma} - \tau \sigma \right)}{\Phi\left( \frac{z - \mu_i}{\sigma} \right)}.
    \end{aligned}
```

Since $s^* = 2 \sigma \sqrt{\ln L} + \|\boldsymbol{\mu}\|_{\infty}$ and $\tau = \omega(\sqrt{\ln L})$, for sufficiently large $L$ we have $z \le s^* < -\|\boldsymbol{\mu}\|_{\infty} + \tau \sigma^2$, and hence $\frac{z - \mu_i}{\sigma} - \tau \sigma < 0$ for every $i \in \left\{1, \dots, L\right\}$. Applying Mills' ratio yields


```math align=center
\Phi\left( \frac{z - \mu_i}{\sigma} - \tau \sigma \right)
    \le \frac{1}{\sqrt{2 \pi} \left( \tau \sigma - \frac{z - \mu_i}{\sigma} \right)} \exp\left( -\frac{1}{2} \tau^2 \sigma^2 + \tau (z - \mu_i) - \frac{(z - \mu_i)^2}{2 \sigma^2} \right).
```

Therefore,


```math align=center
R_i(z) 
    \le \frac{1}{\left( \tau \sigma - \frac{z - \mu_i}{\sigma} \right) \Phi\left( \frac{z - \mu_i}{\sigma} \right)} \cdot \frac{1}{\sqrt{2 \pi}} e^{-\frac{(z - \mu_i)^2}{2 \sigma^2}}
    = \frac{\sigma^2 f_i(z)}{\left( \tau \sigma^2 - (z - \mu_i) \right) F_i(z)}
    \eqqcolon \tilde{R}_i(z).
```

Since $z - \mu_i \ge s_* - \mu_i \ge s_* - \|\boldsymbol{\mu}\|_{\infty} > 0$,


```math align=center
\begin{aligned}
        \frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} z} \ln \tilde{R}_i(z) 
        &= \frac{f_i'(z)}{f_i(z)} - \frac{f_i(z)}{F_i(z)} + \frac{1}{\tau \sigma^2 - (z - \mu_i)} \\
        &= -\frac{z - \mu_i}{\sigma^2} - \frac{f_i(z)}{F_i(z)} + \frac{1}{\tau \sigma^2 - (z - \mu_i)} \\
        &\le -\frac{s_* - \mu_i}{\sigma^2} + \frac{1}{\tau \sigma^2 - (s_* - \mu_i)}.
    \end{aligned}
```

Since $s_* \le \sigma \sqrt{2 \ln L} + \|\boldsymbol{\mu}\|_{\infty}$ and $\tau = \omega(\sqrt{\ln L})$, for sufficiently large $L$ we have $\tau \sigma^2 \gg s^* - \mu_i > s_* - \mu_i$. Thus, $\tilde{R}_i'(z) < 0$, so the maximum of $\tilde{R}_i(z)$ is attained at $s_*$. Moreover, since $F_i(s_*) \ge \frac{1}{2}$ and $\tau \sigma^2 \ge 2 (s^* - \mu_i) \ge 2 (z - \mu_i)$, we have


```math align=center
R_i(z) \le \tilde{R}_i(z) \le \tilde{R}_i(s_*) \le \frac{4}{\tau} f_i(s_*).
```

For any $x \ge 1$, Mills' ratio gives $1 - \Phi(x) \ge \frac{x}{x^2 + 1} \phi(x) \ge \frac{1}{2 x} \phi(x)$, where $\phi$ is the standard normal density. Since $s_* \ge \|\boldsymbol{\mu}\|_{\infty}$, for sufficiently large $L$,


```math align=center
f_i(s_*) 
    = \frac{1}{\sigma} \phi\left( \frac{s_* - \mu_i}{\sigma} \right)
    \le \frac{2}{\sigma} \left( \frac{s_* - \mu_i}{\sigma} \right) \left( 1 - \Phi\left( \frac{s_* - \mu_i}{\sigma} \right) \right)
    \le \frac{4 s_*}{\sigma^2} (1 - F_i(s_*)).
```

Combining these inequalities yields


<div id="eqn:bound Ri"></div>

```math align=center
\sup_{z \in [s_*, s^*]} R_i(z) 
    \le \frac{16 s_*}{\tau \sigma^2} (1 - F_i(s_*)), 
    \  \forall i \in \left\{1, \dots, L\right\}.
```

Summing Eq. (17) over $i \in \left\{1, \dots, L\right\}$, using the definition of $s_*$, and applying $\ln x \le x - 1$, we obtain, for any $z \in [s_*, s^*]$,


```math align=center
\sum_{i=1}^{L} R_i(z)
    \le \frac{16 s_*}{\tau \sigma^2} \sum_{i=1}^{L} (1 - F_i(s_*))
    \le - \frac{16 s_*}{\tau \sigma^2} \sum_{i=1}^{L} \ln F_i(s_*)
    = \frac{16 s_* \gamma_L}{\tau \sigma^2}.
```

Using $s_* \le \sigma \sqrt{2 \ln L} + \|\boldsymbol{\mu}\|_{\infty}$ and $\gamma_L \le \left( \tau / \sqrt{\ln L} \right)^{1/2}$, we have


```math align=center
\sup_{z \in [s_*, s^*]} \sum_{i=1}^{L} R_i(z)
    \le \frac{16}{\tau \sigma^2} \left( \sigma \sqrt{2 \ln L} + \|\boldsymbol{\mu}\|_{\infty} \right) \left( \frac{\tau}{\sqrt{\ln L}} \right)^{1/2}
    \lesssim \left( \frac{\sqrt{\ln L}}{\tau} \right)^{1/2}
    \to 0 \  \mathrm{ as } L \to \infty.
```

**3. Bounding $\zeta_L$.** For any $\delta > 0$, write


<div id="eqn:WL decompose"></div>

```math align=center
\mathbb{P}\left( \zeta_L > \delta) \le \mathbb{P}(\zeta_L > \delta, s_{(L)} \in [s_*, s^*]) + \mathbb{P}(s_{(L)} \notin [s_*, s^*] \right).
```

By the law of total probability,


```math align=center
\begin{aligned}
        \mathbb{P}(\zeta_L > \delta, s_{(L)} \in [s_*, s^*])
        &= \int_{s_*}^{s^*} \mathbb{P}\left( \zeta_L > \delta \;\middle|\; s_{(L)} = z \right) f_{(L)}(z) \mathop{}\!\mathrm{d} z,
    \end{aligned}
```

where $f_{(L)}(z)$ is the PDF of $s_{(L)}$. By Markov's inequality,


```math align=center
\mathbb{P}\left( \zeta_L > \delta \;\middle|\; s_{(L)} = z \right) \le \frac{1}{\delta} \mathbb{E}\left[ \zeta_L \;\middle|\; s_{(L)} = z \right].
```

Since $s_1, \dots, s_L$ are independent continuous random variables, the events $\left\{s_j = z;   s_i < z, \forall i \neq j\right\}$ partition the event $\left\{s_{(L)} = z\right\}$. Thus,


```math align=center
\begin{aligned}
        \mathbb{E}\left[ \zeta_L \;\middle|\; s_{(L)} = z \right]
        &= \sum_{j=1}^{L} \mathbb{E}\left[ \zeta_L \cdot \mathbb{1}\left\{s_j = z;   s_i < z, \forall i \neq j\right\} \;\middle|\; s_{(L)} = z \right] \\
        &= \sum_{j=1}^{L} \pi_j(z) \mathbb{E}\left[ \zeta_L \;\middle|\; s_j = z;   s_i < z, \forall i \neq j \right] \\
        &= \sum_{j=1}^{L} \pi_j(z) \sum_{i \neq j} \mathbb{E}\left[ e^{\tau (s_i - z)} \;\middle|\; s_j = z;   s_i < z, \forall i \neq j \right] \\
        &= \sum_{j=1}^{L} \pi_j(z) \sum_{i \neq j} \mathbb{E}\left[ e^{\tau (s_i - z)} \;\middle|\; s_i < z \right] \\
        &= \sum_{j=1}^{L} \pi_j(z) \sum_{i \neq j} R_i(z)
        \le \sum_{j=1}^{L} \pi_j(z) \sum_{i = 1}^{L} R_i(z)
        = \sum_{i = 1}^{L} R_i(z),
    \end{aligned}
```

where $\pi_j(z) \coloneqq \mathbb{P}\left( s_j = z;   s_i < z, \forall i \neq j \;\middle|\; s_{(L)} = z \right)$ satisfies $\sum_{j=1}^{L} \pi_j(z) = 1$. Therefore,


```math align=center
\begin{aligned}
        \mathbb{P}(\zeta_L > \delta, s_{(L)} \in [s_*, s^*])
        &\le \frac{1}{\delta} \sup_{z \in [s_*, s^*]} \sum_{i = 1}^{L} R_i(z) \left( \int_{s_*}^{s^*} f_{(L)}(z) \mathop{}\!\mathrm{d} z \right) \\
        &\le \frac{1}{\delta} \sup_{z \in [s_*, s^*]} \sum_{i = 1}^{L} R_i(z)
        \to 0 \  \mathrm{ as } L \to \infty.
    \end{aligned}
```

Combining this with Eq. (16) and substituting into Eq. (18) yields $\zeta_L \overset{\mathbb{P}}{\longrightarrow} 0$, which completes the proof.

</div>
Using Lemma 2, we prove the law of large numbers in Proposition 2 for the three cases.

<div class="theorem">

**Restatement of Proposition 2.** The statement is given above.

</div>


<div class="proof">

**Proof.**

For any $\beta \neq 1$,


```math align=center
\frac{\mathcal{E}_{\beta}(\boldsymbol{\alpha})}{\hat{\mathcal{E}}_{\beta}(L)}
    = \left. \left( \frac{Z(\boldsymbol{s}; \lambda)}{\mathbb{E} Z(\boldsymbol{s}; \lambda)} \right)^{\frac{\beta}{\beta - 1}} \right/ \left( \frac{Z(\boldsymbol{s}; \beta \lambda)}{\mathbb{E} Z(\boldsymbol{s}; \beta \lambda)} \right)^{\frac{1}{\beta - 1}}.
```

**(i)** If $\Lambda < \sqrt{2} \min\left\{1/\beta, 1\right\}$, then $\Lambda < \sqrt{2}$ and $\beta \Lambda < \sqrt{2}$. By Lemma 2(i),


```math align=center
\frac{Z(\boldsymbol{s}; \lambda)}{\mathbb{E} Z(\boldsymbol{s}; \lambda)} \overset{\mathbb{P}}{\longrightarrow} 1, 
    \ 
    \frac{Z(\boldsymbol{s}; \beta \lambda)}{\mathbb{E} Z(\boldsymbol{s}; \beta \lambda)} \overset{\mathbb{P}}{\longrightarrow} 1, 
    \  \mathrm{ as } L \to \infty.
```

Since $g(x, y) = x^{\frac{\beta}{\beta - 1}} / y^{\frac{1}{\beta - 1}}$ is continuous at $(1, 1)$, the continuous mapping theorem gives


```math align=center
\frac{\mathcal{E}_{\beta}(\boldsymbol{\alpha})}{\hat{\mathcal{E}}_{\beta}(L)} 
    = g\left( \frac{Z(\boldsymbol{s}; \lambda)}{\mathbb{E} Z(\boldsymbol{s}; \lambda)}, \frac{Z(\boldsymbol{s}; \beta \lambda)}{\mathbb{E} Z(\boldsymbol{s}; \beta \lambda)} \right) 
    \overset{\mathbb{P}}{\longrightarrow} g(1, 1) = 1
    \  \mathrm{ as } L \to \infty.
```

**(ii)** If $\Lambda = \sqrt{2} \min\left\{1/\beta, 1\right\}$, then by Lemma 2(i) and (ii), there are two cases. If $\Lambda = \sqrt{2}$ and $\beta \Lambda < \sqrt{2}$, so that $0 < \beta < 1$, then


```math align=center
\frac{\mathcal{E}_{\beta}(\boldsymbol{\alpha})}{\hat{\mathcal{E}}_{\beta}(L)} 
    \overset{\mathbb{P}}{\longrightarrow} g\left( \frac{1}{2}, 1 \right) = 2^{\frac{\beta}{1 - \beta}}
    \  \mathrm{ as } L \to \infty.
```

If $\Lambda < \sqrt{2}$ and $\beta \Lambda = \sqrt{2}$, so that $\beta > 1$, then


```math align=center
\frac{\mathcal{E}_{\beta}(\boldsymbol{\alpha})}{\hat{\mathcal{E}}_{\beta}(L)} 
    \overset{\mathbb{P}}{\longrightarrow} g\left( 1, \frac{1}{2} \right) = 2^{\frac{1}{\beta - 1}}
    \  \mathrm{ as } L \to \infty.
```

Combining these two cases, for $\beta \neq 1$,


```math align=center
\frac{\mathcal{E}_{\beta}(\boldsymbol{\alpha})}{\hat{\mathcal{E}}_{\beta}(L)} \overset{\mathbb{P}}{\longrightarrow} 2^{\frac{1}{\max\left\{\beta, \beta^{-1}\right\} - 1}} \  \mathrm{ as } L \to \infty.
```

**(iii)** If $\Lambda > \sqrt{2} / \beta$, then, by the definition of $\Lambda$, there exist $\varepsilon > 0$ and a subsequence $\left\{L_n\right\} \subset \mathbb{N}$ such that


```math align=center
\frac{\lambda^2(L_n) \sigma^2}{\ln L_n} > \frac{2}{\beta} + \varepsilon, \  \forall n \in \mathbb{N}.
```

Since $\mathcal{E}_{\beta}(\mathring{\boldsymbol{\alpha}}) \le L$, we obtain


```math align=center
\hat{\mathcal{E}}_{\beta}(L_n)
    \le e^{-\frac{\beta}{2} \lambda^2(L_n) \sigma^2} L_n
    < e^{-\frac{\beta}{2} \left( \frac{2}{\beta} + \varepsilon \right) \ln L_n} L_n
    = L_n^{-\frac{\beta}{2} \varepsilon} \to 0
    \  \mathrm{ as } n \to \infty.
```

Therefore, $\liminf_{L \to \infty} \hat{\mathcal{E}}_{\beta}(L) = 0$. Moreover, if $\Lambda = \infty$, then by Lemma 2(iv),


```math align=center
\begin{aligned}
        \mathcal{E}_{\beta}(\boldsymbol{\alpha}) 
        &= \frac{\left( \sum_{i=1}^{L} e^{\lambda s_i} \right)^{\frac{\beta}{\beta - 1}}}{\left( \sum_{i=1}^{L} e^{\beta \lambda s_i} \right)^{\frac{1}{\beta - 1}}}
        = \left. \left( \frac{\sum_{i=1}^{L} e^{\lambda s_i}}{\max_{1 \le i \le L} e^{\lambda s_i}} \right)^{\frac{\beta}{\beta - 1}} \right/ \left( \frac{\sum_{i=1}^{L} e^{\beta \lambda s_i}}{\max_{1 \le i \le L} e^{\beta \lambda s_i}} \right)^{\frac{1}{\beta - 1}} \\
        &= g\left( \frac{\sum_{i=1}^{L} e^{\lambda s_i}}{\max_{1 \le i \le L} e^{\lambda s_i}}, \frac{\sum_{i=1}^{L} e^{\beta \lambda s_i}}{\max_{1 \le i \le L} e^{\beta \lambda s_i}} \right)
        \overset{\mathbb{P}}{\longrightarrow} g(1, 1) = 1
        \  \mathrm{ as } L \to \infty.
    \end{aligned}
```

</div>

<div id="sec:proof of thm:scale nope"></div>
## Proof of Theorem 3
<div class="theorem">

**Restatement of Theorem 3.** The statement is given above.

</div>


<div class="proof">

**Proof.**

By Proposition 1, as $d \to \infty$, the limiting distribution of $\boldsymbol{\alpha}$ is $\boldsymbol{g} \sim \mathcal{N}(\boldsymbol{\mu}, \sigma^2 \boldsymbol{I}_d)$. For NoPE, Eq. (2) gives $\boldsymbol{\mu} = \rho \boldsymbol{1}_d$, so $Z(\boldsymbol{\mu}; \lambda) = L e^{\rho}$ and $Z(\boldsymbol{\mu}; \beta \lambda) = L e^{\rho \beta}$. Therefore, by Eq. (4), Eq. (5),


```math align=center
\mathcal{E}_{\beta}(\mathring{\boldsymbol{g}}) = \frac{Z(\boldsymbol{\mu}; \lambda)^{\frac{\beta}{\beta - 1}}}{Z(\boldsymbol{\mu}; \beta \lambda)^{\frac{1}{\beta - 1}}} = L,
    \ 
    \hat{\mathcal{E}}_{\beta}(L) = e^{-\frac{\beta}{2} \lambda(L)^2 \sigma^2} L.
```

By the continuous mapping theorem,


```math align=center
\frac{\mathcal{E}_{\beta}(\boldsymbol{\alpha})}{\mathcal{E}_{\beta}^*(L)} \overset{\mathcal{D}}{\longrightarrow} \frac{\mathcal{E}_{\beta}(\boldsymbol{g})}{\mathcal{E}_{\beta}^*(L)}
    \  \mathrm{ as } d \to \infty.
```

By the Portmanteau theorem, for any $\varepsilon > 0$,


```math align=center
\limsup_{d \to \infty} \mathbb{P}\left( \left| \frac{\mathcal{E}_{\beta}(\boldsymbol{\alpha})}{\mathcal{E}_{\beta}^*(L)} - 1 \right| \ge \varepsilon \right)
    \le \mathbb{P}\left( \left| \frac{\mathcal{E}_{\beta}(\boldsymbol{g})}{\mathcal{E}_{\beta}^*(L)} - 1 \right| \ge \varepsilon \right).
```

Then, by Proposition 2,


```math align=center
\limsup_{L \to \infty} \limsup_{d \to \infty} \mathbb{P}\left( \left| \frac{\mathcal{E}_{\beta}(\boldsymbol{\alpha})}{\mathcal{E}_{\beta}^*(L)} - 1 \right| \ge \varepsilon \right)
    \le \limsup_{L \to \infty} \mathbb{P}\left( \left| \frac{\mathcal{E}_{\beta}(\boldsymbol{g})}{\mathcal{E}_{\beta}^*(L)} - 1 \right| \ge \varepsilon \right)
    = 0.
```

Hence,


```math align=center
\frac{\mathcal{E}_{\beta}(\boldsymbol{\alpha})}{\mathcal{E}_{\beta}^*(L)}
    \overset{\mathbb{P}}{\longrightarrow} 1
    \  \mathrm{ as } d \to \infty \mathrm{ then } L \to \infty,
```

where the exact form of $\mathcal{E}_{\beta}^*(L)$ depends on the range of $\Lambda$.


**(i)** If $\Lambda < \sqrt{2} \min\left\{1/\beta, 1\right\}$, then Proposition 2(i) gives


```math align=center
\mathcal{E}_{\beta}^*(L) = \hat{\mathcal{E}}_{\beta}(L) = e^{-\frac{\beta}{2} \lambda(L)^2 \sigma^2} L.
```

Thus,


```math align=center
\lambda(L) = \sqrt{\frac{2}{\beta \sigma^2} \ln \left( \frac{L}{\mathcal{E}_{\beta}^*(L)} \right)}.
```

Hence, the condition is equivalent to


```math align=center
\Lambda 
    = \limsup_{L \to \infty} \sqrt{\frac{2}{\beta} \cdot \frac{\ln (L / \mathcal{E}_{\beta}^*(L))}{\ln L}}
    = \sqrt{\frac{2}{\beta} \left( 1 - \liminf_{L \to \infty} \frac{\ln \mathcal{E}_{\beta}^*(L)}{\ln L} \right)}
    < \sqrt{2} \min\left\{1/\beta, 1\right\},
```

which implies


```math align=center
\liminf_{L \to \infty} \frac{\ln \mathcal{E}_{\beta}^*(L)}{\ln L} > 1 - \min\left\{\beta, \beta^{-1}\right\}.
```

**(ii)** If $\Lambda = \sqrt{2} \min\left\{1/\beta, 1\right\}$, then Proposition 2(ii) gives


```math align=center
\mathcal{E}_{\beta}^*(L) = 2^{\frac{1}{\max\left\{\beta, \beta^{-1}\right\} - 1}} \hat{\mathcal{E}}_{\beta}(L) = e^{-\frac{\beta}{2} \lambda(L)^2 \sigma^2} 2^{\frac{1}{\max\left\{\beta, \beta^{-1}\right\} - 1}} L.
```

Thus,


```math align=center
\lambda(L) = \sqrt{\frac{2}{\beta \sigma^2} \ln \left( \frac{2^{\frac{1}{\max\left\{\beta, \beta^{-1}\right\} - 1}} L}{\mathcal{E}_{\beta}^*(L)} \right)}.
```

Hence, the condition is equivalent to


```math align=center
\Lambda 
    = \sqrt{\frac{2}{\beta} \left( 1 - \lim_{L \to \infty} \frac{\ln \mathcal{E}_{\beta}^*(L)}{\ln L} \right)}
    = \sqrt{2} \min\left\{1/\beta, 1\right\},
```

which gives


```math align=center
\lim_{L \to \infty} \frac{\ln \mathcal{E}_{\beta}^*(L)}{\ln L} = 1 - \min\left\{\beta, \beta^{-1}\right\}.
```

**(iii)** If $\Lambda = \infty$, then Proposition 2(iii) gives $\mathcal{E}_{\beta}^*(L) = 1$ and


```math align=center
\Lambda 
    = \sigma \lim_{L \to \infty} \frac{\lambda(L)}{\sqrt{\ln L}}
    = \infty.
```

Therefore, the condition is automatically satisfied.

</div>

<div id="sec:proof of thm:scale rope"></div>
## Proof of Theorem 4
<div class="theorem">

**Restatement of Theorem 4.** The statement is given above.

</div>


<div class="proof">

**Proof.**

By Proposition 1, as $d \to \infty$, the limiting distribution of $\boldsymbol{\alpha}$ is $\boldsymbol{g} \sim \mathcal{N}(\boldsymbol{\mu}, \sigma^2 \boldsymbol{I}_d)$. For RoPE, Eq. (3) gives $\mu_i = \rho \int_{0}^{1} \cos\left( (i-L) b^{-x} \right) \mathop{}\!\mathrm{d} x$ for all $i \in \left\{1, \dots, L\right\}$. By the same argument as in the proof of Theorem 3, we have


```math align=center
\frac{\mathcal{E}_{\beta}(\boldsymbol{\alpha})}{\mathcal{E}_{\beta}^*(L)}
    = \left( \left. \frac{\mathcal{E}_{\beta}(\boldsymbol{\alpha})}{\mathcal{E}_{\beta}^*(L)} \right/ \frac{\mathcal{E}_{\beta}(\boldsymbol{g})}{\mathcal{E}_{\beta}^*(L)} \right) \cdot \frac{\mathcal{E}_{\beta}(\boldsymbol{g})}{\mathcal{E}_{\beta}^*(L)}
    \overset{\mathbb{P}}{\longrightarrow} 1
    \  \mathrm{ as } d \to \infty \mathrm{ then } L \to \infty.
```

When $\Lambda = \infty$ (case (iii)), the proof is identical to that of Theorem 3(iii). It remains to consider $\Lambda \le \sqrt{2} \min\left\{1/\beta, 1\right\}$ (cases (i) and (ii)). As in the proofs of Theorem 3(i) and (ii), by Proposition 2, it suffices to show that for any $\beta \neq 1$,


<div id="eqn:scale rope ess"></div>

```math align=center
\mathcal{E}_{\beta}(\mathring{\boldsymbol{g}}) 
    = \frac{Z(\boldsymbol{\mu}; \lambda)^{\frac{\beta}{\beta - 1}}}{Z(\boldsymbol{\mu}; \beta \lambda)^{\frac{1}{\beta - 1}}} 
    = L + O\left( e^{\rho \max\left\{1, \beta\right\} \lambda} \right)
    \  \mathrm{ as } L \to \infty.
```

Note that $\mathcal{E}_2(\mathring{\boldsymbol{g}}) = L + O(e^{2 \lambda}) = \omega(L / \sqrt{\ln L})$, which satisfies the condition required by Proposition 2(ii). Let $I_k \coloneqq \int_{0}^{1} \cos(k b^{-x}) \mathop{}\!\mathrm{d} x$. Then $\mu_i = \rho I_{i-L} = \rho I_{L-i}$. Moreover,


```math align=center
I_k 
    = \int_{0}^{1} \cos(k b^{-x}) \mathop{}\!\mathrm{d} x
    \xlongequal{u = b^{-x}} \int_{0}^{1/b} \cos(k u) \frac{- \mathop{}\!\mathrm{d} u}{u \ln b}
    = \frac{1}{\ln b} \int_{1/b}^{1} \frac{\cos (k u)}{u} \mathop{}\!\mathrm{d} u.
```

Define


```math align=center
S_L(\tau) \coloneqq Z(\boldsymbol{\mu}; \tau / \rho) = \sum_{i=1}^{L} e^{(\tau / \rho) \mu_i} = \sum_{k=0}^{L-1} e^{\tau I_k},
```

where $\tau \in \left\{\rho \lambda, \rho \beta \lambda\right\}$ corresponds to the inverse temperature. We prove Eq. (19) by showing that $S_L(\tau) = L + e^{\tau} + o(e^{\tau})$ as $L \to \infty$.


**1. Bounding $I_k$.** We first show that, for any $k \ge b$, $|I_k| \le (b + 1) / (k \ln b)$. We estimate $I_k$ using the cosine integral function $\operatorname{Ci}(z)$, defined by


```math align=center
\operatorname{Ci}(z) = - \int_{z}^{\infty} \frac{\cos t}{t} \mathop{}\!\mathrm{d} t.
```

We first prove that, for any $z \in [1, \infty)$, $|\operatorname{Ci}(z)| \le 1 / z$. For any $t > 0$, we have $t^{-1} = \int_{0}^{\infty} e^{-ut} \mathop{}\!\mathrm{d} u$. Applying Fubini's theorem on truncated intervals and then taking the limit gives


```math align=center
\begin{aligned}
        \operatorname{Ci}(z)
        &= - \lim_{K \to \infty} \int_{z}^{K} \frac{\cos t}{t} \mathop{}\!\mathrm{d} t \\
        &= - \lim_{K \to \infty} \int_{z}^{K} \left( \int_{0}^{\infty} e^{-ut} \cos t \mathop{}\!\mathrm{d} u \right) \mathop{}\!\mathrm{d} t \\
        &\overset{\mathrm{(a)}}{=} - \lim_{K \to \infty} \int_{0}^{\infty} \left( \int_{z}^{K} e^{-ut} \cos t \mathop{}\!\mathrm{d} t \right) \mathop{}\!\mathrm{d} u \\
        &= - \lim_{K \to \infty} \int_{0}^{\infty} \left( \left. \frac{e^{-u t} (\sin t - u \cos t)}{1 + u^2} \right|_{t=z}^{K} \right) \mathop{}\!\mathrm{d} u \\
        &= \int_{0}^{\infty} \frac{e^{-uz} (\sin z - u \cos z)}{1 + u^2} \mathop{}\!\mathrm{d} u - \lim_{K \to \infty} \int_{0}^{\infty} \frac{e^{-u K} (\sin K - u \cos K)}{1 + u^2} \mathop{}\!\mathrm{d} u \\
        &\overset{\mathrm{(b)}}{=} \int_{0}^{\infty} e^{-uz} \frac{\sin z - u \cos z}{1 + u^2} \mathop{}\!\mathrm{d} u.
    \end{aligned}
```

Here, step (a) follows from Fubini's theorem on $[z, K] \times [0, \infty)$, and step (b) follows from the dominated convergence theorem since the integrand in the second term is bounded by the integrable function $e^{-uz}$. By the Cauchy–Schwarz inequality, $|\sin z - u \cos z| \le \sqrt{1 + u^2}$, and hence


```math align=center
|\operatorname{Ci}(z)| \le \int_{0}^{\infty} e^{-uz} \frac{\sqrt{1+u^2}}{1+u^2} \mathop{}\!\mathrm{d} u
    = \int_{0}^{\infty}\frac{e^{-uz}}{\sqrt{1+u^2}} \mathop{}\!\mathrm{d} u
    \le \int_{0}^{\infty} e^{-uz} \mathop{}\!\mathrm{d} u
    = \frac{1}{z}.
```

Since $\operatorname{Ci}'(z) = \cos(z) / z$, the chain rule gives $\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} z} \operatorname{Ci}(k z) = \cos(kz) / z$. Therefore,


```math align=center
I_k 
    = \frac{1}{\ln b} \int_{1/b}^{1} \frac{\cos (k u)}{u} \mathop{}\!\mathrm{d} u
    = \left. \frac{\operatorname{Ci}(ku)}{\ln b} \right|_{u = 1/b}^{1}
    = \frac{1}{\ln b} \left[ \operatorname{Ci}(k) - \operatorname{Ci}\left(\frac{k}{b}\right) \right].
```

Since $k \ge b > 1$, applying $|\operatorname{Ci}(z)| \le 1 / z$ yields


```math align=center
|I_k|
    \le \frac{1}{\ln b} \left( |\operatorname{Ci}(k)| + \left| \operatorname{Ci}\left( \frac{k}{b} \right) \right| \right) 
    \le \frac{1}{\ln b} \left( \frac{1}{k} + \frac{b}{k} \right) 
    = \frac{b + 1}{k \ln b}.
```

**2. Bounding $S_L(\tau)$.** By assumption, $\lambda(L) \to \infty$ and $\lambda(L) \ln L = o(e^{\rho \min\left\{1, \beta\right\} \lambda(L)})$ as $L \to \infty$. Hence $\tau \to \infty$ and $\tau \ln L = o(e^{\tau})$ for both $\tau \in \left\{\rho \lambda, \rho \beta \lambda\right\}$. Decompose the partial sum as


```math align=center
S_L(\tau) = \sum_{k=0}^{L-1} 1 + \sum_{k=0}^{L-1} \left( e^{\tau I_k} - 1 \right) = L + (e^{\tau} - 1) + \sum_{k=1}^{L-1} \left( e^{\tau I_k} - 1 \right).
```

We bound the remaining sum by splitting the index set at $k = \left\lfloor \tau \right\rfloor$. Recall that, for any $k \in \mathbb{N}$,


```math align=center
I_k = \frac{1}{\ln b} \int_{1/b}^{1} \frac{\cos (k u)}{u} \mathop{}\!\mathrm{d} u.
```

Thus $I_0 = 1$ and $|I_k| < 1$ for all $k \ge 1$. Since $I_k \to 0$ as $k \to \infty$, there exists


```math align=center
c \coloneqq \max\left\{0,   \sup_{k \ge 1} I_k\right\} \in [0, 1)
```

such that $I_k \le c$ for all $k \ge 1$. For $1 \le k \le \left\lfloor \tau \right\rfloor$, we have


```math align=center
\sum_{k=1}^{\left\lfloor \tau \right\rfloor} |e^{\tau I_k} - 1|
    \le \sum_{k=1}^{\left\lfloor \tau \right\rfloor} \max\left\{e^{\tau I_k}, 1\right\}
    \le \tau e^{\tau c} 
    = o(e^{\tau}) \  \mathrm{ as } L \to \infty.
```

For $\left\lfloor \tau \right\rfloor + 1 \le k \le L - 1$, the bound on $I_k$ gives


```math align=center
|\tau I_k| \le \frac{\tau(b+1)}{k \ln b} \le \frac{b+1}{\ln b}.
```

Using $|e^x - 1| \le |x|e^{|x|}$ for all $x \in \mathbb{R}$, we obtain


```math align=center
|e^{\tau I_k} - 1| 
    = O(|\tau I_k|) 
    = O\left( \frac{\tau}{k} \right) \  \mathrm{ as } L \to \infty.
```

Therefore,


```math align=center
\sum_{k = \left\lfloor \tau \right\rfloor + 1}^{L - 1} |e^{\tau I_k} - 1| 
    \le \sum_{k=\left\lfloor \tau \right\rfloor + 1}^{L-1} O\left( \frac{\tau}{k} \right) 
    = O \left( \tau \ln \left( \frac{L}{\tau} \right) \right) 
    \le O(\tau \ln L)
    = o(e^{\tau}) \  \mathrm{ as } L \to \infty.
```

Combining these estimates yields $S_L(\tau) = L + e^{\tau} + o(e^{\tau})$ as $L \to \infty$.


**3.** It remains to verify Eq. (19). Using $Z(\boldsymbol{\mu}; \lambda) = S_L(\rho \lambda) = L + e^{\rho \lambda} + o(e^{\rho \lambda})$ and $Z(\boldsymbol{\mu}; \beta \lambda) = S_L(\rho \beta \lambda) = L + e^{\rho \beta \lambda} + o(e^{\rho \beta \lambda})$ as $L \to \infty$, we obtain


```math align=center
\begin{aligned}
        \mathcal{E}_{\beta}(\mathring{\boldsymbol{g}})
        &= \frac{Z(\boldsymbol{\mu}; \lambda)^{\frac{\beta}{\beta - 1}}}{Z(\boldsymbol{\mu}; \beta \lambda)^{\frac{1}{\beta - 1}}} \\
        &= \left( \frac{(L + e^{\rho \lambda} + o(e^{\rho \lambda}))^{\beta}}{L + e^{\rho \beta \lambda} + o(e^{\rho \beta \lambda})} \right)^{\frac{1}{\beta - 1}} \\
        &= L \left( \frac{(1 + e^{\rho \lambda}/L + o(e^{\rho \lambda}/L))^{\beta}}{1 + e^{\rho \beta \lambda}/L + o(e^{\rho \beta \lambda}/L)} \right)^{\frac{1}{\beta - 1}} \\
        &= L \left( \left( 1 + \beta \frac{e^{\rho \lambda}}{L} + o \left( \frac{e^{\rho \lambda}}{L} \right) \right) \left( 1 - \frac{e^{\rho \beta \lambda}}{L} + o \left( \frac{e^{\rho \beta \lambda}}{L} \right) \right) \right)^{\frac{1}{\beta - 1}} \\
        &= L \left( 1 + \frac{\beta e^{\rho \lambda} - e^{\rho \beta \lambda}}{L} + o \left( \frac{e^{\rho \max\left\{1, \beta\right\} \lambda}}{L} \right) \right)^{\frac{1}{\beta - 1}} \\
        &= L \left( 1 + \frac{\beta e^{\rho \lambda} - e^{\rho \beta \lambda}}{(\beta - 1) L} + o \left( \frac{e^{\rho \max\left\{1, \beta\right\} \lambda}}{L} \right) \right) \\
        &= L + O\left( e^{\rho \max\left\{1, \beta\right\} \lambda} \right) \  \mathrm{ as } L \to \infty,
    \end{aligned}
```

which completes the proof.

</div>



[^vershynin2026high]: Vershynin, Roman. (2026). *High-Dimensional Probability: An Introduction with Applications in Data Science*. Cambridge University Press.

[^von1965inequalities]: von Bahr, Bengt and Esseen, Carl-Gustav. (1965). *Inequalities for the rth absolute moment of a sum of random variables, $1 \le r \le 2$*. The Annals of Mathematical Statistics. 36, (1), 299--303.

[^molchanov2019limit]: Molchanov, Stanislav and Panov, Vladimir. (2019). *Limit theorems for the alloy-type random energy model*. Stochastics. 91, (5), 754--772.

[^ben2005limit]: Ben Arous, Gerard and Bogachev, Leonid V. and Molchanov, Stanislav A.. (2005). *Limit theorems for sums of random exponentials*. Probability Theory and Related Fields. 132, (4), 579--612.
