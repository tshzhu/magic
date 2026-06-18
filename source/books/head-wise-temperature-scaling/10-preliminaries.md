# Preliminaries

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
  padding: 0.5rem 1rem;
  margin: 1rem 0;
  background: #fafafa;
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

In this report, we introduce the metric *effective sequence length* (Section 2) for attention heads and explain: (i) why temperature scaling is necessary in attention mechanisms (Section 2.1, Section 2.2); and (ii) how to select temperature schedules for different attention heads (Section 3).


<div id="sec:notation"></div>
## Notation

**General Notation.** Let $\mathbb{R}$, $\mathbb{C}$, $\mathbb{Z}$, and $\mathbb{N} = \left\{0, 1, \dots\right\}$ denote the sets of real numbers, complex numbers, integers, and non-negative integers, respectively. We denote by $\mathbb{R}^d$ the $d$-dimensional Euclidean space, and use $\| \cdot \|_p$ for the $p$-norm, $1 \le p \le \infty$ (with $p = 2$ when unspecified). Let $\boldsymbol{0}_d, \boldsymbol{1}_d \in \mathbb{R}^d$ denote the vectors of all zeros and all ones, respectively. Let $\boldsymbol{I}_d$ denote the $d \times d$ identity matrix. We use $O(\cdot)$, $o(\cdot)$, $\Omega(\cdot)$, $\omega(\cdot)$, and $\Theta(\cdot)$ for standard asymptotic notation. Almost sure convergence, convergence in probability, and convergence in distribution are denoted by $\overset{\mathrm{a.s.}}{\longrightarrow}$, $\overset{\mathbb{P}}{\longrightarrow}$, and $\overset{\mathcal{D}}{\longrightarrow}$, respectively. We denote the multivariate normal distribution with mean vector $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$ by $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$. The indicator function is denoted by $\mathbb{1}{\left\{\cdot\right\}}$.

Now we briefly review the essential details of multi-head attention and rotary position embeddings, which form the basis of our analysis.


**Multi-Head Attention.** Let $\boldsymbol{x}_i \in \mathbb{R}^D$ be the $D$-dimensional embedding of the $i$-th token, where $i \in \left\{1, \dots, L\right\}$. Consider the input embedding sequence $\boldsymbol{X} = (\boldsymbol{x}_1, \dots, \boldsymbol{x}_L)^{\mathsf{T}} \in \mathbb{R}^{L \times D}$. Suppose the model uses $H$ attention heads. For each head $h \in \left\{1, \dots, H\right\}$, let $\boldsymbol{W}_Q^{(h)}, \boldsymbol{W}_K^{(h)}, \boldsymbol{W}_V^{(h)} \in \mathbb{R}^{D \times d}$ be the projection matrices, where $d = D / H \in \mathbb{N}$ is the hidden dimension of each head. The query, key, and value matrices for the $h$-th head are computed as


```math align=center
\boldsymbol{Q}^{(h)} = \boldsymbol{X} \boldsymbol{W}_Q^{(h)}, 
    \  \boldsymbol{K}^{(h)} = \boldsymbol{X} \boldsymbol{W}_K^{(h)}, 
    \  \boldsymbol{V}^{(h)} = \boldsymbol{X} \boldsymbol{W}_V^{(h)}.
```
The attention output of head $h$, denoted by $\boldsymbol{O}^{(h)} \in \mathbb{R}^{L \times d}$, is defined as


```math align=center
\boldsymbol{O}^{(h)} = \operatorname{Softmax} \left(\frac{1}{\sqrt{d}} \boldsymbol{Q}^{(h)} (\boldsymbol{K}^{(h)})^{\mathsf{T}} + \boldsymbol{M}\right) \boldsymbol{V}^{(h)},
```
where $\boldsymbol{M} \in \mathbb{R}^{L \times L}$ is the causal mask matrix. Let $\boldsymbol{q}_i^{(h)}, \boldsymbol{k}_i^{(h)}, \boldsymbol{v}_i^{(h)} \in \mathbb{R}^{d}$ denote the $i$-th rows of $\boldsymbol{Q}^{(h)}, \boldsymbol{K}^{(h)}$, and $\boldsymbol{V}^{(h)}$, respectively. For token $t \in \left\{1, \dots, L\right\}$, the output vector $\boldsymbol{o}^{(h)}_t \in \mathbb{R}^{d}$ is


```math align=center
\boldsymbol{o}^{(h)}_t = \sum_{i=1}^{t} \frac{\exp\left( (\boldsymbol{q}_t^{(h)})^{\mathsf{T}} \boldsymbol{k}^{(h)}_i / \sqrt{d} \right)}{\sum_{j=1}^{t} \exp\left( (\boldsymbol{q}^{(h)}_t)^{\mathsf{T}} \boldsymbol{k}^{(h)}_j / \sqrt{d} \right)} \boldsymbol{v}^{(h)}_i.
```
We denote the attention logit between query $\boldsymbol{q}^{(h)}_i$ and key $\boldsymbol{k}^{(h)}_j$ by $s^{(h)}_{i, j} = \frac{1}{\sqrt{d}} (\boldsymbol{q}^{(h)}_i)^{\mathsf{T}} \boldsymbol{k}^{(h)}_j$, and define $\alpha^{(h)}_{t, i} = \operatorname{Softmax}_i ( s^{(h)}_{t, 1}, \dots, s^{(h)}_{t, t} )$ as the attention weight of token $t$ attending to token $i$. Thus, $\boldsymbol{o}^{(h)}_t = \sum_{i=1}^{t} \alpha^{(h)}_{t, i} \boldsymbol{v}^{(h)}_i$.

Finally, the head outputs are concatenated along the feature dimension and linearly projected to produce the multi-head output $\boldsymbol{O} \in \mathbb{R}^{L \times D}$, which is


```math align=center
\boldsymbol{O} = \left[ \boldsymbol{O}^{(1)}, \boldsymbol{O}^{(2)}, \dots, \boldsymbol{O}^{(H)} \right] \boldsymbol{W}_O,
```
where $\boldsymbol{W}_O \in \mathbb{R}^{D \times D}$ is the output projection matrix.


**Rotary Positional Embedding (RoPE).** Here we focus on a single attention head, and omit the superscript $\bullet^{(h)}$ for simplicity. Assuming $d \in 2 \mathbb{N}$, we partition the $d$-dimensional space into $d/2$ pairs of elements. For each subspace index $f \in \left\{0, \dots, d/2-1\right\}$, we define a frequency $\theta_f = b^{-2f/d}$ with a constant $b > 1$ (typically $10,000$). We first define the family of rotation matrices $\boldsymbol{R}_m \in \mathbb{R}^{d \times d}$ for rotation step $m \in \mathbb{Z}$, which is block-diagonal and formed by $d/2$ rotation sub-matrices


```math align=center
\begin{aligned}
        \boldsymbol{R}_m &= \operatorname{diag}\left( \boldsymbol{G}_m(\theta_0), \dots, \boldsymbol{G}_m(\theta_{d/2-1}) \right),
        \  \mathrm{ where }   
        \boldsymbol{G}_m(\theta_f) = \begin{pmatrix}
            \cos(m \theta_f) & -\sin(m \theta_f) \\
            \sin(m \theta_f) & \cos(m \theta_f)
        \end{pmatrix}.
    \end{aligned}
```
Since $( \boldsymbol{G}_m )^{\mathsf{T}} = \boldsymbol{G}_{-m}$ and $\boldsymbol{G}_m \boldsymbol{G}_n = \boldsymbol{G}_{m+n}$, it follows that $\boldsymbol{R}_m = (\boldsymbol{R}_1)^m$. For the $i$-th token, we apply the rotation matrix $\boldsymbol{R}_{i-1}$ to its query and key vectors. (Note: Because token indices in our theoretical analysis start at $1$, the rotation step is shifted by $-1$, so the first token receives the identity matrix $\boldsymbol{R}_0 = \boldsymbol{I}_d$ [Su et al., 2021].) The attention logit is then


```math align=center
\begin{aligned}
        s_{i, j} 
        &= \frac{1}{\sqrt{d}} \left( \boldsymbol{R}_{i-1} \boldsymbol{q}_i \right)^{\mathsf{T}} \left( \boldsymbol{R}_{j-1} \boldsymbol{k}_j \right)
        = \frac{1}{\sqrt{d}} \boldsymbol{q}_i^{\mathsf{T}} \boldsymbol{R}_{j-i} \boldsymbol{k}_j \\
        &= \frac{1}{\sqrt{d}} \sum_{f=0}^{d/2-1} 
        \begin{pmatrix}
            q_{i, 2f+1} & q_{i, 2f+2}
        \end{pmatrix}
        \boldsymbol{G}_{j-i} (\theta_f)
        \begin{pmatrix}
            k_{j, 2f+1} \\ k_{j, 2f+2}
        \end{pmatrix}.
    \end{aligned}
```
Here, $q_{i, \bullet}, k_{j, \bullet} \in \mathbb{R}$ denote the $\bullet$-th coordinates of $\boldsymbol{q}_i$ and $\boldsymbol{k}_j$, respectively. For convenience, we write $\boldsymbol{R} \coloneqq \boldsymbol{R}_1$ and $\boldsymbol{G} \coloneqq \boldsymbol{G}_1$. If $\theta_f \equiv 0$ for all $f \in \left\{0, \dots, d/2 - 1\right\}$, then $\boldsymbol{R}$ reduces to the identity matrix, corresponding to No Positional Encoding (NoPE).


## Related Works
Let $\lambda > 0$ denote the *scaling factor* (or *inverse temperature*). When $\lambda$ is applied to the attention logits, the attention weight assigned by token $t \in \left\{1, \dots, L\right\}$ to token $i$, as defined in Section 1.1, is


```math align=center
\alpha_{t, i} 
    = \operatorname{Softmax}_i \left( \lambda s_{t, 1}, \dots, \lambda s_{t, t} \right)
    = \frac{e^{\lambda s_{t, i}}}{\sum_{j=1}^{t} e^{\lambda s_{t,j}}}.
```
If $\lambda$ is fixed, the distribution of attention weights $(\alpha_{L, 1}, \dots, \alpha_{L, L})$ tends to flatten as the sequence length $L$ increases, causing the attention mechanism to lose selectivity. Therefore, an important problem is to choose an appropriate scaling factor $\lambda = \lambda(L)$.

As summarized in Table 1, many practical implementations are used in modern large language models, such as $\lambda(L) \asymp \ln L$ in Qwen [Bai et al., 2023] and $\lambda(L) \asymp (\ln L)^2$ in YaRN [Peng et al., 2024] for length generalization. Recently, several works have analyzed the scaling factor of NoPE theoretically. Under Gaussian assumptions on the attention logits, one obtains $\lambda(L) \asymp \sqrt{\ln L}$ [Anson et al., 2025]. A random-energy-model analysis can also characterize the associated phase transitions in this setting [Giorlandino and Goldt, 2026]. Under a non-random simplex assumption, one can analyze a phase transition at $\lambda(L) \asymp \ln L$ [Chen et al., 2026].

However, these analyses typically posit a prescribed $\lambda(L)$ and study its consequences, rather than deriving the functional form. We therefore ask: *How should we design $\lambda = \lambda(L)$ to meet task requirements?*


<div id="tab:related works"></div>

**Table 1.** Attention scaling factors for different methods.

| Method | Scaling Factor | Assumption on Logits |
| --- | --- | --- |
| Qwen: Bai et al. (2023) | $\lambda(L) \asymp \ln L$ |  |
| YaRN: Peng et al. (2024) | $\lambda(L) \asymp (\ln L)^2$ |  |
| Anson et al. (2025) | $\lambda(L) \asymp \sqrt{\ln L}$ | IID Gaussian |
| Chen et al. (2026) | $\lambda(L) \asymp \ln L$ | Deterministic simplex |
| Giorlandino and Goldt (2026) | $\lambda(L) \asymp \sqrt{\ln L}$ | Correlated Gaussian |
