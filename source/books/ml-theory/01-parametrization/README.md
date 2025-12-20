# Signal Propagation


```alert type=tip title="TODO"
- 整理clean的MUP推导
- 后续再补充NTK细节
```


## Parametrization Matters in Neural **Nets**

![Convergence of a large 22-layer CNN (left) and a small 30-layer CNN (right) under Xavier and He initialization. (Source: Figure 2 and 3 in He et al., 2015)](figures/he2015_figure2_3.png)

- Question: How to find the "optimal" (in what sense) parametrization
- Approach: Control the signal propagation at initialization [^he2015delving]

[^he2015delving]: He, Kaiming, Zhang, Xiangyu, Ren, Shaoqing, and Sun, Jian. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." *Proceedings of the IEEE international conference on computer vision*. pp. 1026--1034, 2015.

## Typical Size of Sum of iid Random Variables

For iid $x_1, \dots, x_n$ sample from $x$, we have

* Law of large numbers (LLN)

$$
\begin{equation}
    \frac{1}{n} \sum_{i=1}^{n} x_i \xrightarrow{\textrm{a.s. / p}} \mathbb{E}[x]
\end{equation}
$$

* Central limit theorem (CLT)

$$
\begin{equation}
    \frac{1}{\sqrt{n}} \sum_{i=1}^{n} (x_i - \mathbb{E}[x]) \xrightarrow{\textrm{d}} \mathcal{N}(0, \mathrm{Var}\left[x\right])
\end{equation}
$$

Hence, we have the following basic intuition regarding the size of a sum of $x_i$.

When $n$ is large, $\sum_{i=1}^n x_i$ has typical size

$$
\begin{cases}
    \Theta(n) & \text{if } \mathbb{E}[X] \neq 0 \\
    \Theta(\sqrt{n}) & \text{otherwise, \text{w/ high prob}}
\end{cases}
$$

