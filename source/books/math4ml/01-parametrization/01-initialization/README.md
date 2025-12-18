# Maximal Update Parametrization


## abc-Parametrization

An $L$-hidden-layer MLP $f(\xi) \in \mathbb{R}$ with input $\xi \in \mathbb{R}^d$, nonlinearity $\phi: \mathbb{R} \to \mathbb{R}$, weights $\tilde{W}^1 \in \mathbb{R}^{n \times d}$, $\tilde{W}^2, \dots, \tilde{W}^L \in \mathbb{R}^{n \times n}$, $\tilde{W}^{L+1} \in \mathbb{R}^{1 \times n}$, is defined as

$$
h^1(\xi) = \tilde{W}^1 \xi, \enspace x^l(\xi) = \phi(h^l(\xi)), \enspace h^{l+1}(\xi) = \tilde{W}^{l+1} x^l(\xi), \quad 1 \le l \le L,
$$

where $f(\xi) = h^{L+1}(\xi)$.

The abc-parametrization is specified by a set of numbers $\set{a_l, b_l}_{l} \cup \set{c}$ such that

* Weights with multipliers $\tilde{W}^l = n^{-a_l} W^l$
* Actual trainable weights $W^l_{\alpha \beta} \stackrel{\textrm{iid}}{\sim} \mathcal{N}(0, n^{-2b_l})$
* Learning rate $\eta = \eta_0 n^{-c}$ [^yang2021feature]

[^yang2021feature]: Yang, Greg and Hu, Edward J. "Tensor programs IV: Feature learning in infinite-width neural networks." *International Conference on Machine Learning*. PMLR, pp. 11727--11737, 2021.

## Tensor Program Ansatz

```alert type=note
As the width $n$ becomes large, the entries of preactivations and their gradients become $\textcolor{red}{\text{approximately iid}}$ - both at initialization and during training, which is rigorously proved and utilized in the Tensor Program series.

```

* For sufficiently large n, the asymptotic behavior of vector h(\xi) (resp. \partial_h f(\xi)) can be characterized by any of its coordinates - a random variable h(\xi)_\alpha (resp. (\partial_h f(\xi))_\alpha)
* A vector x is said to have typical size \Theta(1) if \|x\|^2 / n \to 1 as n \to \infty
* This ansatz ensures that, in most cases, the order of taking limits for different layer widths can be freely exchanged [^1]

## Condition 1. Stable Forward Propagation

* Recall h^1(\xi) = \tilde{W}^1 \xi and \tilde{W}^1_{\alpha \beta} \stackrel{\textrm{iid}}{\sim} \mathcal{N}(0, n^{-2a_1 -2b_1}), so that h^1(\xi) has iid coordinates h^1(\xi)_{\alpha} = \sum_{\beta} \tilde{W}^1_{\alpha \beta} \xi_{\beta} \sim \mathcal{N}(0, n^{-2a_1-2b_1} \| \xi \|^2). Since d is fixed, \boxed{a_1 + b_1 = 0}
* Condition on h^1 = \Theta(1), we have x^1 = \phi(h^1) = \Theta(1). By CLT, we have h^2_{\alpha} = \sum_{\beta} \tilde{W}^2_{\alpha \beta} x^1_{\beta} \stackrel{\textrm{d}}{\to} \mathcal{N}(0, n^{1-2a_2-2b_2}), so we have a_2 + b_2 = 1/2
* By induction, suppose that h^{l-1} = \Theta(1) has iid coordinates, we have constraints \boxed{a_{2:L} + b_{2:L} = 1/2} and \boxed{a_{L+1} + b_{L+1} \ge 1/2} if require f(\cdot) = O(1).

## *Gaussian Process Behavior

Under the stable parametrization \tilde{W}^1_{\alpha \beta} \stackrel{\textrm{iid}}{\sim} \mathcal{N}(0, 1), \enspace \tilde{W}^{2:L+1}_{\alpha \beta} \stackrel{\textrm{iid}}{\sim} \mathcal{N}(0, n^{-1}), coordinates of each preactivation h^l(\cdot) tend to iid Gaussian processes \mathcal{GP}(0, \Sigma^l) as n \to \infty.

For the first layer, h^1 has iid coordinates h^1(\xi)_{\alpha} \stackrel{\textrm{iid}}{\sim} \mathcal{N}(0, \|\xi\|^2) and covariance

By induction, we should disentangle the widths into n_l \triangleq \mathrm{dim}\left(h^l(\cdot)\right), with weights \tilde{W}^l_{\alpha \beta} \stackrel{\textrm{iid}}{\sim} \mathcal{N}(0, n_{l-1}^{-1}), and take the limits n_1, \dots, n_L \to \infty sequentially.

Conditioned on h^l, the coordinates h^{l+1}_{\alpha}(\cdot) are iid centered Gaussians with covariance

Then by LLN and the induction hypothesis, as n_l \to \infty, we have

and the convergence of \mathbb{E}\left[\exp (i (p h^{l+1}_{\alpha}(\xi) + q h^{l+1}_{\alpha}(\zeta)))\right], which implies that h^{l+1}_{\alpha}(\cdot) are unconditioned Gaussians.

## NNGP and Limits

Taking the limits n_1, \dots, n_L \to \infty sequentially yields the **neural network Gaussian process (NNGP)** [^2]. A natural question is whether taking all widths to infinity simultaneously, i.e., n_1 = \dots = n_L \triangleq n \to \infty, leads to the same result. If so, we can recursively compute f \sim \mathcal{GP}(0, \Sigma^{L+1}) by

and thus predict the behavior of f at initialization.

* One approach is to use an extended version of CLT for exchangeable sequences [^3], but this still breaks down if weights are tied (e.g., f(\xi) = A A v or RNN).
* Another approch relies on the Gaussian conditioning technique which works for any architecture and weight tying scheme [^4].

## Condition 2. Nontrivial Output Evolution

Return to the abc-parametrization, we require that the network output f(\xi) changes by \Theta(1) within \Theta(1) training time. If the change is too small, the network does not learn; if too large, the network diverges. It is summarized as \textcolor{red}{\partial_t f(\cdot) = \Theta(1).}

Denote \theta = \mathrm{vec}\left(W^1, \dots, W^{L+1}\right), loss function \mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \ell(f(\xi_i; \theta), z_i)

The gradient flow writes

For any input \xi, by chain rule

For any two inputs \xi, \zeta

$$
d f = (\partial_{h^l} f)^{\transpose} d h^l = n^{-a_l} (\partial_{h^l} f)^{\transpose} d W^l x^{l-1} = \mathrm{tr}\left(n^{-a_l} x^{l-1} (\partial_{h^l} f)^{\transpose} d W^l\right)
$$

$$
x^0(\xi) = \xi, \quad h^{L+1}(\xi) = f(\xi)
$$

For the first part, the stable condition gives

For the second part, consider $1 \le l \le L$,

$$
\partial_{h^l}f^{\transpose} = \tilde{W}^{L+1} D^{L} \cdots \tilde{W}^{l+1} D^{l}
$$

where each $D^{l}$ has $\Theta(1)$ coordinates, and $\tilde{W}^{2:L} (\tilde{W}^{2:L})^{\transpose}$ are approximately a diagonal matrix with $\Theta(1)$ coordinates.

So the second part has the same scaling as $\tilde{W}^{L+1} (\tilde{W}^{L+1})^{\transpose}$.

By integrating the above constraints into $\partial_t f(\cdot) = \Theta(1)$, we have

Recall

$$
\begin{aligned}
\partial_t h^1(\xi)
&= \partial_t \tilde{W}^1 \xi \\
&= n^{-a_1} \partial_t W^1 \xi \\
&= -\eta n^{-a_1} \frac{1}{N} \sum_{i=1}^{N} \ell'(f(\xi_i), z_i) \partial_{W^1} f(\xi_i) \xi \\
&= -\eta n^{-2a_1} \frac{1}{N} \sum_{i=1}^{N} \ell'(f(\xi_i), z_i) \partial_{h^1} f(\xi_i) \xi_i^{\transpose} \xi
\end{aligned}
$$

$$
\partial_t h^l(\xi) = \textcolor{blue}{\partial_t \tilde{W}^l x^{l-1}(\xi)} + \textcolor{green}{\tilde{W}^l \partial_t x^{l-1}(\xi)}
$$

$$
\begin{aligned}
    	extcolor{blue}{\partial_t \tilde{W}^l x^{l-1}(\xi)}
&= n^{-a_l} \partial_t W^l x^{l-1}(\xi) \\
&= -\eta n^{-a_l} \frac{1}{N} \sum_{i=1}^{N} \ell'(f(\xi_i), z_i) \partial_{W^l} f(\xi_i) x^{l-1}(\xi) \\
&= -\eta n^{-2a_l} \frac{1}{N} \sum_{i=1}^{N} \ell'(f(\xi_i), z_i) \partial_{h^l} f(\xi_i) (x^{l-1}(\xi_i))^{\transpose} x^{l-1}(\xi) \\
&= \Theta(n^{-c -2a_l -a_{L+1} -b_{L+1} + 1}) \\
\\
    	extcolor{green}{\tilde{W}^l \partial_t x^{l-1}(\xi)}
&= n^{-a_l} W^l (\phi'(h^{l-1}(\xi)) \odot \textcolor{purple}{\partial_t h^{l-1}(\xi)})
= \Theta(n^{-a_l -b_l + 1/2}) = \Theta(1)
\end{aligned}
$$

For the first layer, since d h^1(\xi)_{\alpha} = d W^1_{\alpha} \xi = (e^1_{\alpha})^{\transpose} d W^1 \xi, where e^1_{\alpha} \in \mathbb{R}^{n_1} is the \alpha-th onehot vector, we have \nabla_{W^1} h^1(\xi)_{\alpha} = e^1_{\alpha} \xi^{\transpose}. So that K^1_{W^1}(\xi, \zeta)_{\alpha \beta} = \delta_{\alpha \beta} \xi^{\transpose} \zeta, which implies K^1(\xi, \zeta) = \Sigma^1(\xi, \zeta) = \xi^{\transpose} \zeta.

Suppose K^l_{W^{1:l}}(\xi, \zeta)_{\alpha \beta} \xrightarrow{\textrm{p}} K^l(\xi, \zeta) as n_1, \dots, n_{l-1} \to \infty sequentially, decompose

For the first part, since d h^{l+1}_{\alpha} = n_l^{-1/2} (e^{l+1}_{\alpha})^{\transpose} d W^{l+1} x^l, where e^{l+1}_{\alpha} \in \mathbb{R}^{n_{l+1}} is the \alpha-th onehot vector, we have \partial_{W^{l+1}} h^{l+1}_{\alpha} = n_l^{-1/2} e^{l+1}_{\alpha} (x^l)^{\transpose}. So that

For the second part, since d h^{l+1}_{\alpha} = (\partial_{h^l} h^{l+1}_{\alpha})^{\transpose} (\partial_{\mathrm{vec}(W^{1:l})} h^l)^{\transpose} \mathrm{vec}\left(d W^{1:l}\right),

where by the induction hypothesis, \partial_{\mathrm{vec}(W^{1:l})} h^l(\xi)^{\transpose} \partial_{\mathrm{vec}(W^{1:l})} h^l(\zeta) \xrightarrow{\textrm{p}} K^l(\xi, \zeta) \cdot I_{n_l \times n_l}.

And since d h^{l+1}_{\alpha} = n_l^{-1/2} W^{l+1}_{\alpha \bullet} D^l d h^l, as n_l \to \infty,

where \dot{\Sigma}^{l+1}(\xi, \zeta) = \underset{h \sim \mathcal{GP}(0, \Sigma^l)}{\mathbb{E}}\left[\phi'(h(\xi)) \phi'(h(\zeta))\right]. Then by the Slutsky Theorem,

which gives

Taking the limits n_1, \dots, n_L \to \infty sequentially, the NTK can be computed recursively by

Similar to the NNGP case, the order of taking limits remains a important consideration.

In Yang (2020) [^6], this issue is resolved by proving the gradient independence assumption (GIA), which says that the weight W is **approximately independent** from W^{\transpose} used in backward propagation as n \to \infty.

Then, applying LLN to the decomposition yields convergence.

* By a stronger version of convergence argument in Yang (2021) [^7], even if both W and W^{\transpose} are used in forward propagation, the NTK is still convergent but to a different limit.

## NTP Lacks Feature Learning

With a quadratic loss, the linear dynamics of NTK show that f_t(\xi) is controlled by the reproducing kernel Hilbert space (RKHS) generated by K^{L+1}(\cdot, \cdot). There exists \alpha_1, \dots, \alpha_m such that

Although an MLP under NTP can be trained, its features do not change during training. Thus, the NTK limits **do not enable linear transfer learning**: after pretraining, replacing and retraining the linear classifier on top of fixed features.

In NTP, pretraining gives no improvement on such kind of transfer learning.

## Condition 3. Maximal Feature Learning

We require that any embedding h^{1:L}(\cdot) enables linear transfer learning, so each preactivation should evolve "maximally" by \Theta(1) within \Theta(1) training time to allow feature learning. This is summarized as \textcolor{red}{\partial_t h^{1:L}(\cdot) = \Theta(1)}.

Recall

$$
\begin{aligned}
\partial_t h^1(\xi)
&= \partial_t \tilde{W}^1 \xi \\
&= n^{-a_1} \partial_t W^1 \xi \\
&= -\eta n^{-a_1} \frac{1}{N} \sum_{i=1}^{N} \ell'(f(\xi_i), z_i) \partial_{W^1} f(\xi_i) \xi \\
&= -\eta n^{-2a_1} \frac{1}{N} \sum_{i=1}^{N} \ell'(f(\xi_i), z_i) \partial_{h^1} f(\xi_i) \xi_i^{\transpose} \xi
\end{aligned}
$$

$$
\partial_t h^l(\xi) = \textcolor{blue}{\partial_t \tilde{W}^l x^{l-1}(\xi)} + \textcolor{green}{\tilde{W}^l \partial_t x^{l-1}(\xi)}
$$

$$
\begin{aligned}
    	extcolor{blue}{\partial_t \tilde{W}^l x^{l-1}(\xi)}
&= n^{-a_l} \partial_t W^l x^{l-1}(\xi) \\
&= -\eta n^{-a_l} \frac{1}{N} \sum_{i=1}^{N} \ell'(f(\xi_i), z_i) \partial_{W^l} f(\xi_i) x^{l-1}(\xi) \\
&= -\eta n^{-2a_l} \frac{1}{N} \sum_{i=1}^{N} \ell'(f(\xi_i), z_i) \partial_{h^l} f(\xi_i) (x^{l-1}(\xi_i))^{\transpose} x^{l-1}(\xi) \\
&= \Theta(n^{-c -2a_l -a_{L+1} -b_{L+1} + 1}) \\
\\
	extcolor{green}{\tilde{W}^l \partial_t x^{l-1}(\xi)}
&= n^{-a_l} W^l (\phi'(h^{l-1}(\xi)) \odot \textcolor{purple}{\partial_t h^{l-1}(\xi)})
= \Theta(n^{-a_l -b_l + 1/2}) = \Theta(1)
\end{aligned}
$$

\boxed{c + 2a_{2:L} + a_{L+1} + b_{L+1} \ge 1}

$$
1 = c + 2a_{L+1} + \min\{2a_1 + 2b_{L+1}, 0\} = \min\{a_{L+1} + b_{L+1}, \; c + 2a_{L+1}\}
$$

$$
\begin{aligned}
& \textcolor{blue}{\partial_t \tilde{W}^{L+1} x^{L}(\xi)} = \Theta(n^{-c -2a_{L+1} + 1}) = O(1) \\
& \textcolor{green}{\tilde{W}^{L+1} \partial_t x^{L}(\xi)} = \Theta(n^{-a_{L+1} -b_{L+1} +1/2}) = O(n^{-1/2})
\end{aligned}
$$

$$
\begin{aligned}
& \textcolor{blue}{\partial_t \tilde{W}^l x^{l-1}(\xi)} = \Theta(1), \enspace \textcolor{green}{\tilde{W}^l \partial_t x^{l-1}(\xi)} = \Theta(1), \quad 2 \le l \le L \\
& \textcolor{blue}{\partial_t \tilde{W}^{L+1} x^{L}(\xi)} = \Theta(1), \enspace \textcolor{green}{\tilde{W}^{L+1} \partial_t x^{L}(\xi)} = \Theta(n^{-1/2})
\end{aligned}
$$

$$
a_1 = -\frac{1+c}{2}, \quad a_{2:L} = -\frac{c}{2}, \quad a_{L+1} = \frac{1-c}{2}, \quad b_{1:L+1} = \frac{1+c}{2}
$$

$$
\begin{aligned}
	extcolor{blue}{\partial_t \tilde{W}^l x^{l-1}(\xi)}
&= -\eta_l n^{-2a_l} \frac{1}{N} \sum_{i=1}^{N} \ell'(f(\xi_i), z_i) \partial_{h^l} f(\xi_i) (x^{l-1}(\xi_i))^{\transpose} x^{l-1}(\xi) \\
&= \begin{cases}
\Theta(n^{-c_1 -2a_1 -a_{L+1} -b_{L+1}}) & l = 1 \\
\Theta(n^{-c_l -2a_l -a_{L+1} -b_{L+1} + 1}) & 2 \le l \le L \\
\Theta(n^{-c_{L+1} -2a_{L+1} + 1}) & l = L+1
\end{cases}
\quad = \Theta(1)
\end{aligned}
$$

$$
a_l \gets a_l + \theta_l, \quad b_l \gets b_l - \theta_l, \quad c_l \gets c_l - 2\theta_l
$$

$$
	extcolor{magenta}{b_1 = 0, \quad b_{2:L} = \tfrac{1}{2}, \quad b_{L} = 1, \quad c_1 = -1, \quad c_{2:L} = 0, \quad c_{L+1} = 1}
$$

[^yang2021tuning]: Yang, Greg and Hu, Edward J. "Tensor programs IV: Feature learning in infinite-width neural networks." *International Conference on Machine Learning*. PMLR, pp. 11727--11737, 2021.

## \muP Admits Feature Learning

## Zero-Shot Hyperparameter Transfer in \muP

## mup package

See the [mup package](https://github.com/microsoft/mup) for \muP and Hyperparameter Transfer (\muTransfer) in PyTorch

```python
    !pip install mup
    from mup import MuReadout, make_base_shapes, set_base_shapes, MuSGD, MuAdam

    class MyModel(nn.Module):
        def __init__(self, width, ...):
            
            ### In model definition, replace output layer with MuReadout
            # readout = nn.Linear(width, d_out)
            readout = MuReadout(width, d_out)

        def forward(self, ...):
            
            ### If using a transformer, make sure to use
            ###   1/d instead of 1/sqrt(d) attention scaling
            # attention_scores = query @ key.T / d**0.5
            attention_scores = query @ key.T * 8 / d

    ### Instantiate a base model
    base_model = MyModel(width=1)

    ### Instantiate a "delta" model that differs from the base model
    ###   in all dimensions ("widths") that one wishes to scale.
    delta_model = MyModel(width=2)

    ### Instantiate the target model (the model you actually want to train).
    ### This should be the same as the base model except 
    ###   the widths could be potentially different.
    model = MyModel(width=100)

    ### Set base shapes
    set_base_shapes(model, base_model, delta=delta_model)

    ### Replace your custom init, if any
    for param in model.parameters():
        ### If initializing manually with fixed std or bounds,
        ### then replace with same function from mup.init
        # torch.nn.init.uniform_(param, -0.1, 0.1)
        mup.init.uniform_(param, -0.1, 0.1)

    ### Use the optimizers from `mup.optim` instead of `torch.optim`
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer = MuSGD(model.parameters(), lr=0.1)
```

$$
\begin{aligned}
    extcolor{blue}{\partial_t \tilde{W}^l x^{l-1}(\xi)}
&= -\eta_l n^{-2a_l} \frac{1}{N} \sum_{i=1}^{N} \ell'(f(\xi_i), z_i) \partial_{h^l} f(\xi_i) (x^{l-1}(\xi_i))^{\transpose} x^{l-1}(\xi) \\
&= \begin{cases}
\Theta(n^{-c_1 -2a_1 -a_{L+1} -b_{L+1}}) & l = 1 \\
\Theta(n^{-c_l -2a_l -a_{L+1} -b_{L+1} + 1}) & 2 \le l \le L \\
\Theta(n^{-c_{L+1} -2a_{L+1} + 1}) & l = L+1
\end{cases}
\quad = \Theta(1)
\end{aligned}
$$

$$
a_l \gets a_l + \theta_l, \quad b_l \gets b_l - \theta_l, \quad c_l \gets c_l - 2\theta_l
$$

$$
    extcolor{magenta}{b_1 = 0, \quad b_{2:L} = \tfrac{1}{2}, \quad b_{L} = 1, \quad c_1 = -1, \quad c_{2:L} = 0, \quad c_{L+1} = 1}
$$


[^4]: Yang, Greg. "Tensor program I: Wide feedforward or recurrent neural networks of any architecture are Gaussian processes." *Advances in Neural Information Processing Systems*, vol. 32. 2019.

[^5]: Jacot, Arthur, Gabriel, Franck, and Hongler, Clément. "Neural tangent kernel: Convergence and generalization in neural networks." *Advances in neural information processing systems*, vol. 31. 2018.

[^6]: Yang, Greg. "Tensor programs II: Neural tangent kernel for any architecture." *arXiv preprint arXiv:2006.14548* (2020).

[^7]: Yang, Greg. "Tensor programs III: Neural matrix laws." *arXiv preprint arXiv:2009.10685* (2021).