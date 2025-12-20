# Optimization Trajectory


```mermaid
flowchart LR
    subgraph method
        sde{SDE}
        adam{Adam}
    end

    li2022what["2022-[ICLR]-What Happens after SGD Reaches Zero Loss--A Mathematical Framework"]
    gu2023why["2023-[ICLR]-Why (and When) does Local SGD Generalize Better than SGD"]
    li2025adam["2025-[NeurIPS]-Adam Reduces a Unique Form of Sharpness Theoretical Insights Near the Minimizer Manifold"]
    cohen2025under["2025-[ICLR]-Understanding Optimization in Deep Learning with Central Flows"]

    li2022what --> gu2023why
    li2022what & gu2023why --> li2025adam

    sde -.-> li2022what & gu2023why & cohen2025under
    adam -.-> li2025adam & cohen2025under
```


- progressive shapening
- slow sde
- central flow, 能否用同样的time-average来估计谱的变换？
- adam ode 2025-ODE approximation for the Adam algorithm General and overparametrized setting
- valley-river
    - saddle cascade https://scottpesme.github.io/Articles/Neurips_s2s-CR.pdf
    - deep relu saddle https://arxiv.org/pdf/2505.21722
    - 感觉llm和其他loss差别还是挺大的，希望找到一些不变量
- saddle cascade
