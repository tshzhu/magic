# Conclusions

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

This report introduces the effective sequence length $\mathcal{E}_{\beta}(\boldsymbol{\alpha})$ for attention heads and shows that attention heads exhibit distinct $\mathcal{E}_{\beta}$ scaling. Building on this, we derive scaling factors to achieve a target $\mathcal{E}_{\beta}^*(L)$ for both RoPE and NoPE. The key conclusions are:


- **Different route from prior work.** Rather than starting from a prescribed logit model or a fixed scaling schedule, we start from the task requirements of each attention head.
- **ESS as the metric.** $\mathcal{E}_{\beta}(\boldsymbol{\alpha})$ maps softmax concentration to an effective sequence length, enabling comparison of head behavior across tasks and context lengths.
- **Task-induced head differentiation.** Retrieval heads benefit from a near-constant ESS, whereas global aggregation heads require ESS that grows linearly with $L$; robustness and aggregation results indicate that ESS captures this split.
- **Temperature as a design consequence.** Given a head-specific target $\mathcal{E}_{\beta}^*(L)$, the NoPE and RoPE scaling laws determine the required temperature. This covers constant, linear, polynomial, and boundary regimes, enabling a more flexible design than a single global temperature.
