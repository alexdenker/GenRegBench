# A Stability Benchmark of Generative Regularizers for Inverse Problems

> Generative (diffusion) priors demonstrate remarkable performance in addressing inverse problems in imaging. Yet, for scientific and medical imaging, it is crucial that reconstruction techniques remain stable and reliable under imperfect settings. Typical definitions of stability encompass the notion of ``convergent regularization'', robustness to out-of-distribution data, and to inaccuracies in the forward operator or noise model. We evaluate these properties numerically. Furthermore, we benchmark generative approaches against modern optimization-based methods inspired by the widely used variational techniques. Our results give insights for which settings and applications generative priors can deliver state-of-the-art reconstructions, and on those in which they fall short or may even be problematic.

We want to compare different generative and non-generative algorithms for image reconstruction and image restoration. 

| Method                          | Note | Reference | 
|--------------------------------------|------|------|
| DiffPIR | Diffusion models for plug-and-play image restoration    | [Zhu et al. (2023)](https://arxiv.org/abs/2008.13751) |
|  RED-Diff  | A Variational Perspective on Solving Inverse Problems with Diffusion Models   | [Mardani et al. (2023)](https://arxiv.org/abs/2305.04391) |
|  DPS  | Diffusion Posterior Sampling for General Noisy Inverse Problems | [Chung et al. (2022)](https://arxiv.org/abs/2209.14687) |
|  DMPlug  | DMPlug: A Plug-in Method for Solving Inverse Problems with Diffusion Models | [Wang et al. (2024)](https://arxiv.org/abs/2405.16749) |
| FlowDPS | FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems | [Kim et al. (2025)](https://arxiv.org/abs/2503.08136) | 


### Datasets

| Dataset                          | Note | Reference | 
|--------------------------------------|------|------|
|  Walnut             |       |   [Der Sarkissian et al. (2019)](https://arxiv.org/abs/1905.04787) |
| ... | ... | ... |


