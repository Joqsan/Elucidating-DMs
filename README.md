An implementation of the sampling algorithms in [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) [Karras et al., 2022].

- Algorithm 1. Deterministic sampling. With two versions:
    - Heun's 2nd order method.
    - Euler's (1st order) method (aka Heun's without steps 6-8).
- Algorithm 2. Stochastic sampling.
    - A modified Euler-Maruyama method.

(the algorithm version for the iDDPM case is not yet implemented).
