# Learning Allen-Cahn Equation using a Modified DeepONet Architecture

This notebook demonstrates learning the operator for the **Allen-Cahn equation**, a reaction-diffusion PDE:

$$
\frac{\partial u}{\partial t} = D \Delta u + u - u^3, \quad x \in [0, 1]^2, \quad t \in [0, T]
$$

where $u(x, t)$ is the order parameter and $D$ is the diffusion coefficient.

### Deep Operator Network (DeepONet)
DeepONet learns a mapping from input functions to output functions via two subnetworks:
1.  **Branch Network**: Encodes the input function $u_0(\cdot)$ (discretized at sensors).
2.  **Trunk Network**: Encodes the query locations $(x, y, t)$.

### Modified DeepONet Architecture
We implement a variation called **Modified DeepONet**:
*   Instead of a simple dot product, the outputs of Branch and Trunk are combined via **element-wise multiplication**.
*   The result is processed by a final linear layer (`self.fc`) to produce the scalar output.
*   This architecture often shows improved convergence and accuracy over the vanilla dot-product DeepONet.

$$
G(u)(y) \approx \text{Linear}(\text{Branch}(u) \odot \text{Trunk}(y))
$$

Inspired by the paper: [Learning the solution operator of parametric partial differential equations with physics-informed DeepOnets](https://doi.org/10.48550/arXiv.2103.10974)


<img width="2287" height="1415" alt="ac_output" src="https://github.com/user-attachments/assets/b291ba86-50d2-45d2-b786-a8afc93179e7" />
