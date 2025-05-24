# Physics-Based Mechanisms for Radiation Effects in Materials

## Mathematical and Technical Overview

The `rad_ml/physics` module implements a multi-scale, multi-physics simulation framework for radiation effects in materials, combining classical field theory, quantum field theory, stochastic processes, and radiation transport. The following sections detail the mathematical foundations and computational strategies of each major component.

---

## 1. Field Theory Models — `field_theory.hpp`

**Mathematical Foundation:**
Classical field theory models the evolution of defect concentrations $C_i(\mathbf{r}, t)$ in a material using partial differential equations derived from a free energy functional $F[\{C_i\}]$:

$$
F[\{C_i\}] = \int \left[ \frac{\kappa}{2} \sum_i |\nabla C_i|^2 + \sum_{i,j} \gamma_{ij} C_i C_j \right] d^3\mathbf{r}
$$

The time evolution is governed by gradient flow (e.g., Cahn-Hilliard or Allen-Cahn equations):

$$
\frac{\partial C_i}{\partial t} = -M_i \frac{\delta F}{\delta C_i}
$$

where $M_i$ is a mobility parameter.

**Technical Implementation:**
- Discretizes fields on a 3D grid: $C_i[x, y, z]$.
- Computes functional derivatives $\delta F / \delta C_i$ numerically.
- Solves time evolution equations using explicit or implicit schemes.
- Calculates clustering ratios and other statistical measures.

---

## 2. Quantum Field Theory Models — `quantum_field_theory.hpp`

**Mathematical Foundation:**
Quantum field theory (QFT) extends the description to quantum fields $\hat{\phi}(\mathbf{r}, t)$, governed by equations such as:

- **Klein-Gordon Equation (scalar fields):**
  $$
  \left( \frac{\partial^2}{\partial t^2} - c^2 \nabla^2 + m^2 c^4 / \hbar^2 \right) \phi(\mathbf{r}, t) = 0
  $$
- **Dirac Equation (spinor fields):**
  $$
  (i\hbar \gamma^\mu \partial_\mu - mc) \psi = 0
  $$
- **Maxwell Equations (electromagnetic fields):**
  $$
  \nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}, \quad \nabla \times \mathbf{B} - \frac{1}{c^2} \frac{\partial \mathbf{E}}{\partial t} = \mu_0 \mathbf{J}
  $$

Quantum corrections to defect formation energy $E_{\text{defect}}$ and tunneling probabilities $P_{\text{tunnel}}$ are computed using path integrals and WKB approximations:

$$
P_{\text{tunnel}} \sim \exp\left(-\frac{2}{\hbar} \int_{x_1}^{x_2} \sqrt{2m(V(x) - E)} dx \right)
$$

**Technical Implementation:**
- Lattice discretization of quantum fields.
- Split-operator and Crank-Nicolson methods for time evolution.
- Calculation of propagators, correlation functions, and zero-point energy.

---

## 3. Quantum Integration and Correction — `quantum_integration.hpp`

**Mathematical Foundation:**
Quantum corrections are applied when environmental parameters (temperature $T$ and feature size $d$) cross certain thresholds. The quantum enhancement factor $Q$ is modeled as:

$$
Q(T, d) = 1 + \alpha_T \exp\left(\frac{T_0}{T} - 1\right) + \alpha_d \exp\left(\frac{d_0}{d} - 1\right)
$$

where $\alpha_T, \alpha_d$ are scaling coefficients.

**Technical Implementation:**
- Decision logic for when to apply quantum corrections.
- Parameter mapping from material and environmental properties to QFT parameters.
- Correction of classical defect distributions using quantum models.

---

## 4. Quantum Models and Extensions — `quantum_models.hpp`

**Mathematical Foundation:**
Models include decoherence (modeled as exponential decay of off-diagonal density matrix elements) and dissipation (Lindblad or Caldeira-Leggett formalism):

![equation](https://latex.codecogs.com/svg.latex?\frac{d\rho}{dt}=-\frac{i}{\hbar}[H,\rho]+\mathcal{L}_{\text{decoh}}[\rho]+\mathcal{L}_{\text{diss}}[\rho])




Transition probabilities and displacement energies are computed using quantum statistical mechanics and scattering theory.

**Technical Implementation:**
- Extended QFT parameters for decoherence and dissipation.
- Simulation of displacement cascades using quantum-corrected cross-sections.
- Multi-particle field interactions and energy transfer.

---

## 5. Stochastic Models — `stochastic_models.hpp`

**Mathematical Foundation:**
Defect evolution is modeled by stochastic differential equations (SDEs):

$$
d\mathbf{C} = \mathbf{A}(\mathbf{C}, t) dt + \mathbf{B}(\mathbf{C}, t) d\mathbf{W}_t
$$

where $\mathbf{A}$ is the drift term, $\mathbf{B}$ is the diffusion term, and $d\mathbf{W}_t$ is a Wiener process increment.

**Technical Implementation:**
- Euler-Maruyama and higher-order SDE solvers.
- Calculation of drift and diffusion terms from material and environmental parameters.
- Statistical analysis of simulation results (mean, variance, error bars).

---

## 6. Radiation Transport Equation Models — `transport_equation.hpp`

**Mathematical Foundation:**
The Boltzmann transport equation for particle fluence $\Phi(\mathbf{r}, \Omega, E, t)$:

$$
\frac{\partial \Phi}{\partial t} + \Omega \cdot \nabla \Phi + \Sigma_t \Phi = \int \Sigma_s(\Omega' \rightarrow \Omega, E' \rightarrow E) \Phi(\Omega', E') d\Omega' dE' + S
$$

where $\Sigma_t$ is the total cross-section, $\Sigma_s$ is the scattering cross-section, and $S$ is the source term.

**Technical Implementation:**
- Tensor-based discretization of spatial, angular, and energy variables.
- Iterative solvers for the transport equation (e.g., discrete ordinates, Monte Carlo).
- Calculation of dose distributions and attenuation factors.

---

## Scientific Impact and Integration

By combining these mathematical models, the framework enables predictive, physically grounded simulation of radiation effects from the atomic to the device scale. The modular design allows integration with machine learning and control systems, supporting the development of next-generation radiation-tolerant technologies.

---

**References:**
- J. F. Ziegler, "The Stopping and Range of Ions in Matter," Pergamon Press, 1985.
- P. Hänggi, P. Talkner, and M. Borkovec, "Reaction-rate theory: fifty years after Kramers," Rev. Mod. Phys., vol. 62, no. 2, pp. 251–341, 1990.
- C. Kittel, "Introduction to Solid State Physics," Wiley, 8th Edition.
- S. Datta, "Quantum Transport: Atom to Transistor," Cambridge University Press, 2005.
- H. Risken, "The Fokker-Planck Equation: Methods of Solution and Applications," Springer, 2nd Edition.
- NASA Goddard Space Flight Center. (2016). Radiation Effects and Analysis Home Page. https://radhome.gsfc.nasa.gov/
