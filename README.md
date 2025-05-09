# Variational Monte Carlo (VMC) for Atomic and Molecular Systems

This project implements Variational Monte Carlo (VMC) simulations for various quantum systems, including the Hydrogen atom, Helium atom, Hydrogen molecule (H₂), LiH, BeH₂, and the Quantum Harmonic Oscillator (QHO). It provides tools for defining trial wavefunctions, computing energies, optimizing variational parameters, and visualizing results.

## Features

* **VMC Implementations:**

  * Hydrogen atom (ground state)
  * Helium atom (ground state)
  * Hydrogen molecule (H₂) with bonding and anti-bonding wavefunctions
  * Quantum Harmonic Oscillator (QHO)
* **Trial Wavefunctions:** User-defined trial wavefunctions with adjustable variational parameters.
* **Analytical Solutions:** Exact analytical solutions for the Hydrogen atom's ground state and the QHO wavefunction for comparison.
* **Jastrow Factor:** Includes a Jastrow correlation factor for the H₂ trial wavefunction to account for electron-electron repulsion.
* **Numerical Optimization:** Optimizes variational parameters (e.g., alpha, beta) to minimize energy.
* **Local Energy Calculation:** Implements a finite difference method for calculating the local energy.
* **Metropolis Sampling:** Efficient Monte Carlo sampling using the Metropolis algorithm.
* **Data Analysis & Visualization:** Generates interactive plots using Plotly for better result interpretation.

## Installation

Ensure the required libraries are installed:

```bash
pip install matplotlib numpy tqdm numba plotly
```

Clone the repository:

```bash
git clone https://github.com/SuvamT0071/VMC.git
cd VMC
```

## Code Structure

The project is organized into separate files for different systems and functionalities:

### Single-Atom Systems (e.g., Hydrogen, Helium, QHO)

* **Trial Wavefunctions:** Define the ground state wavefunctions (e.g., `Hyd_GS`, `Helium_GS`, `QHO_GS`).
* **Probability Density:** Calculate the wavefunction probability density (e.g., `Hyd_GSPDF`, `He_GSPDF`, `QHO_GSPDF`).
* **Local Energy:** Compute the local energy (e.g., `Hyd_local`, `He_loc_en`, `QHO_local`).
* **VMC Simulations:** Perform VMC sweeps to sample configurations and estimate energies (e.g., `Hyd_VMC`, `Helium_VMC`).
* **Optimization:** Optimize variational parameters (e.g., `Hyd_alpha_opt`, `Helium_alpha_opt`).

### Molecular Systems (e.g., H₂)

* **Proton Positioning:** Define proton positions for H₂ (`proton_points`).
* **Single-Electron Wavefunction:** Calculate single-electron wavefunctions (`single_electron_wavefunction`).
* **Jastrow Factor:** Implement the Jastrow correlation factor (`calc_Jastrow`).
* **Total Wavefunction:** Construct the full molecular wavefunction (`total_wavefunction`).
* **VMC Simulations:** Run VMC for H₂, including Jastrow factors and bonding/anti-bonding states (`H2_VMC`).
* **Parameter Optimization:** Optimize alpha and beta parameters for minimal energy (`alpha_opt`, `beta_opt`).

## Usage

To run the notebooks, open them in Jupyter:

```bash
jupyter notebook QMC_HYD.ipynb
jupyter notebook QMC_H2_matplotlib.ipynb
```

### Key Considerations for H₂

* **Bonding vs. Anti-Bonding:** Use the `sign` parameter in functions like `single_electron_wavefunction` to select bonding (`sign=1`) or anti-bonding (`sign=-1`) states.
* **Optimization Order:** Alpha is typically optimized before beta, but simultaneous optimization can yield better results.

## Recommendations for Improvement

* **Simultaneous Alpha and Beta Optimization:** Use `scipy.optimize.minimize` for joint parameter optimization.
* **Advanced Wavefunctions:** Explore more complex functional forms for improved accuracy.
* **Multiple Starting Points:** Run optimizations from different starting points to avoid local minima.
* **Increased Sample Size:** Use more VMC steps for reduced statistical noise.

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute the code as you see fit.

---

For any questions or contributions, feel free to reach out or submit a pull request. Happy coding!
