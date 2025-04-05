# Quantum Monte Carlo for Hydrogen Atom, Helium atom and Quantum Harmonic Oscillator

This project implements a **Variational Monte Carlo (VMC)** simulation for the **Hydrogen atom's ground state wavefunction** and the **Quantum Harmonic Oscillator (QHO)** using Python. It includes functions for defining the trial wavefunction, probability density function (PDF), local energy calculation, and Monte Carlo sampling to optimize the variational parameter (alpha).

---

## Features
✅ Variational Monte Carlo (VMC) implementation for the Hydrogen atom and QHO  
✅ Analytic solution for the Hydrogen atom's ground state wavefunction and QHO wavefunction  
✅ Numerical optimization of the variational parameter 
✅ Efficient Monte Carlo sampling with controlled step size 
✅ Stable PDF calculation with improved numerical stability 

---

## Installation

To run this project, ensure you have the following libraries installed:

```bash
pip install numpy tqdm
```

---

## Code Structure

- **`Hyd_GS()`** : Defines the trial wavefunction for the ground state of the Hydrogen atom.  
- **`Hyd_GSPDF()`** : Computes the probability density function (PDF) for the wavefunction of Hydrogen atom.  
- **`Hyd_local()`** : Calculates the local energy of the Hydrogen atom's wavefunction.  
- **`Hyd_VMC()`** : Performs Variational Monte Carlo sweeps to sample positions and calculate energy for Hydrogen atom.  
- **`Hyd_alpha_opt()`** : Optimizes the variational parameter (alpha) for minimum energy.  
- **`psi_analytic()`** : Defines the exact analytic solution for the Hydrogen atom's ground state.  
- **`QHO_GS()`** : Defines the trial wavefunction for the Quantum Harmonic Oscillator.  
- **`QHO_GSPDF()`** : Computes the PDF for the QHO wavefunction.  
- **`QHO_local()`** : Calculates the local energy of the QHO wavefunction.
- **`Helium_GS`**: Defines the trial wavefunction for the ground state of the Helium atom
- **`He_GSPDF`**:  Computes the probability density function (PDF) for the wavefunction for the Helium atom. 
- **`He_loc_en`**: Calculates the local energy of the Helium atom's wavefunction.
- **`Helium_VMC`**: Performs Variational Monte Carlo sweeps to sample positions and calculate energy for Helium atom
- **`Helium_alpha_opt`**: Optimizes the variational parameter (alpha) for minimum energy.  

---

## Usage

### Example Code
```python
import numpy as np
from tqdm import tqdm

def Hyd_GS(r, alpha=2):
    return alpha * r * np.exp(-alpha * r)

def Hyd_GSPDF(r, alpha=1):
    wave_func = Hyd_GS(r, alpha)
    return np.abs(wave_func)**2

def Hyd_local(r, alpha=2):
    if r <= 0:
        raise ValueError("Position value 'r' must be greater than zero.")
    return -1/r - (alpha/2) * (alpha - (2/r))

def Hyd_VMC(r, step, samples=10000, alpha=2):
    position_saved = []
    energy_saved = []
    for n in range(samples):
        r_new = r + np.random.uniform(-step, step)
        if r_new <= 0:
            continue
        P_old = Hyd_GSPDF(r, alpha)
        P_new = Hyd_GSPDF(r_new, alpha)
        ratio = P_new / (P_old + 1e-10)
        s = np.random.rand()
        if ratio > s:
            r = r_new
        position_saved.append(r)
        energy_saved.append(Hyd_local(r, alpha))
    return position_saved, energy_saved

def psi_analytic(r, alpha):
    return (alpha**(3/2)) * np.exp(-alpha * r) / np.sqrt(np.pi)
```

---

### Running the Code
To run the code, follow these steps:
1. Clone the repository:
```bash
git clone https://github.com/SuvamT0071/VMC.git
cd VMC
```

2. Run the Jupyter Notebook or Python script:
```bash
jupyter notebook "QMC QHO.ipynb"
jupyter notebook "QMC HYD.ipynb"
jupyter notebook "QMC HEL.ipynb"
```

3. Use the provided functions to test wavefunctions, visualize results, or optimize alpha.

---

## Mathematical Background

**Choosing a Trial Wavefunction**

In the Variational Monte Carlo method, we approximate the ground state wavefunction with a trial function containing a free parameter (alpha). For accurate energy estimation, the chosen trial function should capture key properties of the system such as symmetry, boundary conditions, and decay behavior.

**For example:**

**Hydrogen Atom Trial Function:** Ψ(r) = α r e^(-α r)

**Quantum Harmonic Oscillator Trial Function:** Ψ(x) = α^(1/2) / π^(1/4) e^(-α^2 x^2 / 2)

**Local Energy Calculation**

The local energy is defined as:

E_local(r) = (HΨ(r)) / Ψ(r)

Where H is the Hamiltonian operator. 

**For the hydrogen atom:**

E_local(r) = -1/r - (α/2) * (α - (2/r))

**For the Quantum Harmonic Oscillator:**

E_local(x) = α^2 + x^2(1 - α^4)

Minimizing the mean local energy leads to the optimal parameter α that best approximates the true ground state energy.
---

## Sample Plot
To visualize the results, you can plot the optimized wavefunction and compare it with the analytic solution.

```python
import matplotlib.pyplot as plt

x_vals = np.linspace(0, 5, 500)
optimal_alpha = 1.5  # Example optimized value
plt.plot(x_vals, psi_analytic(x_vals, optimal_alpha), label="Analytic Solution")
plt.legend()
plt.title("Hydrogen Atom Ground State Wavefunction")
plt.xlabel("r")
plt.ylabel(r"$\Psi(r)$")
plt.grid(True)
plt.show()
```

![image](https://github.com/user-attachments/assets/83e4ab58-614f-4cc5-98e2-551d218fb6c1)
![image](https://github.com/user-attachments/assets/017d20e4-0bee-44fc-8c7b-dc93c35c1b14)
![image](https://github.com/user-attachments/assets/04a39937-6028-4d5c-9d2d-e37b40ea2168)
![image](https://github.com/user-attachments/assets/00982c05-4ab1-4896-adaf-6dc6742e751f)
![image](https://github.com/user-attachments/assets/328b5d98-9ff9-48cc-9dd5-b36619b7dd02)


---

## This was made by:
- **Suvam Tripathy, A MSc Physics student at IIT Madras as a part of a mini project. (2023-2025)**

---

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute the code.

