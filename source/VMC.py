import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import warnings

# Defining the wavefunction for the Quantum Harmonic Oscillator

def QHO_GS1D(x, alpha=1):
    """
    Ground state wavefunction of the 1D Quantum Harmonic Oscillator.

    Parameters:
    - x (float or int): Position of the particle in the 1D QHO.
    - alpha (float or int): Variational parameter (default = 1).

    Returns:
    - float: Value of the wavefunction at position x for a given alpha.
    """
    return np.sqrt(alpha) * np.exp(- (x**2) * (alpha**2) / 2) / (np.pi ** (1/4))


def GS_PDF(x, alpha=2):
    """
    Probability Density Function (PDF) of the ground state wavefunction of the QHO.

    Parameters:
    - x (float or int): Position of the particle in the 1D QHO.
    - alpha (float or int): Variational parameter (default = 2).

    Returns:
    - float: Probability density at position x.
    """
    wave_func = QHO_GS1D(x, alpha)
    return np.abs(wave_func)**2


def local_energy(x, alpha=2):
    """
    Local energy for the ground state wavefunction of the QHO.

    Parameters:
    - x (float or int): Position of the particle in the 1D QHO.
    - alpha (float or int): Variational parameter (default = 2).

    Returns:
    - float: Local energy at position x.
    """
    return alpha**2 + (x**2) * (1 - alpha**4)


def energy_var(alpha=2):
    """
    Variance in the local energy for the ground state wavefunction of the QHO.

    Parameters:
    - alpha (float or int): Variational parameter (default = 2).

    Returns:
    - float: Variance in the local energy.
    """
    return ((alpha**4 - 1)**2) / (2 * alpha**4)


def VMC_sweeps(x, step, samples=10000, alpha=2):
    """
    Performs Variational Monte Carlo (VMC) sweeps for the QHO.

    Parameters:
    - x (float or int): Initial position.
    - step (float): Step size for position updates.
    - samples (int): Number of VMC sweeps (default = 10000).
    - alpha (float or int): Variational parameter (default = 2).

    Returns:
    - list: Sampled particle positions.
    - list: Local energy values for the sampled positions.
    """
    position_saved, energy_saved = [], []

    for n in range(samples):
        r = np.random.rand()
        x_new = x + (r - 0.5) * step

        P_old = GS_PDF(x, alpha)
        P_new = GS_PDF(x_new, alpha)
        ratio = P_new / (P_old + 1e-10)

        if ratio > np.random.rand():
            x = x_new

        position_saved.append(x)
        energy_saved.append(local_energy(x, alpha))

    return position_saved, energy_saved


def alpha_optimizer(alpha_list, x, step, samples=10000):
    """
    Optimizes the value of alpha for the QHO wavefunction.

    Parameters:
    - alpha_list (list): List of alpha values for optimization.
    - x (float or int): Initial position for the particle.
    - step (float): Step size for position updates.
    - samples (int): Number of VMC sweeps (default = 10000).

    Returns:
    - list: Mean energy values for each alpha in the list.
    - float: Optimal alpha value that minimizes the energy.
    """
    saved_energies = []
    for a in tqdm(alpha_list, unit='alpha', desc='Optimizing alpha'):
        _, energies = VMC_sweeps(x, step, samples, a)
        saved_energies.append(np.mean(energies))

    optimal_alpha = alpha_list[np.argmin(saved_energies)]

    return saved_energies, optimal_alpha


def Hyd_GS(r, alpha=2):
    """
    Ground state wavefunction of the Hydrogen atom.

    Parameters:
    - r (float or int): Position of the electron in the Hydrogen atom.
    - alpha (float or int): Variational parameter (default = 2).

    Returns:
    - float: Value of the wavefunction at position r.
    """
    return alpha * r * np.exp(-alpha * r)


def Hyd_GSPDF(r, alpha=1):
    """
    Probability Density Function (PDF) for the ground state of the Hydrogen atom.

    Parameters:
    - r (float or int): Position of the electron in the Hydrogen atom.
    - alpha (float or int): Variational parameter (default = 1).

    Returns:
    - float: Probability density at position r.
    """
    wave_func = Hyd_GS(r, alpha)
    return np.abs(wave_func)**2


def Hyd_local(r, alpha=2):
    """
    Local energy function for the ground state of the Hydrogen atom.

    Parameters:
    - r (float or int): Position of the electron in the Hydrogen atom.
    - alpha (float or int): Variational parameter (default = 2).

    Returns:
    - float: Local energy value at position r.
    """
    return -1 / r - (alpha / 2) * (alpha - (2 / r))


def Hyd_VMC(r, step, samples=10000, alpha=2):
    """
    Performs Variational Monte Carlo (VMC) sweeps for the Hydrogen atom.

    Parameters:
    - r (float or int): Initial position.
    - step (float): Step size for position updates.
    - samples (int): Number of VMC sweeps (default = 10000).
    - alpha (float or int): Variational parameter (default = 2).

    Returns:
    - list: Sampled electron positions.
    - list: Local energy values for the sampled positions.
    """
    position_saved, energy_saved = [], []

    for n in range(samples):
        r_new = r + np.random.uniform(-step, step)
        if r_new <= 0:
            continue

        P_old = Hyd_GSPDF(r, alpha)
        P_new = Hyd_GSPDF(r_new, alpha)
        ratio = P_new / (P_old + 1e-10)

        if ratio > np.random.rand():
            r = r_new

        position_saved.append(r)
        energy_saved.append(Hyd_local(r, alpha))

    return position_saved, energy_saved


def Hyd_alpha_opt(alpha_list, r, step, samples=10000):
    """
    Optimizes the value of alpha for the Hydrogen atom wavefunction.

    Parameters:
    - alpha_list (list): List of alpha values for optimization.
    - r (float or int): Initial position for the particle.
    - step (float): Step size for position updates.
    - samples (int): Number of VMC sweeps (default = 10000).

    Returns:
    - list: Mean energy values for each alpha in the list.
    - float: Optimal alpha value that minimizes the energy.
    """
    saved_energies = []
    for a in tqdm(alpha_list, unit='alpha', desc='Optimizing alpha'):
        _, energies = Hyd_VMC(r, step, samples, a)
        saved_energies.append(np.mean(energies))

    optimal_alpha = alpha_list[np.argmin(saved_energies)]

    return saved_energies, optimal_alpha
