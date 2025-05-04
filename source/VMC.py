import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from numba import njit, prange
import warnings

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

@njit(parallel=True)
def lightatoms_GS(r, alpha=2.0, beta=1.0, ansatz=1):
    """
    Ground state trial wavefunction for the light atoms.

    Parameters:
    - r: 2x3 array, positions of 2 electrons in 3D.
    - alpha: variational parameter
    - beta: variational parameter
    - ansatz: choose 1 or 2 for different trial wavefunctions.

    Returns:
    - psi_T: Trial wavefunction evaluated at r.
    """
    r_comb = 0.0
    for i in prange(r.shape[0]):
        r_electron = np.sqrt(np.sum(r[i]**2))
        r_comb += r_electron

    if ansatz == 1:
        psi_T = np.exp(-alpha * r_comb)

    elif ansatz == 2:
        r12 = np.sqrt(np.sum((r[0] - r[1])**2))
        psi_T = np.exp(-alpha * r_comb) * (1 + beta*r12)

    elif ansatz == 3:
        r12 = np.sqrt(np.sum((r[0] - r[1])**2))
        psi_T = np.exp(-alpha * r_comb) * np.exp(r12 / (2 * (1 + beta * r12)))

    else:
        psi_T = 0.0

    return psi_T

@njit(fastmath=True)
def lightatoms_GSPDF(r, alpha=2.0, beta=1.0, ansatz = 1):
    """
    This function defines the ground state trial
    Probability density function for light atoms

    Parameters:
    - r: Takes in a 2X3 matrix for position of the 2 electrons
    - alpha: Optimizing parameter (Default = 2)
    - beta: Optimizing parameter (Default = 1)
    - ansatz: choose 1 or 2 for different wave function
    Returns:
    The trial PDF for the Helium atom
    """

    wave_func = lightatoms_GS(r, alpha, beta, ansatz)
    return np.abs(wave_func)**2

@njit(parallel=True)
def light_loc_en(r, Z, alpha=2.0, beta=1.0, ansatz = 1):
    """
    This function defines the ground state
    Local energy for light atoms

    Parameters:
    - r: Takes in a 2X3 matrix for position of the 2 electrons
    - alpha: Optimizing parameter (Default = 2)
    - beta: Optimizing parameter (Default = 2)
    - Z: Atomic number or proton number
    - ansatz: choose a trial wavefunction
    Returns:
    The Local energy for a given light atom
    """
    GS_zero = lightatoms_GS(r, alpha, beta, ansatz)
    step = 1e-5
    KE = 0.0

    #Performing central difference method for calculation of laplacian
    for i in prange(r.shape[0]):
        for j in range(r.shape[1]):
            #forward jump
            r_plus = np.copy(r)
            r_plus[i][j] += step
            GS_plus = lightatoms_GS(r_plus, alpha, beta, ansatz)
            #backward jump
            r_minus = np.copy(r)
            r_minus[i][j] -= step
            GS_minus = lightatoms_GS(r_minus, alpha, beta, ansatz)

            KE += (GS_plus + GS_minus - 2.0 * GS_zero) / step**2

    kinetic = -0.5 * KE / GS_zero

    #Accounting for the electron-nucleus interaction
    PE1 = 0.0
    for i in prange(r.shape[0]):
        r_electron = np.sqrt(np.sum(r[i]**2))
        PE1 += -Z / r_electron

    #Accounting for the electron-electron repulsion
    r12 = np.sqrt(np.sum((r[0] - r[1])**2))
    if r12 != 0:
        PE2 = 1.0/r12
    else:
        PE2 = 0

    potential = PE1 + PE2
    local_energy = kinetic + potential
    return local_energy

@njit(fastmath=True)
def lightatoms_VMC(r, Z, step, samples=10000, alpha=2.0, beta = 1.0, ansatz = 1):
    """
    This function performs Variational Monte Carlo
    method for light atoms

    Parameters:
    - r: Takes in a 2X3 matrix for position of the 2 electrons
    - step: step size for the movement of MH samples
    - samples: enter the number of VMC sweeps you would like to perform
               (By default: 10000)
    - alpha: Optimizing parameter (Default = 2)
    - beta: Optimizing parameter (Default = 1)
    - ansatz: choose a trial wavefunction

    Returns:
    The saved positions and saved energies respectively
    """
    position_saved = []
    energy_saved = []
    r_current = r.copy()

    for n in prange(samples):
        q = np.random.rand(r_current.shape[0], r_current.shape[1])
        r_new = r_current + step * (q - 0.5) #for symmetry purpose
        P_old = lightatoms_GSPDF(r_current, alpha, beta, ansatz)
        P_new = lightatoms_GSPDF(r_new, alpha, beta, ansatz)
        ratio = P_new / (P_old + 1e-10)

        s = np.random.rand()
        if ratio > s:
            r_current = r_new

        position_saved.append(r_current.copy())
        energy_saved.append(light_loc_en(r_current, Z, alpha, beta, ansatz))

    return position_saved, energy_saved

def lightatoms_alpha_opt(alpha_list, r, Z, step, samples=10000, ansatz=1):
    """
    This function optimizes alpha by performing
    Variational Monte Carlo method on light atoms

    Parameters:
    - alpha_list: Takes in a list of alpha values for which VMC will be performed
    - r: Takes in a 2X3 matrix for position of the 2 electrons
    - step: step size for the movement of MH samples
    - samples: enter the number of VMC sweeps you would like to perform
               (By default: 10000)

    Returns:
    The saved positions and saved energies respectively
    """
    saved_energies = []
    variance = []
    mean_energies = []

    for a in tqdm(alpha_list, unit='alpha', desc='Optimizing alpha'):
        _, energies = lightatoms_VMC(r, Z, step, samples, a, ansatz)
        mean_e = np.mean(energies)
        saved_energies.append(mean_e)
        variance.append(np.var(energies))
        mean_energies.append(mean_e)

    optimal_alpha = alpha_list[np.argmin(saved_energies)]
    return saved_energies, optimal_alpha, variance, mean_energies

#optimizing beta for ansatz 2
def lightatoms_beta_opt(beta_list, optimal_alpha, r, Z, step, samples=10000, ansatz=1):
    """
    This function optimizes alpha by performing
    Variational Monte Carlo method on light atoms

    Parameters:
    - alpha_list: Takes in a list of alpha values for which VMC will be performed
    - r: Takes in a 2X3 matrix for position of the 2 electrons
    - step: step size for the movement of MH samples
    - samples: enter the number of VMC sweeps you would like to perform
               (By default: 10000)

    Returns:
    The saved positions and saved energies respectively
    """
    saved_energies = []
    variance = []
    mean_energies = []

    for b in tqdm(beta_list, unit='beta', desc='Optimizing beta'):
        _, energies = lightatoms_VMC(r, Z, step, samples, alpha=optimal_alpha, beta=b, ansatz=ansatz)
        mean_e = np.mean(energies)
        saved_energies.append(mean_e)
        variance.append(np.var(energies))
        mean_energies.append(mean_e)

    optimal_beta = beta_list[np.argmin(saved_energies)]
    return saved_energies, optimal_beta, variance, mean_energies
