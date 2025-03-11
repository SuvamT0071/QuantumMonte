import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import warnings

#defining the wavefunction
def QHO_GS1D(x, alpha=1):
    """
    This function defines the first wavefunction of the quantum harmonic oscillator
    
    Parameters:
    - x: The position of the particle in the 1D QHO
    - alpha: Optimizing parameter, initiate with some value
             (By default = 1)
             
    Returns:
    
    This function returns you the value of the wavefunction for a particular position and optimizing parameter alpha
    """
    
    if not isinstance(x,(float,int)):
        raise ValueError("Value other than float or integer was called")
    if not isinstance(alpha, (float, int)):
        raise ValueError("Value other than float or integer was called")
    
    return np.sqrt(alpha) * np.exp(- (x**2) * (alpha**2) / 2) / (np.pi ** (1/4))

#defining the probability density function
def GS_PDF(x,alpha=2):
    """
    This function defines the PDF of the 
    ground state wavefunction of QHO
    
    Parameters:
    - x: The position of the particle in the 1D QHO
    - alpha: Optimizing parameter, initiate with some value
             (By default = 1)
    
    Returns:
    
    This function returns you the value of the PDF for a particular position and optimizing parameter alpha
    corresponding to the ground state of the Quantum Harmonic Oscillator
    """
    
    if not isinstance(x,(float,int)):
        raise ValueError("Value other than float or integer was called")
    if not isinstance(alpha, (float, int)):
        raise ValueError("Value other than float or integer was called")
    
    wave_func = QHO_GS1D(x, alpha)
    return np.abs(wave_func)**2

#defining the local energy function
def local_energy(x, alpha=2):
    """
    This function defines the local energy of the 
    ground state wavefunction of QHO.

    Parameters:
    - x: The position of the particle in the 1D QHO
    - alpha: Optimizing parameter, initiate with some value
             (By default = 1)

    Returns:
    - Local energy of the ground state of Quantum harmonic oscillator system at position x
    """
    
    if not isinstance(x,(float,int)):
        raise ValueError("Value other than float or integer was called")
    if not isinstance(alpha, (float, int)):
        raise ValueError("Value other than float or integer was called")
    
    return alpha**2 + (x**2) * (1 - alpha**4)

#defining the variance of the energy function
def energy_var(alpha=2):
    """
    This function defines the variance in 
    the local energy of the ground state wavefunction of QHO
    
    Parameters:
    - alpha: Optimizing parameter, initiate with some value
             (By default = 1)
    
    Returns:
    - Variance in the Local energy of the ground state of Quantum harmonic oscillator system at position x
    """
    if not isinstance(alpha, (float, int)):
        raise ValueError("Value other than float or integer was called")
    
    return ((alpha**4 - 1)**2) / (2 * alpha**4) 

def VMC_sweeps(x, step, samples = 10000,alpha=2):
    """
    This funciton performs Variational Monte Carlo sweeps
    
    Parameters:
    - x: initial position
    - step: step size
    - alpha_list: in takes a list of alpha values for optimization
    - samples: no. of VMC sweeps to be performed
               (By default: 10000)
    """
    position_saved = []
    energy_saved = []
    
    for n in range(samples):
        r = np.random.rand()
        x_new = x + (r-0.5)*step
        
        P_old = GS_PDF(x, alpha)
        P_new = GS_PDF(x_new, alpha)
        ratio = P_new/(P_old + 1e-10)
        
        s = np.random.rand()
        
        if ratio > s:
            x = x_new  
        
        position_saved.append(x)
        energy_saved.append(local_energy(x, alpha))
    
    return position_saved, energy_saved

#defining a function to optimize alpha
def alpha_optimizer(alpha_list, x, step, samples=10000):
    """
    This function will optimize the value of alpha for the wavefunction
    
    Parameters:
    - alpha_list: takes in a list of alpha for optimization
    - x: give an initial value of the position for the particle
    - step: give a step size for movement
    - samples: enter the number of VMC sweeps you would like to perform
               (By default: 10000)
    """
    saved_energies = []
    for a in tqdm(alpha_list, unit='alpha', desc='Optimizing alpha'):
        positions, energies = VMC_sweeps(x, step, samples, a) #energies gives a list of energy values for a particular round, so you need to mean it up
        saved_energies.append(np.mean(energies))

    optimal_alpha = alpha_list[np.argmin(saved_energies)]
    
    return saved_energies, optimal_alpha

#defining the trial wavefunction for the ground state of hydrogen atom
def Hyd_GS(r, alpha = 2):
    """
    This function defines the Ground state wavefunction of Hydrogen atom    
    Parameters:
    - r: The position of the electron in the hydrogen atom
    - alpha: Optimizing parameter, initiate with some value
            (By default = 2)
             
    Returns:
    
    This function returns you the value of the wavefunction for a particular position and optimizing parameter alpha
    """
    
    if not isinstance(r,(float,int)):
        raise ValueError("Value other than float or integer was called")
    if not isinstance(alpha, (float, int)):
        raise ValueError("Value other than float or integer was called")
    
    return alpha * r * np.exp(-alpha*r)

def Hyd_GSPDF(r, alpha = 1):
    """
    This function defines the PDF of the 
    ground state wavefunction of Hydrogen atom
    
    Parameters:
    - r: The position of the particle in the Hydrogen atom 
    - alpha: Optimizing parameter, initiate with some value
             (By default = 1)
    
    Returns:
    
    This function returns you the value of the PDF for a particular position and optimizing parameter alpha
    corresponding to the ground state of the Hydrogen atom
    """
    
    if not isinstance(r,(float,int)):
        raise ValueError("Value other than float or integer was called")
    if not isinstance(alpha, (float, int)):
        raise ValueError("Value other than float or integer was called")
    
    wave_func = Hyd_GS(r, alpha)
    return np.abs(wave_func)**2

#defining the local energy function
def Hyd_local(r, alpha=2):
    """
    This function defines the local energy of the 
    ground state wavefunction of Hydrogen atom.

    Parameters:
    - r: The position of the particle in the Hydrogen atom
    - alpha: Optimizing parameter, initiate with some value
             (By default = 2)

    Returns:
    - Local energy of the ground state of Quantum harmonic oscillator system at position x
    """
    
    if not isinstance(r,(float,int)):
        raise ValueError("Value other than float or integer was called")
    if not isinstance(alpha, (float, int)):
        raise ValueError("Value other than float or integer was called")
    
    return - 1/r - (alpha/2) * (alpha - (2/r))

def Hyd_VMC(r, step, samples = 10000,alpha=2):
    """
    This funciton performs Variational Monte Carlo sweeps
    
    Parameters:
    - r: initial position
    - step: step size
    - alpha_list: in takes a list of alpha values for optimization
    - samples: no. of VMC sweeps to be performed
               (By default: 10000)
    """
    position_saved = []
    energy_saved = []
    
    for n in range(samples):
        q = np.random.uniform(-step, step)
        r_new = r + q
        if r_new <= 0:
            continue
        
        P_old = Hyd_GSPDF(r, alpha)
        P_new = Hyd_GSPDF(r_new, alpha)
        ratio = P_new/(P_old + 1e-10)
        
        s = np.random.rand()
        
        if ratio > s:
            r = r_new  
        
        position_saved.append(r)
        energy_saved.append(Hyd_local(r, alpha))
    
    return position_saved, energy_saved

#defining a function to optimize alpha
def Hyd_alpha_opt(alpha_list, r, step, samples=10000):
    """
    This function will optimize the value of alpha for the wavefunction
    
    Parameters:
    - alpha_list: takes in a list of alpha for optimization
    - r: give an initial value of the position for the particle
    - step: give a step size for movement
    - samples: enter the number of VMC sweeps you would like to perform
               (By default: 10000)
    """
    saved_energies = []
    for a in tqdm(alpha_list, unit='alpha', desc='Optimizing alpha'):
        positions, energies = Hyd_VMC(r, step, samples, a) #energies gives a list of energy values for a particular round, so you need to mean it up
        saved_energies.append(np.mean(energies))

    optimal_alpha = alpha_list[np.argmin(saved_energies)]
    
    return saved_energies, optimal_alpha

