# Quantum many-body ground-state optimization by VMC of RBM
Optimizing GS energy of quantum many-body system by VMC with RBM as trial function,
based on [Carleo & Troyer, Science 355, 602 (2017)](https://arxiv.org/abs/1606.02318).
The description of each codes are given as follows.
## main.ipynb
The main execution file of the scheme.
## Auxiliary files
**ExactWaveFunc_(model).py**:
Exact wave function and GS energy of finite-size system.

**Hamiltonian_(model).py**:
 Defining the Hamiltonian of finite-size system.
 
**RBM_(model).py**:
 Class file of RBM for the model. Derivatives for RBM parameters, local energy, and unsampled (analytical) wave function amplitudes are computed. 
 
**RealSamplingTools.py**:
 Gibbs sampler from the RBM by the alternate sampling method.
 
**MetropolisSamplingTools.py**:
 Sampler from the RBM by the ordinary single-flip algorithm based on Metropolis-Hastings rule.
 
**Real/ComplexWFtools.py**:
 Auxiliary functions to compare the exact and numerically optimized wave function.
 
**VMC_(model).py**:
 VMC scheme. Sampling the spin configuration, calculating the gradients, and updating RBM parameters.
 
**plottools_(model).py**:
 Save, load, and name the result.

