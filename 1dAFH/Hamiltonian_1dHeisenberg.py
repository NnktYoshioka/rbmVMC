import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import scipy
from scipy.linalg import kron


# Spin operators defined below.
s=1/2.0 # total spin
m =int(2*s + 1)

# Ladder operators
Sp = np.zeros((m,m))
for i in range(1,m):
    Sp[i-1,i] = np.sqrt(i*(m-i)) # sqrt(j(j+1)- m(m+1)) for (S,Sz) = (j,m)
Sm = np.zeros((m,m))
for i in range(0,m-1):
    Sm[i+1,i] = np.sqrt((m-i-1)*(i+1)) # sqrt(j(j+1)+ m(m+1)) for (S,Sz) = (j,m)

# Spin op.
Sz = np.zeros((m,m))
for i in range(m):
    Sz[i,i] = (m-1)*0.5 - i
Sx = (Sp+Sm)/2.0
Sy = (Sp-Sm)/(2.0j)
S0 = np.diag(np.ones(m))

Ss = scipy.zeros((4,m,m), dtype = np.complex)
Ss = scipy.array([S0,Sx,Sy,Sz])

def Skron(a,i,leng):
    """
    a: The element of the spin operator. 0,1,2,3 <--> 0, x,y,z
    i: The location of S[a]. Note "i"th in python order.
    Returns S0 \otimes ... \otimes Sa \otimes ... \otimes S0
    where S0 is m*m identity matrix.
    Note Ss = sig/2.0 for spin 1/2.
    """
    assert i<=leng-1, 'Keep i<=leng.'

    matarray = scipy.array([S0 for l in range(leng)], dtype = np.complex)
    matarray[i] = Ss[a]
    init = matarray[0]
    Skron = scipy.copy(init)
    for i in range(1,leng):
        Skron = kron(Skron,matarray[i])

    return Skron

# Hamiltonian for Heisenberg chain
def HeisenHam(Leng, PBCflag = 0, J = 1, Delta = 1, hz = 0, hx = 0):
    """
    Leng: The Length of the Heisenberg chain
    PBCflag: OBC(=0) or PBC(=1)
    """

#    hranx = field*ranvecx # in [-field, field]
#    hranz = field*ranvecz
#    hrany = 0 * np.zeros(len(ranvecx))

    # Random field part
#    hxranham =  np.sum([hranx[i] * Skron(1,i,Leng) for i in range(Leng)],axis = 0)
#    hyranham =  np.sum([hrany[i] * Skron(2,i,Leng) for i in range(Leng)],axis = 0)
#    hzranham =  np.sum([hranz[i] * Skron(3,i,Leng) for i in range(Leng)],axis = 0)

    # Uniform field part
    hzfield = hz * np.ones(Leng)
    hxfield = hx * np.ones(Leng)

    hzham = np.sum([hzfield[i] * 2 * Skron(3, i, Leng) for i in range(Leng)], axis = 0) # 2 due to sigma(mu) = 2 * Skron
    hxham = np.sum([hxfield[i] * 2 * Skron(1, i, Leng) for i in range(Leng)], axis = 0) # 2 due to sigma(mu) = 2 * Skron

    # Heisenberg interaction part
    # We perform gauge transformation for every odd site:
    # S_i^x -> - S_i^x,
    # S_i^y -> - S_i^y.
    HeisenHam = 4 * J * np.sum([ - Skron(1,i,Leng).dot(Skron(1,i+1,Leng)) -
                            Skron(2,i,Leng).dot(Skron(2,i+1,Leng)) +
                            Delta * Skron(3,i,Leng).dot(Skron(3,i+1,Leng))
                            for i in range(Leng-1)], axis = 0) # 4 due to sigma(mu) = 2 * Skron
    if PBCflag == 1:
        HeisenHam = HeisenHam + 4 * J *(- Skron(1,0,Leng).dot(Skron(1,Leng-1,Leng)) -
                            Skron(2,0,Leng).dot(Skron(2,Leng-1,Leng)) +
                            Delta * Skron(3,0,Leng).dot(Skron(3,Leng-1,Leng))) # 4 due to sigma(mu) = 2 * Skron

#    return hxranham + hyranham + hzranham + HeisenHam
    return HeisenHam + hzham + hxham



# returns a binary list with its length unnormalized
def PreBinaryList(dec_list):
    binarylist = map(lambda x:format(x, 'b'), dec_list)
    return map(lambda x: [int(y) for y in list(x)], binarylist)

#returns a binarylist with its length n_digits
def FullList(list_element, n_digits):
    assert len(list_element) <= n_digits, "n_digit is too small to allocate the data."
    n_diff = n_digits - len(list_element)
    full_list = np.append(np.zeros(n_diff), np.array(list_element))
    full_list = (-1) * (2 * np.array(full_list) -1)
    return full_list

# Translate dec_list in binary notation with length = n_digits.
def BinaryList(dec_list, n_digits):
    prebinarylist = PreBinaryList(dec_list)
    return map(lambda x: FullList(x, n_digits), prebinarylist)
