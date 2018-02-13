from Hamiltonian_1dHeisenberg import HeisenHam
import scipy
import numpy as np

def GSvec(leng, PBCflag, hz = 0, hx = 0):
    testHam = HeisenHam(Leng = leng, PBCflag = PBCflag, J = 1, Delta = 1, hz = hz, hx = hx)
    #print testHam

    vals,vecs = scipy.linalg.eigh(testHam)

    return vecs[:,0]

def EnPerSite_exact(leng, PBCflag, hz = 0, hx = 0):
    testHam = HeisenHam(Leng = leng, PBCflag = PBCflag, J = 1, Delta = 1, hz = hz, hx = hx)
    #print testHam

    vals,vecs = scipy.linalg.eigh(testHam)

    return vals[0]
