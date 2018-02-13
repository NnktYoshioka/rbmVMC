from Hamiltonian_1dJ1J2 import J1J2Ham
import scipy
import numpy as np

def GSvecs(leng, PBCflag, J1=1, J2 = 1, hz = 0, hx = 0):
    testHam = J1J2Ham(Leng = leng, PBCflag = PBCflag, J1 = J1, J2 = J2, hz = hz, hx = hx)
    #print testHam

    vals,vecs = scipy.linalg.eigh(testHam)

    return [vecs[:,0], vecs[:,1]]

def EnPerSite_exact(leng, PBCflag, J1 = 1, J2 = 1, hz = 0, hx = 0):
    testHam = J1J2Ham(Leng = leng, PBCflag = PBCflag, J1 = J1, J2 = J2, hz = hz, hx = hx)
    #print testHam

    vals, vecs = scipy.linalg.eigh(testHam)

    return vals[0]
