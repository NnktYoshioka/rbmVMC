import numpy as np


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

def AllSpinConfig(n_digits):
    return BinaryList(np.arange(2**n_digits), n_digits)

def theta(RBM, v):
    W_RBM = RBM.W
    a_RBM = RBM.a
    b_RBM = RBM.b
    n_k = W_RBM.shape[0]
    n_j = W_RBM.shape[1]

    v_kj = np.kron(v, np.ones(n_j)).reshape(n_k,n_j)
    return np.sum(v_kj * W_RBM, axis = 0) + b_RBM

# The following gives the wave function by RBM when \Psi^2 = F_RBM
def GSvec_raw(RBM,v):
    W_RBM = RBM.W
    a_RBM = RBM.a
    b_RBM = RBM.b
    thetas = theta(RBM, v)

    phi1 = np.sum(a_RBM * v)
    phi2 = - np.sum(thetas)
    phi3 = np.sum( np.log1p(np.exp(2 * thetas)))

    return np.exp((phi1 + phi2 + phi3)/1.0) # Real or Complex

# Normalize a vector
def Normalize(v):
    v_unnorm = v/(np.max(np.abs(v)) *1.0)
    return v_unnorm/(np.linalg.norm(v_unnorm))

# Take the overlap of two vectors
def Overlap(v1, v2):
    v1_normed = Normalize(v1)
    v2_normed = Normalize(v2)
    signed_overlap = np.dot(v1_normed.conjugate(), v2_normed)
    return np.abs(signed_overlap)
