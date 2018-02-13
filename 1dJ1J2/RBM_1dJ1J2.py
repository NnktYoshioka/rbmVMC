import numpy as np
import scipy
import matplotlib.pyplot as plt

# Note that the RBM here takes v,h = {-1, +1} as variable.
# Correspondingly, the activation is Logistic(+- 2 \theta(\gamma)) for v-->h (h-->v),
# where \theta = v \dot W + b, \gamma = W \dot h + a.

# Calculate the local energy in a way introduced by Carleo&Troyer (2017).
# In order to calculate the local energy,
# Eloc(x) = \sum_x' H_{x,x'} \psi(x')/\psi(x),
# all we need is the information on connected matrix elements.
# Thus, we find
#    1. 1D J1J2 Hamiltonian expressed by spin basis along z-axis,
#    2. the ratio \psi(x')/\psi(x). Given as (2.23) in Carleo's tutorial,
# is neccessary to compute the Eloc for each configuration.

# After all, we take the average over the samples to update the parameters.
# Other than Eloc, we also need to compute D_a, D_b, D_W, which can be obtained very easily from \psi.

# Definition of v_connected, mel, wfratio explicitly depends on Hamiltonian;
# needs extra care to apply to other systems.

class ComplexRBM(object):
    def __init__(
        self,

        # Parameter of RBM
        n_vis = 16, # system size
        alpha = 1, # Ratio of #hidden to #visible

        W = None, # Weight matrix w/ translational symmetry(TRS)
        Wf = None,
        b = None, # magnetic field for hidden neurons w/TRS
        bf = None,

        a = None, # magnetic field for the visible spin. No TRS.
        sigma = 0.01, # width, or maximum value, for random initial parameters

        # Parameter of Hamiltonian
        J1 = 1, # Nearest-neighbor AF Heisenberg
        J2 = 1, # Next nearest-neighbor AF Heisenberg
        hz = 0., # magnetic field along z
        hx = 0, # magnetic field along x
        Delta = 1, # XXZ anisotropy. Isotropic if Delta = 1.
        allspinconfig = None
                ):

        self.input = input
        self.n_vis = n_vis
        self.n_hid = n_vis * alpha
        self.alpha = alpha
        self.sigma = sigma


        # Initialize the RBM parameters
        if Wf == None:
            Wwid = self.sigma
            RanAbs = Wwid * (2 * np.random.rand(self.n_vis, self.alpha) -1)
            RanPha = RanAbs * np.pi/2.0
            Wf = RanAbs + 1j * RanPha # W_f.shape = (n_vis, alpha)

        if a == None:
            awid = self.sigma
            RanAbs = awid * (2 * np.random.rand(n_vis) -1)
            RanPha =  RanAbs * np.pi/2.0

            a = RanAbs + 1j * RanPha # len(a) = n_vis

        if bf == None:
            bwid = self.sigma
            RanAbs = bwid * (2 * np.random.rand(self.alpha) -1)
            RanPha =  RanAbs * np.pi/2.0
            bf = RanAbs + 1j * RanPha #len(b) = alpha * n_vis

        self.Wf = Wf
        self.a = a
        self.bf = bf

        self.J1 = J1
        self.J2 = J2
        self.hz = hz
        self.hx = hx
        self.Delta = Delta

        self.gsvec = None
        self.allspinconfig = allspinconfig

    # Cyclic transformation of vector.
    def CycleRow(self, vec, n):
        return np.append(vec[n:], vec[:n])

    # Cyclic transformation of vector. Inverse direction.
    def InverseCycleRow(self, vec, n):
        return np.append(vec[-n:], vec[:-n])

    # Update W and b based on Wf and bf.
    # W consists Wf in a cyclic form.
    def SetW(self):
        assert len(self.Wf.shape) ==2, "Wf.shape should be (n_vis, alpha)."
        assert len(self.bf) == self.alpha, "bf.shape should be (alpha)."
        self.W = np.array([self.InverseCycleRow(self.Wf[:,i],j)  for i in range(self.alpha) for j in range(self.n_vis)]).transpose()
        self.b = np.kron(self.bf, np.ones(self.n_vis))

    # Returns Wf, a, bf
    def GetParams(self):
        """
        Returns the parameters, i.e., a, b, and W of RBM.
        Output: [Wf, a, bf]
        """
        return [self.Wf, self.a, self.bf]

    # Update Wf, W a, bf, b.
    def SetParams(self,params):
        """
        Input: [Wf, a, bf]
        """
        nWf, na, nbf = params

        self.Wf = nWf
        self.a = na
        self.bf = nbf

        self.SetW()

    # Logistic(+-2*theta[j]) is probability for hidden spin j to be +-1.
    def Theta(self, v0_sample):
        return np.dot( v0_sample, self.W) + self.b

    def ProbDens(self, v0_sample):
        alpha0 = np.exp( np.sum(self.a * v0_sample))
        alpha1 = np.prod(  np.cosh(self.theta(v0_sample))) # abbreviate (2 *) cosh
        return alpha0 * alpha1

    # Matrix elements of Hamiltonian
    def mel(self,v0_sample):
        mel =  np.ones(2 * self.n_vis+1, dtype = np.complex)

        mel[0] =   np.sum( [self.J1 * v0_sample[l] * v0_sample[(l+1)%self.n_vis]
                                        + self.J2 * v0_sample[l] * v0_sample[(l+2)%self.n_vis]
                            for l in range(self.n_vis)] )# diagonal for Periodic chain
        mel[1:self.n_vis + 1] =  + self.J1 *  np.array([(1 - v0_sample[l] * v0_sample[ (l+1)%self.n_vis ]) for l in range(self.n_vis)])
        mel[self.n_vis + 1:] =  + self.J2 *  np.array([(1 - v0_sample[l] * v0_sample[ (l+2)%self.n_vis ]) for l in range(self.n_vis)])

        return mel

    # wfratio[0] =1 by definition.
    # wfratio[k+1] is the ratio of wave function between spin configuration \sigma
    # and \sigma'_k, in which k- and (k+1)-th spin is flipped by NN Heisenberg int.
    # wfratio[n_vis + k + 1] is the ratio between
    # \sigma''_{k} in which k- and (k+2)-th spin is flipped by NNN Heisenberg int.

    # Since multiplication of Cosh(x) is numerically unstable,
    # we compute exp(log(Cosh(x))). Note log1p(x) = log(1+x), which is more stable.

    def wfratio(self,v0_sample):
        wfratio = np.zeros(2 * self.n_vis + 1, dtype = np.complex)
        wfratio[0] = 1

        alpha0 = -2 * self.a * v0_sample
        alpha0next = np.append(alpha0[1:], alpha0[0])
        alpha = self.Alpha(v0_sample)

        gamma0 = alpha0
        gamma0_2next = np.append(gamma0[2:], alpha0[0:2])
        gamma = self.Gamma(v0_sample)

        wfratio[1: self.n_vis + 1] = np.exp((alpha0 + alpha0next + alpha )/1.0) # /1.0 corresponding to \Psi = F_RBM
        wfratio[self.n_vis + 1: ] = np.exp((gamma0 + gamma0_2next + gamma )/1.0) # /1.0 corresponding to \Psi = F_RBM
        return wfratio

    def Alpha(self, v0_sample):
        alpha1 = self.Alpha1(v0_sample)
        alpha2 = self.Alpha2(v0_sample)
        alpha3 = self.Alpha3(v0_sample)

        alpha1next = np.append(alpha1[1:,:], [alpha1[0,:]], axis = 0)

        return np.sum(alpha1 + alpha1next +  alpha2 + alpha3,axis = 1)

    def Gamma(self, v0_sample):
        gamma1 = self.Gamma1(v0_sample)
        gamma2 = self.Gamma2(v0_sample)
        gamma3 = self.Gamma3(v0_sample)

        gamma1_2next = np.append(gamma1[2:,:], gamma1[:2,:], axis = 0)

        return np.sum(gamma1 + gamma1_2next +  gamma2 + gamma3,axis = 1)

    def Alpha1(self,v0_sample):
        n_k = self.W.shape[0]
        n_j = self.W.shape[1]

        v0sample_kj = np.kron(v0_sample, np.ones(n_j)).reshape(n_k,n_j)
        return 2 * v0sample_kj * self.W

    def Alpha2(self,v0_sample):
        n_k = self.W.shape[0]
        n_j = self.W.shape[1]
        thetas = self.Theta(v0_sample)

        # faster than for**2, but still slower than c++.
        v0sample_kj = np.kron(v0_sample, np.ones(n_j)).reshape(n_k,n_j)
        thetas_kj = np.kron(np.ones(n_k), thetas).reshape(n_k,n_j)

        alpha_kj = (-4) * v0sample_kj * self.W
        alpha_next_kj = np.append(alpha_kj[1:, :], [alpha_kj[0,:]], axis = 0)

        PreExp = 2 * thetas_kj  + alpha_kj + alpha_next_kj
        Exp = np.exp(PreExp)
        return np.log1p( Exp )


    def Alpha3(self,v0_sample):

        n_k = self.W.shape[0]
        n_j = self.W.shape[1]
        thetas = self.Theta(v0_sample)

        thetas_kj = np.kron(np.ones(n_k), thetas).reshape(n_k,n_j)
        PreExp = 2 * thetas_kj

        Exp = np.exp(PreExp)
        return - np.log1p( Exp ) # alpha3 = - log(1 + exp(PreExp))

    def Gamma1(self,v0_sample):
        return self.Alpha1(v0_sample)

    def Gamma2(self, v0_sample):
        n_k = self.W.shape[0]
        n_j = self.W.shape[1]
        thetas = self.Theta(v0_sample)

        # faster than for**2, but still slower than c++.
        v0sample_kj = np.kron(v0_sample, np.ones(n_j)).reshape(n_k,n_j)
        thetas_kj = np.kron(np.ones(n_k), thetas).reshape(n_k,n_j)

        gamma_kj = (-4) * v0sample_kj * self.W
        gamma_2next_kj = np.append(gamma_kj[2:, :], gamma_kj[:2,:], axis = 0)

        PreExp = 2 * thetas_kj  + gamma_kj + gamma_2next_kj
        Exp = np.exp(PreExp)
        return np.log1p( Exp )

    def Gamma3(self, v0_sample):
        return self.Alpha3(v0_sample)

    # Local Energy
    def Eloc(self,v0_sample):
        return np.sum(self.mel(v0_sample) * self.wfratio(v0_sample))

    # Variational derivatives
    def Der_all(self,v0):
        Der_a = v0/1.0 # /2.0 or /1.0 according to real or complex
        Der_bfs = np.tanh(self.Theta(v0))/1.0 # /2.0 or /1.0 according to real or complex
        Der_bf = np.array([np.sum(Der_bfs[i * self.n_vis: (i+1) * self.n_vis]) for i in range(self.alpha)])

        cyclev = self.CycleV(v0)
        tanhmat = np.tanh(self.Theta(v0)).reshape(self.alpha, self.n_vis)

        Der_Wf = np.dot(tanhmat, cyclev)/1.0 # /2.0 or /1.0 according to real or complex
        Der_Wf = Der_Wf.transpose()
        return Der_Wf, Der_a, Der_bf

    # Allocate vectorized Der_all.
    def Der_all_vec(self,v0):
        Der_a = v0/1.0 # /2.0 or /1.0 according to real or complex
        Der_bfs = np.tanh(self.Theta(v0))/1.0 # /2.0 or /1.0 according to real or complex
        Der_bf = np.array([np.sum(Der_bfs[i * self.n_vis: (i+1) * self.n_vis]) for i in range(self.alpha)])

        cyclev = self.CycleV(v0)
        tanhmat = np.tanh(self.Theta(v0)).reshape(self.alpha, self.n_vis)

        Der_Wf = np.dot(tanhmat, cyclev)/1.0 # /2.0 or /1.0 according to real or complex
        Der_Wf = Der_Wf.transpose()
        return np.append(Der_Wf.flatten(), np.append(Der_a, Der_bf))


    def DkDk_mat(self,v0):
        Dk = self.Der_all_vec(v0)
        n_param = len(Dk)
        return np.kron(Dk.conjugate(), Dk).reshape(n_param, n_param)

    def CycleV(self,v0):
        return np.array([ self.CycleRow(v0, i) for i in range(self.n_vis)])


    # The following gives the wave function by RBM as \Psi = F_RBM
    def GSamp_raw(self,v):
        W_RBM = self.W
        a_RBM = self.a
        b_RBM = self.b
        thetas = self.Theta(v)

        phi1 = np.sum(a_RBM * v)
        phi2 = - np.sum(thetas)
        phi3 = np.sum( np.log1p(np.exp(2 * thetas)))

        return np.exp((phi1 + phi2 + phi3)/1.0) # /2.0 or /1.0 according to real or complex

    def GSvec_raw(self):
        assert self.allspinconfig != None, "Define the basis of the wavefunction by inputting the allspinconfig."
        return np.array(map(lambda x: self.GSamp_raw(x), self.allspinconfig))
