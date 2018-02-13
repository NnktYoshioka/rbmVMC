import numpy as np
from MetropolisSamplingTools import *
from RBM_1dJ1J2 import *
from Hamiltonian_1dJ1J2 import *

# Outline of main program
class VMC:
    def __init__(
        self,

        eta = 0.1, # learning rate
        n_batch = 100, # (#sample)  for Gibbs Sampling
        n_vis = 16, # system size
        alpha = 1, # ratio of #hidden to #visible
        sigma = 0.01, # width, or maximum value, for random initial parameters

        l2reg = 0, # L2 regularization
        lamb = 0, # regularization factor of S matrix
        lamb2 = 0, # regularization factor of S matrix

        momentum = 0., # momentum
        dropout_p = 0., # dropout
        threshold = 0.2, # threshold for S-matrix
        threshold2 = 2,


        J1 = 1, # NN AF Heisenberg
        J2 = 1, # NNN AF Heisenberg
#        hz = 0,
#        hx = 0,
        Delta = 1, # XXZ Anisotropy
        deggsvec_exact = None
    ):
        self.v_chain = None
        self.chainsamples = ComplexChainSamples()
        self.rbm = None
        self.eta = eta
        self.l2reg = l2reg
        self.sigma = sigma
        self.lamb = lamb
        self.lamb2 = lamb2

        self.momentum = momentum
        self.dropout_p = dropout_p
        self.threshold = threshold

        self.J1 = J1
        self.J2 = J2
#        self.hz = hz
#        self.hx = hx
        self.Delta = Delta

        self.n_batch = n_batch
        self.n_vis = n_vis
        self.alpha = alpha
        self.n_hid = n_vis * alpha

        self.n_param_Wf =  self.alpha * n_vis
        self.n_param_a = self.n_vis
        self.n_param_b = self.alpha
        self.n_param = self.n_param_Wf + self.n_param_a + self.n_param_b

        self.grads = None
        self.eloc = None
        self.elocpersite = None

        self.deggsvec_exact = deggsvec_exact
        self.overlaps = []

        self.acceptances = None

    # Initialize RBM
    def SetRBM(self):
        self.rbm = ComplexRBM(n_vis = self.n_vis, alpha = self.alpha, sigma = self.sigma, Delta = self.Delta, J1 = self.J1, J2 = self.J2,
                          allspinconfig = BinaryList(np.arange(2**self.n_vis), self.n_vis))
        self.rbm.SetW()


    # v_chain is allocated as self.v_chain
    def ChainSampling(self):
        chainsamples = ComplexChainSamples()
        chainsamples.SetSampler(self.rbm)
        chainsamples.Reset()
        self.v_chain = chainsamples.V_Chain(self.n_batch)
        self.acceptances = chainsamples.acceptances


    # Choose S^(-1) * G or G as the gradient
    def Gradient(self):
        self.grads = self.SinvG(self.v_chain)


    # The covariant matrix is multiplied to the force G
    def SinvG(self,v_chain):
        g_mean = self.G_means(self.v_chain)
        grad_all = np.linalg.solve(self.Smat(self.v_chain), g_mean)
        return grad_all


    # Skk' = <Dk* Dk'> - <Dk><Dk'>
    def Smat(self, v_chain):
        # Compute <Dk * Dk'>
        DkDk_chain = np.array([self.rbm.DkDk_mat(v_chain[j]) for j in range(self.n_batch)])
        DkDkmean = DkDk_chain.mean(axis = 0)

        # Compute <Dk> * <Dk'>
        Dkmean = self.Der_all_vec_mean(v_chain)
        Dkmean_Dkmean = np.kron(Dkmean.conjugate(), Dkmean).reshape(self.n_param, self.n_param)

        # Take the difference.
        # lamb, lamb2 is the regularization term
        Smat = DkDkmean - Dkmean_Dkmean
        Smat = Smat + self.lamb * np.diag(np.diag(Smat)) + self.lamb2 * np.diag(np.ones(self.n_param)) * np.max(np.diag(Smat))

#        Smat = np.diag(np.ones(self.n_param))
        return Smat


    # Compute <Dk>
    def Der_all_vec_mean(self,v_chain):
        Der_all_vec_chain = np.zeros((self.n_batch, self.n_param), dtype = np.complex)

        for j in range(self.n_batch):
            Der_all_vec_chain[j]= self.rbm.Der_all_vec(v_chain[j])

        Der_all_vec_mean = Der_all_vec_chain.mean(axis = 0)
        return Der_all_vec_mean


    # Local energy chain for given v_chains
    def Eloc_stats(self, v_chain):
        Eloc_chain = np.array([self.rbm.Eloc(v_chain[j]) for j in range(self.n_batch)])
        Eloc_mean = Eloc_chain.mean()
        Eloc_std = Eloc_chain.std()
        return Eloc_chain, Eloc_mean, Eloc_std



    # Compute the mean of Variational Derivatives
    def G_means(self, v_chain):
        Eloc_chain, Eloc_mean, Eloc_std = self.Eloc_stats(v_chain)
        self.eloc = Eloc_mean
        self.elocpersite = Eloc_mean/(1.0 * self.n_vis)

        Der_Wf_chain = np.zeros((self.n_batch, self.n_vis, self.alpha), dtype = np.complex)
        Der_a_chain = np.zeros((self.n_batch, self.n_vis), dtype = np.complex)
        Der_bf_chain = np.zeros((self.n_batch, self.alpha), dtype = np.complex)


        for j in range(self.n_batch):
            Der_Wf_chain[j], Der_a_chain[j], Der_bf_chain[j] = self.rbm.Der_all(v_chain[j])


        GWf_chain = np.array([   (Eloc_chain[j] - Eloc_mean) * np.conjugate(Der_Wf_chain[j])
                        for j in range(self.n_batch)])
        Ga_chain = np.array([   (Eloc_chain[j] - Eloc_mean) * np.conjugate(Der_a_chain[j])
                            for j in range(self.n_batch)])
        Gbf_chain = np.array([   (Eloc_chain[j] - Eloc_mean) * np.conjugate(Der_bf_chain[j])
                            for j in range(self.n_batch)])

        return np.append(GWf_chain.mean(axis = 0).flatten(), np.append(Ga_chain.mean(axis = 0), Gbf_chain.mean(axis = 0)))

    # Update the parameters of RBM
    def Update(self):
        params = self.rbm.GetParams()
        newparams = self.UpdateParams(self.grads, params) # grads = [G_W_mean, G_a_mean, G_b_mean]
        self.rbm.SetParams(newparams)

        if self.n_vis <= 10:
            self.overlaps.append(self.DegGSoverlap())

    # Compute new parameters
    def UpdateParams(self, grads, params):
        Wf, a, bf = params
        # Scale the update if it is too big.
        if max(np.abs(grads)) > self.threshold:
            grads = grads * self.threshold/max(np.abs(grads))

        gWf,  ga, gbf = [grads[:self.n_param_Wf].reshape(self.n_vis, self.alpha ), grads[self.n_param_Wf: self.n_param_Wf + self.n_param_a],
                       grads[self.n_param_Wf + self.n_param_a:]]


        # Dropout the update of element by specific propability.
        randomdist_Wf, randomdist_a, randomdist_bf = [np.random.rand(self.n_vis, self.alpha ),  np.random.rand(self.n_vis),
                                                    np.random.rand(self. alpha)]
        dropout_Wf, dropout_a, dropout_bf = [np.heaviside(randomdist_Wf - self.dropout_p, 0), np.heaviside( randomdist_a - self.dropout_p,0),
                                          np.heaviside(randomdist_bf - self.dropout_p, 0)]

        # Gradient step
        Wf = (1- self.momentum) * Wf - self.eta * (gWf  + np.conjugate(self.l2reg * Wf)) * dropout_Wf
        a = (1- self.momentum) * a - self.eta * (ga  + np.conjugate(self.l2reg * a)) * dropout_a
        bf = (1- self.momentum) * bf - self.eta * (gbf + np.conjugate(self.l2reg * bf)) * dropout_bf

        newparams = [Wf,a,bf]

        return newparams

    def PrintStats(self):
        print "Eloc/site = ",self.eloc/(1.0*self.n_vis)


    # Normalize a vector
    def Normalize(self, v):
        v_unnorm = v/(np.max(np.abs(v)) *1.0)
        return v_unnorm/(np.linalg.norm(v_unnorm))


    # Take the overlap of two vectors
    def Overlap(self, v1, v2):
        v1_normed = self.Normalize(v1)
        v2_normed = self.Normalize(v2)
        signed_overlap = np.dot(v1_normed.conjugate(), v2_normed)
        return np.abs(signed_overlap)

    # Compute the overlap between exact solution.
    def DegGSoverlap(self):
#        assert (self.deggsvec_exact != None).any(), "Prepare the exact gs wavefunction to compare with ansatz."
        gsvec_raw = self.rbm.GSvec_raw()
        deggsvec_exact = self.deggsvec_exact
        return [self.Overlap(gsvec_raw, gsvec_exact) for gsvec_exact in deggsvec_exact]

# "Oracle" for one step.
    def run(self):
        self.ChainSampling()
#        print "acceptance average = ",np.array(self.acceptances).mean()
        self.Gradient()
        self.Update()
#        self.PrintStats()
