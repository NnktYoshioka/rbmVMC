import numpy as np

class ComplexChainSamples:
    def __init__(self, v_init = None, RBMSampler = None):
        self.rbm = None
        self.v_init = None
        self.RBMSampler = None
        self.n_vis = None
        self.acceptances = None

    def SetSampler(self, RBM):
        self.RBMSampler = ComplexSampler()
        self.RBMSampler.rbm = RBM
        self.n_vis = len(RBM.a)

    def Setv_init(self, v_sample):
        self.v_init = v_sample

    def Reset(self):
        self.v_init = 2 * np.random.randint(0,2,self.n_vis) -1

    def V_Chain(self, n_batch):
        v_seed = self.v_init
        v_chain = np.zeros((n_batch, len(v_seed)))
        v_chain[0] = v_seed

        self.acceptances = []

        # Gibbs sampling and store the config.
        for j in range(n_batch):
            v_seed, acceptance_ = self.RBMSampler.gibbs_vnew(v_seed)
            v_chain[j] = v_seed
            self.acceptances.append(acceptance_)

        return v_chain

class ComplexSampler:
    def __init__(self, rbm = None):
        self.rbm = None

    def SetRBM(self, RBM):
        self.rbm = RBM

    def gibbs_vnew(self, v0sample):
        # Flip a random spin and accept according to Metropolis-Hastings.
        k = np.random.randint(0, self.rbm.n_vis)
        vnew = np.copy(v0sample)

        # Flipped state
        vref = np.copy(v0sample)
        vref[k] = (-1) * vref[k]

        # Compute the acceptance and update the config.
        acceptance =self.Acceptance(v0sample, vref)  ##
        flip_ref = np.random.random_sample()
        if acceptance > flip_ref:
            vnew = vref

        return vnew, acceptance

    # Computes the ratio of probability between two cpin config.
    def ProbRatio(self, v0sample, vref):
        Amp1 = self.rbm.GSamp_raw(v0sample)
        Amp2 = self.rbm.GSamp_raw(vref)
        return np.abs(Amp2/Amp1)**2

    def Acceptance(self,v0sample, vref):
        return np.min([1, self.ProbRatio(v0sample, vref)])
