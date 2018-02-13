import numpy as np

class RealChainSamples:
    def __init__(self, v_init = None, RBMSampler = None):
        self.rbm = None
        self.v_init = None
        self.RBMSampler = None
        self.n_vis = None

    def SetSampler(self, RBM):
        self.RBMSampler = RealSampler()
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

        # Gibbs sampling and store the config.
        for j in range(n_batch):
            v_seed = self.RBMSampler.gibbs_vhv(v_seed)[5]
            v_chain[j] = v_seed

        return v_chain

# Sampler "samples", i.e., generate new spin configuration
# based on the Metropolis-Hastings method.
# In the case of RBM, this is alternate sampling for visible and hidden layer,
 # whose update proposal is always accepted.

class RealSampler:
    def __init__(self, rbm = None):
        self.rbm = None

    def SetRBM(self, RBM):
        self.rbm = RBM

    # The activation function
    def sigmoid(self,z):
        return 1/(np.exp(-z) + 1)

    def propup(self,vis):
        """
        Posterior probabilities P(h|v) given by sigmoid function.
        pre_activation is probably not neccessary for pure-python coding.
        """
        pre_activation = 2 * (np.dot(vis, self.rbm.W) + self.rbm.b)
        return pre_activation, self.sigmoid(pre_activation)

    def sample_h_given_v(self,vis):
        """
        Propose h based on P(h|v) computed by propup.
        Input: v
        Output: [presigmoid, h1_mean, h1_sample]
        """
        pre_sigmoid_h1, h1_mean = self.propup(vis)
        h1_ref = np.random.rand(len(h1_mean))
        h1_sample = np.sign(h1_mean - h1_ref) # Proposed configuration
        return pre_sigmoid_h1, h1_mean, h1_sample

    def propdown(self, hid):
        """
        Posterior probabilities P(v|h) given by sigmoid function.
        pre_activation is probably not neccessary for pure-python coding.
        """
        pre_activation = 2 * (np.dot(self.rbm.W, hid) + self.rbm.a)
        return pre_activation, self.sigmoid(pre_activation)

    def sample_v_given_h(self, hid):
        """
        Propose v based on P(v|h) computed by propdown.
        Input: h
        Output: [presigmoid, v1_mean, v1_sample]
        """
        pre_sigmoid_v1, v1_mean = self.propdown(hid)
        v1_ref = np.random.rand(len(v1_mean))
        v1_sample = np.sign(v1_mean - v1_ref) # Proposed configuration
        return pre_sigmoid_v1, v1_mean, v1_sample



    def gibbs_vhv(self, v0_sample):
        """
        Input: visible variables (dim = n_vis)
        Output: [pre_sigmoid_h1, h1_mean, h1_sample,
               pre_sigmoid_v1, v1_mean, v1_sample]
        """
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
               pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        """
        Input: hidden variables (dim = n_hid)
        Output: [pre_sigmoid_v1, v1_mean, v1_sample,
               pre_sigmoid_h1, h1_mean, h1_sample]
        """
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
               pre_sigmoid_h1, h1_mean, h1_sample]
