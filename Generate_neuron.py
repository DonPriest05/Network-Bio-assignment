from Generate_soma import Soma
from Generate_dendrite import Dendrite 

class Neuron:

    def __init__(self, num_dendrites, params_soma, params_dendrite, V_soma_init, V_dend_init, n_init, h_init, k_init, l_init, q_init, r_init, s_init):
        self.V_soma_init = V_soma_init
        self.V_dend_init = V_dend_init
        self.n_init = n_init
        self.h_init = h_init
        self.k_init = k_init
        self.l_init = l_init
        self.q_init = q_init
        self.r_init = r_init
        self.s_init = s_init
        self.soma = Soma(params_soma, V_soma_init, n_init, h_init, k_init, l_init, q_init)
        self.dendrites = []
        for _ in range(num_dendrites):
            self.dendrites.append(Dendrite(params_dendrite, V_dend_init, r_init, s_init))


