from Generate_soma import Soma
from Generate_dendrite import Dendrite 
import numpy as np

class Neuron:

    def __init__(self, num_dendrites, connections,length_connection, size_soma, radius_dendrites, params_soma, params_dendrite, V_soma_init, V_dend_init, n_init, h_init, k_init, l_init, q_init, r_init, s_init):
        self.V_soma_init = V_soma_init 
        self.V_dend_init = V_dend_init 
        self.n_init = n_init
        self.h_init = h_init
        self.k_init = k_init
        self.l_init = l_init
        self.q_init = q_init
        self.r_init = r_init
        self.s_init = s_init
        self.soma = Soma(params_soma, V_soma_init, size_soma, n_init, h_init, k_init, l_init, q_init)
        self.dendrites = []
        self.connections = connections
        
        for i in range(num_dendrites):
            self.dendrites.append(Dendrite(connections[i],length_connection[i], radius_dendrites[i], params_dendrite, V_dend_init, r_init, s_init))


