from Generate_neuron import Neuron
from Generate_network import generate_network
from plotting_functions import plotting
import numpy as np
import matplotlib.pyplot as plt
import random

seed = 1
random.seed(seed)
plots = plotting(seed)
num_nodes = 20
num_clusters = 1
inter_prob = 0.01
P = np.ones([num_clusters ,num_clusters])* inter_prob
intra_prob = 0.2
np.fill_diagonal(P, intra_prob)


params_soma = []
params_dendrite = []
I = -0.5e-3
V_rest = -57

def alpha_n(V): 
    return 0.016 * (V + 52) / (1 - np.exp(-(V + 52) / 5))
def beta_n(V): 
    return 0.25 * np.exp(-(V + 57) / 40)
def alpha_h(V): 
    return 0.128 * np.exp(-(V + 50) / 18)
def beta_h(V): 
    return 4 / (1 + np.exp(-(V + 27) / 5))
def alpha_k(V): 
    return 0.02 * (V + 50) / (1 - np.exp(-(V + 50) / 10))
def beta_k(V): 
    return 0.1 * np.exp(-(V + 60) / 20)
def alpha_l(V): 
    return 0.05 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
def beta_l(V): 
    return 0.25 * np.exp(-(V + 65) / 40)
def alpha_q(V): 
    return 0.03 * (V + 45) / (1 - np.exp(-(V + 45) / 10))
def beta_q(V): 
    return 0.2 * np.exp(-(V + 55) / 30)
def alpha_r(V): 
    return 0.02 * (V + 20) / (1 - np.exp(-(V + 20) / 9))
def beta_r(V): 
    return 0.05 * (V + 8) / (np.exp((V + 8) / 9) - 1)
def alpha_s(V): 
    return 0.016 * (V + 52) / (1 - np.exp(-(V + 52) / 5))
def beta_s(V): 
    return 0.25 * np.exp(-(V + 57) / 40)

def steady_state_value(alpha, beta):
    return alpha / (alpha + beta)

n_init = steady_state_value(alpha_n(V_rest), beta_n(V_rest))
h_init = steady_state_value(alpha_h(V_rest), beta_h(V_rest))
k_init = steady_state_value(alpha_k(V_rest), beta_k(V_rest))
l_init = steady_state_value(alpha_l(V_rest), beta_l(V_rest))
q_init = steady_state_value(alpha_q(V_rest), beta_q(V_rest))
r_init = steady_state_value(alpha_r(V_rest), beta_r(V_rest))
s_init = steady_state_value(alpha_s(V_rest), beta_s(V_rest))
print(r_init)
print(s_init)


# Parameters soma
params_soma = [
    0.5,    # g_Ca conductance (mS/cm²)
    0.05,   # g_h conductance (mS/cm²)
    120.0,  # E_Ca Calcium reversal potential (mV)
    -43.0,  # E_h reversal potential (mV)
    0.1,    # g_L Leak conductance (mS/cm²)
    -65.0,  # E_L Leak reversal potential (mV)
    20.0,   # g_Na Sodium conductance (mS/cm²)
    55.0,   # E_Na Sodium reversal potential (mV)
    5.0,    # g_K Potassium conductance (mS/cm²)
    -75.0,  # E_K Potassium reversal potential (mV)
    1,    # g_sd Coupling conductance between soma and dendrites (mS/cm²)
    1.0,    # Cm Membrane capacitance (µF/cm²)
    0.2     # p (scaling factor)
]
params_dendrite = [
    0.1,    # g_Ca conductance (mS/cm²)
    120.0,  # E_Ca Calcium reversal potential (mV)
    0.05,   # g_L Leak conductance (mS/cm²)
    -65.0,  # E_L Leak reversal potential (mV)
    5.0,    # g_K Potassium conductance (mS/cm²)
    -75.0,  # E_K Potassium reversal potential (mV)
    1,    # g_sd Coupling conductance between soma and dendrites (mS/cm²)
    1.0,    # Cm Membrane capacitance (µF/cm²)
    0.2     # p (scaling factor)
]

network = generate_network(seed)

W,G = network.get_network(num_nodes, num_clusters, P)
#W, G  = network.create_interconnected_network(int(num_nodes/num_sub_networks),k,p,inhib_ratio, inhib_str, num_sub_networks, interconnect_ratio)

plots.plot_network(G,W)
#plots.plot_combined_subnetworks(G, num_nodes, num_sub_networks)
plots.visualize_weight_matrix(W)
plots.plot_degree(G)


edge_lengths = network.calculate_edge_lengths(G)


neurons = []
for i in range(num_nodes):
    size_soma = random.uniform(1e-5, 1e-4) #area in cm2
    number_of_dendrites = int(np.sum(W[i]))
    connections = []
    length_connection = []
    radius_dendrites = []
    for j in range(len(W[i])):
        if W[i][j] == 1:
            radius_dendrites.append(random.uniform(0.1e-5, 2e-5)) #radius in cm
            connections.append(j)
            length_connection.append(edge_lengths[(i,j)])
    neurons.append(Neuron(number_of_dendrites, connections, length_connection, size_soma, radius_dendrites, params_soma, params_dendrite, V_rest,V_rest,n_init,h_init,k_init,l_init,q_init,r_init,s_init))


g_GJ_dend_dend = 0.1
g_GJ_dend_soma = 0.1
def simulate(T,T_pre,dt):

    
    V_all_soma =  [[] for _ in range(len(neurons))]
    V_all_dend =  [[] for _ in range(len(neurons))]
    n_all = [[] for _ in range(len(neurons))]
    h_all = [[] for _ in range(len(neurons))]
    k_all = [[] for _ in range(len(neurons))]
    l_all = [[] for _ in range(len(neurons))] 
    q_all = [[] for _ in range(len(neurons))]
    r_all = [[] for _ in range(len(neurons))] 
    s_all = [[] for _ in range(len(neurons))]

    for num,neuron in enumerate(neurons):

        n_all[num].append(neuron.soma.n_init)
        h_all[num].append(neuron.soma.h_init)
        k_all[num].append(neuron.soma.k_init)
        l_all[num].append(neuron.soma.l_init)
        q_all[num].append(neuron.soma.q_init)
        V_all_soma[num].append(neuron.soma.V_soma_init)
        V_dend = []
        r = []
        s = []
        for dendrite in neuron.dendrites:
            V_dend.append(dendrite.V_dend_init)
            r.append(dendrite.r_init)
            s.append(dendrite.s_init)
        V_all_dend[num].append(V_dend)
        r_all[num].append(r)
        s_all[num].append(s)
    t = 0

    while t < T + T_pre:
        for num,neuron in enumerate(neurons):
            V_soma = V_all_soma[num][-1]
            V_dend = V_all_dend[num][-1]
            n = n_all[num][-1]
            h = h_all[num][-1]
            k = k_all[num][-1]
            l = l_all[num][-1]
            q = q_all[num][-1]
            r = r_all[num][-1]
            s = s_all[num][-1]

            coupled_neurons_index = np.where(W[:,num] == 1)[0]
            I_GJ_dend_dend = 0
            I_GJ_dend_soma = 0
            if len(coupled_neurons_index) != 0:
                for j in range(len(coupled_neurons_index)):
                    
                    if num in neurons[coupled_neurons_index[j]].connections:
                        coupled_dendrite = neurons[coupled_neurons_index[j]].connections.index(num)
                        V_dendrite_coupled = V_all_dend[coupled_neurons_index[j]][-1][coupled_dendrite]
                        if len(neurons[num].dendrites) != 0:
                            random.seed((num, j))
                            random_dendrite_ind = random.randint(0, len(neurons[num].dendrites) - 1) 
                            I_GJ_dend_dend = I_GJ_dend_dend+ (V_all_dend[num][-1][random_dendrite_ind] - V_dendrite_coupled)*g_GJ_dend_dend * neurons[coupled_neurons_index[j]].dendrites[coupled_dendrite].radius_dendrite**2*np.pi
                        else:
                            I_GJ_dend_soma = I_GJ_dend_soma + (V_soma- V_dendrite_coupled)*g_GJ_dend_soma * neurons[coupled_neurons_index[j]].dendrites[coupled_dendrite].radius_dendrite**2*np.pi

            radii = [dendrite.radius_dendrite for dendrite in neuron.dendrites]
            num_dends = len(neuron.dendrites)
            if T > T_pre:
                V_soma,n,h,k,l,q  = neuron.soma.RK_soma(V_soma, V_dend, n, h, k, l, q, I + abs(I/10) * np.random.randn()   , dt, radii, num_dends, I_GJ_dend_soma)
            else:
                V_soma,n,h,k,l,q  = neuron.soma.RK_soma(V_soma, V_dend, n, h, k, l, q, I + abs(I/10) * np.random.randn() , dt, radii,num_dends,  0)

            
            for i,dendrite in enumerate(neuron.dendrites):
                if i == random_dendrite_ind:
                    V_dend[i], r[i], s[i]= dendrite.RK_dend(V_soma, V_dend[i], r[i], s[i], dt, num_dends, I_GJ_dend_dend)
                else:
                    V_dend[i], r[i], s[i] = dendrite.RK_dend(V_soma, V_dend[i], r[i], s[i], dt, num_dends)


            V_all_soma[num].append(V_soma)
            V_all_dend[num].append(V_dend)
            n_all[num].append(n)
            h_all[num].append(h)
            k_all[num].append(k)
            l_all[num].append(l)
            q_all[num].append(q)
            r_all[num].append(r)
            s_all[num].append(s)
            
        t+= dt
    return V_all_soma, V_all_dend 



T_pre = 0
T = 10
dt = 0.01
V_all_soma, V_all_dend  = simulate(T,T_pre, dt)
plots.plot_traces(V_all_soma, T_pre, dt)

