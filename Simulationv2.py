from Generate_neuron import Neuron
from Generate_network import generate_network
from plotting_functions import plotting
import numpy as np
import matplotlib.pyplot as plt

seed = 1
plots = plotting(seed)
num_nodes = 20
k = 13
p = 0.1
inhib_ratio = 0
inhib_str = 0

params_soma = []
params_dendrite = []
I = 0.5
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

print(n_init)
# Parameters soma
params_soma = [
    0.5,    # g_Ca conductance (mS/cm²)
    0.1,    # g_h conductance (mS/cm²)
    120.0,  # E_Ca Calcium reversal potential (mV)
    -43.0,  # E_h reversal potential (mV)
    0.02,   # g_L Leak conductance (mS/cm²)
    -65.0,  # E_L Leak reversal potential (mV)
    50.0,   # g_Na Sodium conductance (mS/cm²)
    55.0,   # E_Na Sodium reversal potential (mV)
    10.0,   # g_K Potassium conductance (mS/cm²)
    -75.0,  # E_K Potassium reversal potential (mV)
    0.1,    # g_sd Coupling conductance between soma and dendrites (mS/cm²)
    1.0,    # Cm Membrane capacitance (µF/cm²)
    0.1     # p
]

params_dendrite = [
    1.0,    # g_Ca conductance (mS/cm²)
    120.0,  # E_Ca Calcium reversal potential (mV)
    0.02,   # g_L Leak conductance (mS/cm²)
    -65.0,  # E_L Leak reversal potential (mV)
    20.0,   # g_K Potassium conductance (mS/cm²)
    -75.0,  # E_K Potassium reversal potential (mV)
    0.1,    # g_sd Coupling conductance between soma and dendrites (mS/cm²)
    1.0,    # Cm Membrane capacitance (µF/cm²)
    0.1     # p
]

network = generate_network(seed)
W, G  = network.get_network(num_nodes,k,p,inhib_ratio, inhib_str)
plots.plot_network(G)
plots.visualize_weight_matrix(W)


neurons = []
for i in range(num_nodes):
    number_of_dendrites = int(np.sum(W[i]))
    connections = []
    for j in range(len(W[i])):
        if W[i][j] == 1:
            connections.append(j)
    neurons.append(Neuron(number_of_dendrites, connections, params_soma, params_dendrite, V_rest,V_rest,n_init,h_init,k_init,l_init,q_init,r_init,s_init))


g_GJ =0.01
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
            I_GJ = 0
            if len(coupled_neurons_index) != 0:
                for j in range(len(coupled_neurons_index)):
                    
                    if num in neurons[coupled_neurons_index[j]].connections:
                        coupled_dendrite = neurons[coupled_neurons_index[j]].connections.index(num)
                        V_dendrite_coupled = V_all_dend[coupled_neurons_index[j]][-1][coupled_dendrite]
                        I_GJ = I_GJ + (V_dendrite_coupled - V_soma)*g_GJ


            if t > T_pre:
                V_soma,n,h,k,l,q  = neuron.soma.RK_soma(V_soma, V_dend, n, h, k, l, q, I + np.random.normal(0, abs(I) / 4) + I_GJ, dt)
            else:
                V_soma,n,h,k,l,q  = neuron.soma.RK_soma(V_soma, V_dend, n, h, k, l, q, I_GJ, dt)

            for i,dendrite in enumerate(neuron.dendrites):
                V_dend[i], r[i], s[i] = dendrite.RK_dend(V_soma, V_dend[i], r[i], s[i], dt)

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




dt = 0.01
V_all_soma, V_all_dend  = simulate(100,100,dt)




def plot_soma_voltages(V_all_soma, T_pre, dt):
    num_neurons = len(V_all_soma)
    fig, axes = plt.subplots(num_neurons, 1, figsize=(10, 2 * num_neurons), sharex=True)
    time_to_track = int(T_pre/dt)
    for i in range(num_neurons):
        axes[i].plot(V_all_soma[i][time_to_track:])
        axes[i].set_title(f'Neuron {i+1} Soma Voltage')
        axes[i].set_ylabel('Voltage (mV)')
    
    axes[-1].set_xlabel('Time (ms)')
    plt.tight_layout()
    plt.show(block = False)

plot_soma_voltages(V_all_soma, 100, dt)