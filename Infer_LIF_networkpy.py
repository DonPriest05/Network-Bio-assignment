#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Justin Priest 

"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from pyrqa.analysis_type import Classic
from pyrqa.computation import RQAComputation, RPComputation
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius
from pyrqa.settings import Settings
from pyrqa.time_series import TimeSeries
from pyrqa.neighbourhood import Unthresholded
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, Normalizer
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import jpype
from jpype import *

scaler = StandardScaler()
normal_transform = Normalizer()
seed = 1

# %% Set up java environment

## Download infodynamics.jar from the github repo and change directory to this location
## Run only once after that comment out again
jarLocation = "B:/JP/Documents/Network Bio/JIDT/jidt-master/infodynamics.jar"  # replace with your path to the infodynamics.jar file
jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)


# %% Parameters

g_L = .05  # leak channel conductance
C_m = 1  # membrane capacitance
E_L = -70  # reversal potential leak
V_th = -55  # spike threshold
V_reset = -70  # reset value
V_peak = 0  # spike peak
tau_syn = 20E-3  # synaptic transfer time constant in ms
E_syn_exc = 0  # synaptic reversal potential for excitatory neurons
E_syn_inh = -70  # synaptic reversal potential for inhibitory neurons
g_syn_exc = .5  # Excitatory synapse conductance
g_syn_inh = 1  # inhibitory synapse conductance
dt = 0.1  # time steps


# %%
class Node():

    def __init__(self, Iapp, W, ID, nodes, spikes, S):

        """
        Initializes a node object/neuron according to the LIF eqs
        ----------

        Iapp: float
            Input current
        
        W: 2d array with ints
            Contains the weight/adjecency matrix, values are -1,0 or 1
            
        ID: int
            Gives the ID of the node in question w.r.t the entire network
            
        nodes: list of lists of floats
            Contains the voltage traces for all nodes
            
        spikes list of list of ints
            Contains the spike trains for all nodes
            
        S: int
            Gives the current time point in the simulation
            
        -------
            
        """
        self.Vt, self.spike = self.LIF(Iapp, W, ID, nodes, spikes, S)

    def LIF(self, Iapp, W, ID, nodes, spikes, S):

        """
        Computes LIF model equations
        ----------

        Iapp: float
            Input current
        
        W: 2d array with ints
            Contains the weight/adjecency matrix, values are -1,0 or 1
            
        ID: int
            Gives the ID of the node in question w.r.t the entire network
            
        nodes: list of lists of floats
            Contains the voltage traces for all nodes
            
        spikes list of list of ints
            Contains the spike trains for all nodes
            
        S: int
            Gives the current time point in the simulation
            
        Returns
        -------
        
        V:  float
            Returns the voltage for the node object
            
        spike: int
            Returns whether a spike occured or not
            
        """

        V = nodes[ID][-1]
        I = Iapp + self.I_syn(W, ID, spikes, V, E_L, S)
        spike = 0

        if spikes[ID][-1] == 1:
            V = V_reset
        else:

            V = self.RK(V, g_L, E_L, I, C_m, dt)
            if V >= V_th:
                spike = 1
                V = V_peak

        return V, spike

    def RK(self, V, g_L, E_L, I, C_m, dt):

        """
        Runge kutta integrator
        ----------
        V: float
            Voltage of the node object
            
        g_L: float
            leak channel conductance
            
        E_L: float 
            Reversal potential leak channels
            
        I: float
            Current going into the node/neuron
            
        C_m: float
            Membrane conductance
            
        dt: float
            Timestep that will be used
            
        Returns
        -------
        
        V:  float
            Returns the voltage for the node object
            
            
        """
        k1 = (g_L * (E_L - V) + I) / C_m
        k2 = (g_L * (E_L - (V + k1 * 0.5 * dt)) + I) / C_m
        k3 = (g_L * (E_L - (V + k2 * 0.5 * dt)) + I) / C_m
        k4 = (g_L * (E_L - (V + k3 * dt)) + I) / C_m

        V = V + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)

        return V

    def I_syn(self, W, ID, spikes, V, E_L, S):
        """
        Computes the synaptic current for the node/neuron
        ----------

        Iapp: float
            Input current
        
        W: 2d array with ints
            Contains the weight/adjecency matrix, values are -1,0 or 1
            
        ID: int
            Gives the ID of the node in question w.r.t the entire network
         
        V: float
            Voltage of the node object
            
        E_L: float
            reversal potential of the leak channel
            
        spikes list of list of ints
            Contains the spike trains for all nodes
            
        S: int
            Gives the current time point in the simulation
            
        Returns
        -------
        
        V:  float
            Returns the voltage for the node object
            
        spike: int
            Returns whether a spike occured or not
            
        """

        weight_matrix = W

        # Find the indices of presynaptic neurons with connections to the current neuron (ID) excitatory
        connections_exc = np.where((weight_matrix[:, ID] > 0) & (weight_matrix[:, ID] != 0))[0]
        mask_exc = np.zeros([len(weight_matrix), ])
        mask_exc[connections_exc] = 1
        spike_filtered_exc = np.array(spikes) * mask_exc[:, np.newaxis]

        # Extract spike times for the connected presynaptic neurons and convert to milliseconds

        nonzero_indices_exc = np.where(spike_filtered_exc != 0)[1]
        nonzero_indices_exc = np.split(nonzero_indices_exc,
                                       np.cumsum(np.unique(nonzero_indices_exc, return_counts=True)[1])[:-1])

        spike_times_exc = np.concatenate(nonzero_indices_exc)

        I_exc = g_syn_exc * np.sum(np.exp(-(S * dt - spike_times_exc * dt) / tau_syn) * (V - E_syn_exc))

        # Find the indices of presynaptic neurons with connections to the current neuron (ID) inhibitory
        connections_inh = np.where((weight_matrix[:, ID] < 0) & (weight_matrix[:, ID] != 0))[0]
        mask_inh = np.zeros([len(weight_matrix), ])
        mask_inh[connections_inh] = 1
        spike_filtered_inh = np.array(spikes) * mask_inh[:, np.newaxis]

        # Extract spike times for the connected presynaptic neurons and convert to milliseconds

        nonzero_indices_inh = np.where(spike_filtered_inh != 0)[1]
        nonzero_indices_inh = np.split(nonzero_indices_inh,
                                       np.cumsum(np.unique(nonzero_indices_inh, return_counts=True)[1])[:-1])

        spike_times_inh = np.concatenate(nonzero_indices_inh)

        I_inh = g_syn_inh * np.sum(np.exp(-(S * dt - spike_times_inh * dt) / tau_syn) * (V - E_syn_inh))

        return -I_exc + I_inh


# %% Setup network

def plot_network(graph, sigma=None, title="Network"):
    """
    Plots network objects
    ----------
        
    graph: networkx object
        networkx object
        
    sigma: float
        sigma value of the network (measure of small-worldness)
        
    title: string
        Title of the plot
    
    Returns
    -------
    void
        
    """
    plt.figure(figsize=(10, 8))

    # Compute network metrics
    average_degree = np.mean(list(dict(graph.degree()).values()))
    betweenness_centralities = nx.betweenness_centrality(graph)
    average_betweenness = np.mean(list(betweenness_centralities.values()))
    closeness_centralities = nx.closeness_centrality(graph)
    average_closeness = np.mean(list(closeness_centralities.values()))
    clustering_coefficient = nx.average_clustering(graph)

    # Construct the text for the textbox
    stats_text = (f'Average Degree: {average_degree:.2f}\n'
                  f'Average Betweenness: {average_betweenness:.4f}\n'
                  f'Average Closeness: {average_closeness:.4f}\n'
                  f'Clustering Coefficient: {clustering_coefficient:.4f}')

    # Place the textbox inside the plot
    plt.gca().text(0.05, 0.05, stats_text, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.3",
                                                                      edgecolor="grey", facecolor="aliceblue"))

    # Extract edge weights from the graph
    edge_weights = nx.get_edge_attributes(graph, 'weight')

    # Compute betweenness centralities
    betweenness_centralities = nx.betweenness_centrality(graph)
    centrality_values = list(betweenness_centralities.values())

    # Compute node degrees for size
    degrees = dict(graph.degree())
    min_degree = min(degrees.values())
    max_degree = max(degrees.values())
    min_size = 200
    max_size = 1000
    node_sizes = [min_size + (degree - min_degree) * (max_size - min_size) / (max_degree - min_degree) for degree in
                  degrees.values()]

    # Set up the graph layout
    pos = nx.spring_layout(graph, seed=seed, k=1)

    # Draw nodes with adjusted sizes and colors based on betweenness centrality
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color=centrality_values, cmap='viridis')

    # Classify edges based on their weights
    orange_edges = [edge for edge, weight in edge_weights.items() if weight == 1]
    blue_edges = [edge for edge, weight in edge_weights.items() if weight == -1]

    # Draw edges with adjusted colorblind-friendly colors
    if orange_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=orange_edges, edge_color='orange')
    if blue_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=blue_edges, edge_color='lightblue')

    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_color='white')

    # Add legend with custom legend entries
    custom_legend = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markeredgecolor='black', markersize=10,
                   label='Node (Size ~ Degree)'),
        plt.Line2D([0], [0], color='orange', lw=2, label='Edge (Weight = 1)'),
        plt.Line2D([0], [0], color='lightblue', lw=2, label='Edge (Weight = -1)')
    ]
    plt.legend(handles=custom_legend, loc='upper right')

    # Display sigma as the title
    if sigma is not None:
        plt.title('{} Small-worldness coefficient (σ): {}'.format(title, sigma))
    else:
        plt.title('{} Small-worldness coefficient (σ): Can not be determined'.format(title))

    # Add colorbar for betweenness centrality
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=min(centrality_values), vmax=max(centrality_values)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), label='Betweenness Centrality')
    plt.show()


def plot_degree(full_graph, title):
    """
    Plots the degree distriution of the network
    ----------
        
    ful_graph: networkx object
        networkx object
        
    title: string
        Title of the plot
    
    Returns
    -------
    void
        
    """
    # Calculate the degree of each node
    degrees = [degree for node, degree in full_graph.degree()]

    # Plot the degree distribution as a histogram
    plt.figure()
    plt.hist(degrees, bins=20, alpha=0.75)
    plt.title(title)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()

    return degrees


def visualize_weight_matrix(weight, title=' '):
    """
    Plots network objects
    ----------
        
    weight: 2d array of ints
        The weight/adjecency matrix of the network
        
    title: string
        Title of the plot
    
    Returns
    -------
    void
        
    """

    # Set colorblind-friendly palette (e.g., 'viridis')
    cmap = sns.color_palette("viridis")
    plt.figure()
    # Create the heatmap
    sns.heatmap(weight, cmap=cmap, annot=True, square=True)

    # Show the plot
    plt.title("{} Matrix Heatmap".format(title))
    plt.xlabel("Node")
    plt.ylabel("Node")
    plt.show()


def get_sigma(network, num_nodes):
    """
    Plots network objects
    ----------
        
    network: networkx object
        networkx object
        
    num_nodes: int
        Gives the number of nodes
    
    Returns
    -------
    void
        
    """
    path_length_network = nx.average_shortest_path_length(network) if nx.is_strongly_connected(network) else float(
        'inf')
    clustering_network = nx.average_clustering(network, count_zeros=True)

    # get random metrics
    random_graph = nx.fast_gnp_random_graph(num_nodes, p, directed=True, seed=seed)
    clustering_random = nx.average_clustering(random_graph, count_zeros=True)
    largest_scc_random = max(nx.strongly_connected_components(random_graph), key=len)
    subgraph_random = random_graph.subgraph(largest_scc_random)
    path_length_random = nx.average_shortest_path_length(subgraph_random) if nx.is_strongly_connected(
        subgraph_random) else float('inf')

    if path_length_network != 0 and path_length_network != np.inf:
        sigma = (clustering_network / clustering_random) / (path_length_network / path_length_random)
    else:
        sigma = None

    return sigma


class LIF_network():
    def __init__(self, seed):
        """
        Generates and runs the network of LIF neurons
        ----------
            
        seed:int
            Value of the random seed
        
        Returns
        -------
        void
            
        """

        random.seed(seed)
        np.random.seed(seed)

    def get_network(self, num_nodes, k, p, inhibitory_ratio, inhibitory_strength, seed):

        """
        Generatethe network structure using an adapted Strogatz-Watts algorithm
        ----------
            
        num_nodes: int
            number of nodes in the network
            
        k: float
            Determines the number of neighbours in the inital lattice structure of the network
            
        p: float
            The rewiring probility of the network
            
        inhibitory_ratio: float
            Determines what fraction of the nodes will be inhibitory
            
        inhibitory_strengh:
            The weight of the inhibitory connections
            
        seed: int
            Random seed value to be used
        
        Returns
        -------
        weight_matrix: 2d array of ints
            The weight matrix of the resulting network
            
        directed_graph: networkx object
            The network object of the resulting network
            
        """

        # Create an initial undirected network using the Watts-Strogatz small-world model
        undirected_graph = nx.watts_strogatz_graph(num_nodes, k, p, seed=seed)

        # Convert the undirected graph to a directed graph with random edge directionality
        directed_graph = nx.DiGraph()
        for edge in undirected_graph.edges():
            if random.choice([True, False]):
                directed_graph.add_edge(edge[0], edge[1], weight=1)
            else:
                directed_graph.add_edge(edge[1], edge[0], weight=1)

        # Determine the number of inhibitory connections
        num_edges = directed_graph.number_of_edges()
        num_inhibitory_connections = int(inhibitory_ratio * num_edges)

        # Convert a portion of the connections to inhibitory
        all_edges = list(directed_graph.edges(data=True))
        random.shuffle(all_edges)
        for edge in all_edges[:num_inhibitory_connections]:
            edge[2]['weight'] = inhibitory_strength

        # Metrics for directed graph
        clustering_directed = nx.average_clustering(directed_graph, count_zeros=True)
        largest_scc = max(nx.strongly_connected_components(directed_graph), key=len)
        subgraph = directed_graph.subgraph(largest_scc)
        path_length_directed = nx.average_shortest_path_length(subgraph) if nx.is_strongly_connected(
            subgraph) else float('inf')

        # Regular lattice (k nearest neighbors in a directed ring)
        lattice = nx.DiGraph()
        nodes = list(range(num_nodes))
        half_k = k // 2  # considering k nearest neighbors: half from left and half from right
        for i in nodes:
            for j in range(1, half_k + 1):
                lattice.add_edge(i, (i + j) % num_nodes)
                lattice.add_edge(i, (i - j) % num_nodes)
        clustering_lattice = nx.average_clustering(lattice, count_zeros=True)
        path_length_lattice = nx.average_shortest_path_length(lattice)

        # Random directed graph

        random_graph = nx.fast_gnp_random_graph(num_nodes, p, directed=True, seed=seed)
        clustering_random = nx.average_clustering(random_graph, count_zeros=True)
        largest_scc_random = max(nx.strongly_connected_components(random_graph), key=len)
        subgraph_random = random_graph.subgraph(largest_scc_random)
        path_length_random = nx.average_shortest_path_length(subgraph_random) if nx.is_strongly_connected(
            subgraph_random) else float('inf')

        # Create weight matrix
        weight_matrix = nx.to_numpy_array(directed_graph)

        return weight_matrix, directed_graph

    def init_model(self, num_nodes, s_V, mean_V, s_I, mean_I):
        """
        Initializes the voltages, currents and spikes for every node
        ----------
            
        num_nodes: int
            Gives the number of nodes
            
        s_V: float
            standard dev of the initial voltages
            
        mean_V: float
            Mean value of the inital voltages
        
        Returns
        -------
        nodes: list of lists offloats
            Contains the time series of the voltage trace (now just one index)
            
        current: list of floats
            Gives the input current of each node
            
        spikes:
            Contains the spike trains of all nodes (now just one index)
            
        """
        np.random.seed(seed)

        # Generate initial voltages using a normal distribution
        initial_voltages = np.random.normal(mean_V, s_V, num_nodes)
        # initial_voltages = mean_V + s_V * np.random.standard_cauchy(num_nodes)
        nodes = [[v] for v in initial_voltages]  # Convert each voltage into a list

        # Generate initial currents using a normal distribution
        currents = np.random.normal(mean_I, s_I, num_nodes)
        # currents = mean_I + s_I * np.random.standard_cauchy(num_nodes)

        # Initialize spikes as zeros
        spikes = [[0] for _ in range(num_nodes)]

        return nodes, currents, spikes

    def run_model(self, spikes, weight_matrix, nodes, currents, num_nodes):

        """
        IRuns the model of the network
        ----------
            
        spikes: list of listsof ints
            The spike train for eachof thenodes
            
        s_V: float
            standard dev of the initial voltages
            
        mean_V: float
            Mean value of the inital voltages
        
        Returns
        -------
        nodes: list of lists offloats
            Contains the time series of the voltage trace (now just one index)
            
        current: list of floats
            Gives the input current of each node
            
        spikes:
            Contains the spike trains of all nodes (now just one index)
            
        """
        s = 1
        while s - 1 < T:
            print(s - 1)
            prev_spikes = np.array(spikes)
            n = 3

            for ID in range(num_nodes):
                I = currents[ID]
                V_node = Node(I, weight_matrix, ID, nodes, prev_spikes, s - 1).Vt
                spike_node = Node(I, weight_matrix, ID, nodes, prev_spikes, s - 1).spike
                nodes[ID].append(V_node)
                spikes[ID].append(spike_node)
            s += 1
        return nodes, spikes

    def spike_plot(self, spikes):
        """
        PLots the spike train of each node
        ----------
        spikes: list of lists of ints
        
        Returns
        -------
        void
            
        """
        binary_matrix = np.array(spikes)
        # Get the number of rows and columns in the matrix
        num_rows, num_cols = binary_matrix.shape

        # Create a new figure for the plot
        plt.figure()

        # Iterate through each row of the binary matrix
        for row_index, row in enumerate(binary_matrix):
            # Create a list of x-coordinates for ones in this row
            x_coords = [col_index for col_index, value in enumerate(row) if value == 1]

            # Create y-coordinates for this row (all the same)
            y_coords = [row_index] * len(x_coords)

            # Plot ones as points
            plt.scatter(x_coords, y_coords, marker='o', s=100, label=f'Row {row_index + 1}')

        # Set labels and title
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")
        plt.title("Binary Matrix Plot")

        # Invert the y-axis to have the rows from top to bottom
        plt.gca().invert_yaxis()

        # Show the plot
        plt.grid()
        plt.show()


# %% run network 1


dur = 300  # in s
T = dur / dt

num_nodes = 50
k = 9
p = 0.05

inhib_ratio = 0.2
inhibitory_strength = -1  # Strength of inhibitory connections

mean_V = -80
s_V = 1
mean_I = 0.9
s_I = .1
mean_osc = 3
s_osc = 0

network_obj = LIF_network(seed)
weight_matrix, graph = network_obj.get_network(num_nodes, k, p, inhib_ratio, inhibitory_strength, seed)
sigma = get_sigma(graph, num_nodes)
plot_network(graph, sigma, "True network")
degrees = plot_degree(graph, "Degree dist true network")
no_noise_nodes, currents, spikes = network_obj.init_model(num_nodes, s_V, mean_V, s_I, mean_I)

no_noise_nodes, spikes = network_obj.run_model(spikes, weight_matrix, no_noise_nodes, currents, num_nodes)
network_obj.spike_plot(spikes)
visualize_weight_matrix(weight_matrix, "True network")


# %% add noise

def add_noise_to_timeseries(time_series, mean=0, std=.25):
    """Add Gaussian noise to a time series."""
    noise = np.random.normal(mean, std, len(time_series))
    return time_series + noise


def add_noise_to_nodes(nodes, mean=0, std=.25):
    """Add noise to all time series in a list."""
    return [add_noise_to_timeseries(ts, mean, std) for ts in nodes]


def extract_elements_from_index(list_of_lists, index):
    return [sublist[index:] for sublist in list_of_lists]


nodes = no_noise_nodes

# %%

weight_flattened = weight_matrix.flatten()

# Import the java packages
teCalcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
teCalc = teCalcClass()
teCalc.setProperty("NORMALISE", "true")  # Normalise the individual variables
teCalc.initialise(1, .5)


def transfer_entropy_stored(nodes, n):
    """
    Computes the transfer entropy for the timeseries of each pair of nodes
    ----------
    nodes: list of lists of floats
        Contains all the nodes with their timeseries voltage trace
        
    n: int
        Number of nodes
    
    Returns
    -------
    TEs: 2d numpy array of float
        Contains the transfer entropy for each pair of nodes
        
    """

    # downsampled_nodes = [scipy.signal.resample(node,int(len(node/10))) for node in nodes]
    # encoded_nodes = [encode_series(node) for node in downsampled_nodes]
    TEs = np.zeros([n, n])
    for i in tqdm(range(n), desc="Rows TE", position=0):
        for j in tqdm(range(n), desc="Cols TE", position=1):
            teCalc.setObservations(JArray(JDouble, 1)(nodes[i]), JArray(JDouble, 1)(nodes[j]))
            result = teCalc.computeAverageLocalOfObservations()
            TE_value = result
            TEs[i, j] = TE_value

    return TEs


def infer_network_direction(TE_grid, weight_flattened, n, TEs):
    """
    Infer edges of the network with their direction
    ----------
    TE_grid: list of float
        Grid containing the threshold values for the transfer entropy 
        
    weight_flattened: 1d numpy array of int
        Flattened weight/adjecency matrix
        
    n: int
        Number of nodes
        
    TEs: 2d array of float
        The array that has transfer entropy for each pair stored
    
    Returns
    -------
    recall_vals: list of float
        Returns recall values for each threshold
        
    precision_vals: list of float
        Returns precision values for each threshold
        
    f1_scores: list of float
        Returns f1-scores for each threshold
        
    inferred_network: networkx object
        Returns the inferred network at a threshold value
        
    """
    precision_vals = []
    recall_vals = []
    f1_scores = []

    for TE_threshold in TE_grid:

        inferred_network = nx.DiGraph()
        inferred_network.add_nodes_from(range(n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    inferred_network.add_edge(i, j, weight=0)
                    continue

                TE_value = TEs[i, j]

                # Adjust this condition as necessary; for instance, you might want to set a threshold on TE_value.
                if TE_value > TE_threshold:
                    inferred_network.add_edge(i, j, weight=1)  # Assigning weight based on TE value. Modify as needed.

        weight_flattened[weight_flattened == -1] = 1
        flat_inferred = nx.to_numpy_array(inferred_network).flatten()
        flat_inferred[flat_inferred == -1] = 1
        avg_f1 = f1_score(weight_flattened, flat_inferred, labels=[0, 1], average='macro')
        avg_p = precision_score(weight_flattened, flat_inferred, labels=[0, 1], average='macro')
        avg_r = recall_score(weight_flattened, flat_inferred, labels=[0, 1], average='macro')
        f1_scores.append(avg_f1)
        precision_vals.append(avg_p)
        recall_vals.append(avg_r)

    return recall_vals, precision_vals, f1_scores, inferred_network


# Define the grid of threshold values
te_threshold_grid = np.linspace(0, 1, 1000)

n = len(nodes)

TEs = transfer_entropy_stored(nodes, n)

recall_vals, precision_vals, f1_scores, __ = infer_network_direction(te_threshold_grid, weight_flattened, n, TEs)

plt.figure(figsize=(10, 8))


def is_far_enough(current_point, previous_point, threshold=0.025):
    """
    Makes it so that labels of points in the plots do not overlap
        
    """
    # Compute Euclidean distance between the two points
    distance = ((current_point[0] - previous_point[0]) ** 2 + (current_point[1] - previous_point[1]) ** 2) ** 0.5
    return distance > threshold


# Plot all the points and connect them with a line
plt.plot(recall_vals, precision_vals, marker='o', linestyle='--', color='b')

# Initialize a variable to keep track of the last annotated point
last_annotated = (recall_vals[0], precision_vals[0])
plt.annotate(f"{te_threshold_grid[0]:.2e}", (last_annotated[0], last_annotated[1] + 0.01), fontsize=8)

for i in range(1, len(te_threshold_grid)):
    current_point = (recall_vals[i], precision_vals[i])
    if is_far_enough(current_point, last_annotated):
        plt.annotate(f"{te_threshold_grid[i]:.2e}", (current_point[0], current_point[1] + 0.01), fontsize=8)
        last_annotated = current_point

# Add grid lines
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Additional plot properties like labels, title, etc.
plt.xlabel('Recall (Macro-average)')
plt.ylabel('Precision (Macro-average)')
plt.title('Macro-Average Precision-Recall Curve with TE thresholds')

plt.show()


# %%

def get_all_interspike_interval_indices(spike_train):
    """
    Get the interspike intervals from the spike_train
    ----------
    spike_train: 1d array of int
        Spike train for a node
    
    Returns
    -------
    intervals: list of tuples
        Each tuple in the list corresponds to the start and end of the interval
        
    """
    # Assuming the spike_train is a binary list where 1s represent spikes
    spike_indices = [idx for idx, val in enumerate(spike_train) if val == 1]

    # Check if we have less than 2 spikes, then no intervals to consider
    if len(spike_indices) < 2:
        return []

    intervals = [(spike_indices[i], spike_indices[i + 1]) for i in range(len(spike_indices) - 1)]
    return intervals


def infer_sign(nodes, n, spikes, lag=1):
    """
    Calculates the correlation matrix between pairs of nodes using the interspiking
    time intervals as correlation window
    ----------
    nodes: list of lists of float
        Contains the timeseries voltae trace of each node
        
    n: int
        Number of nodes
        
    spikes:
        Contains the spike train of each node
    
    Returns
    -------
    correlation_matrix: 2d numpy array
        Correlation matrix between each pair of nodes 
        
    """
    correlation_matrix = np.zeros((n, n))

    for i in range(n):
        intervals = get_all_interspike_interval_indices(spikes[i])
        temp_correlation_matrix = np.zeros((n, n))

        # Loop through all interspike intervals of time series i
        for start, end in intervals:
            for j in range(n):
                correlation, _ = spearmanr(add_noise_to_timeseries(nodes[i])[start + lag:end],
                                           add_noise_to_timeseries(nodes[j])[start:end - lag])
                temp_correlation_matrix[i, j] += correlation

        # Average out the correlations
        correlation_matrix[i, :] = temp_correlation_matrix[i, :] / len(intervals) if intervals else 0

    return correlation_matrix


def infer_network_sign(inferred_network, n, signs, sign_thresholds, weight_flattened):
    """
    Infers the sign of each edge in the network
    ----------
    nodes: list of lists of float
        Contains the timeseries voltae trace of each node
        
    n: int
        Number of nodes
        
    spikes:
        Contains the spike train of each node
    
    Returns
    -------
    correlation_matrix: 2d numpy array
        Correlation matrix between each pair of nodes 
        
    """
    inferred_signed = nx.DiGraph()
    inferred_signed.add_nodes_from(range(n))
    inferred_network_weights = nx.to_numpy_array(inferred_network)
    f1_scores = []
    precision_vals = []
    recall_vals = []
    for sign_thresh in tqdm(sign_thresholds, desc="Threshololds", position=0):
        inferred_weights_signed = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                if i == j:
                    inferred_signed.add_edge(i, j, weight=0)
                    continue

                sign = signs[i, j]

                # Adjust this condition as necessary; for instance, you might want to set a threshold on TE_value.
                if inferred_network_weights[i, j] == 1 and sign >= sign_thresh:
                    inferred_signed.add_edge(i, j, weight=1)  # Assigning weight based on TE value. Modify as needed.
                elif inferred_network_weights[i, j] == 1 and sign < sign_thresh:
                    inferred_signed.add_edge(i, j, weight=-1)

        flat_inferred = nx.to_numpy_array(inferred_signed).flatten()
        avg_f1 = f1_score(weight_flattened, flat_inferred, labels=[-1, 0, 1], average='macro')
        avg_p = precision_score(weight_flattened, flat_inferred, labels=[-1, 0, 1], average='macro')
        avg_r = recall_score(weight_flattened, flat_inferred, labels=[-1, 0, 1], average='macro')

        f1_scores.append(avg_f1)
        precision_vals.append(avg_p)
        recall_vals.append(avg_r)

    return recall_vals, precision_vals, f1_scores, inferred_signed


TE_grid = [1.1e-2]
__, __, __, best_network = infer_network_direction(TE_grid, weight_flattened, n, TEs)

n = len(nodes)
sign_thresholds = np.linspace(-1, 1, 100)
signs = infer_sign(nodes, n, spikes)

recall_vals_sign, precision_vals_sign, f1_scores_sign, __ = infer_network_sign(best_network, n, signs, sign_thresholds,
                                                                               weight_flattened)

plt.figure(figsize=(10, 8))

# Plot all the points and connect them with a line
plt.plot(recall_vals_sign, precision_vals_sign, marker='o', linestyle='--', color='b')

# Initialize a variable to keep track of the last annotated point
last_annotated = (recall_vals_sign[0], precision_vals_sign[0])
plt.annotate(f"{sign_thresholds[0]:.2e}", (last_annotated[0], last_annotated[1] + 0.01), fontsize=8)

for i in range(1, len(sign_thresholds)):
    current_point = (recall_vals_sign[i], precision_vals_sign[i])
    if is_far_enough(current_point, last_annotated, threshold=0.01):
        plt.annotate(f"{sign_thresholds[i]:.2e}", (current_point[0], current_point[1] + 0.01), fontsize=8)
        last_annotated = current_point

# Add grid lines
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Additional plot properties like labels, title, etc.
plt.xlabel('Recall (Macro-average)')
plt.ylabel('Precision (Macro-average)')
plt.title('Macro-Average Precision-Recall Curve with sign thresholds')
plt.show()

# %%
sign_thresholds = [2.53e-1]
n = len(nodes)
__, __, __, best_network_signs = infer_network_sign(best_network, n, signs, sign_thresholds, weight_flattened)

# %% Print reports for directed and directed signed networks

# Display results directed network
best_network_weights = nx.to_numpy_array(best_network)
report = classification_report(weight_flattened, best_network_weights.flatten(), labels=[0, 1], zero_division=0)
print(report)

# Display results directed signed network
best_network_signs_weights = nx.to_numpy_array(best_network_signs)
report_signed = classification_report(weight_flattened, best_network_signs_weights.flatten(), labels=[-1, 0, 1],
                                      zero_division=0)
print(report_signed)

# %%


sigma = get_sigma(best_network_signs, len(nodes))
# Plot directed network and degree dist 
visualize_weight_matrix(nx.to_numpy_array(best_network), "Directed inferred network")
plot_degree(best_network, "Degree dist directed inferred network")
plot_network(best_network, sigma, "Directed inferred network ")

# Plot directed signed network and degree dist
plot_network(best_network_signs, sigma, "Directed inferred network with sign")
plot_degree(best_network_signs, "Degree dist directed inferred network")
visualize_weight_matrix(best_network_weights, "Directed inferred network with sign")


# %% Get metrics

def RQA_metrics(spikes, nodes):
    mean_isi = []
    recurrence_matrix = []
    recurrence_nt_matrix = []
    recurrence_flattened = []
    metrics = []
    time_delay = 10
    embedding_dimension = 5
    freq_dist = []
    spiking_times = []
    neighbourhood = FixedRadius(5)
    for i, train in enumerate(spikes):
        train = np.array(train)
        spike_times = np.where(train != 0)[0] * dt
        sorted_times = np.sort(spike_times)
        isi = np.diff(sorted_times)

        signal = nodes[i]
        time_series = TimeSeries(signal,
                                 embedding_dimension=embedding_dimension,
                                 time_delay=time_delay)
        settings = Settings(time_series,
                            analysis_type=Classic,
                            neighbourhood=neighbourhood,
                            similarity_measure=EuclideanMetric,
                            theiler_corrector=1)
        computation = RQAComputation.create(settings,
                                            verbose=False)

        # Create image from recurrence matrix
        computation_im = RPComputation.create(settings,
                                              verbose=False)
        result = computation.run()

        settings = Settings(time_series,
                            analysis_type=Classic,
                            neighbourhood=Unthresholded(),
                            similarity_measure=EuclideanMetric)
        computation = RPComputation.create(settings)
        result_nt = computation.run()
        recurrence_nt_matrix.append(result_nt.recurrence_matrix)

        recurrence_matrix.append(computation_im.run())

        spiking_times.append(np.where(np.array(spikes[i]) != 0)[0])

        non_zero_diag_freq = [x for x in result.diagonal_frequency_distribution if x != 0]
        non_zero_vert_freq = [x for x in result.vertical_frequency_distribution if x != 0]

        rec_result = {'ratio determinism and recurrence rate': result.ratio_determinism_recurrence_rate,
                      'ratio determinism and laminarity': result.ratio_laminarity_determinism,
                      'entropy vertical lines': result.entropy_vertical_lines,
                      'entropy diag lines': result.entropy_diagonal_lines,
                      'entropy white vertical lines': result.entropy_white_vertical_lines,
                      'average diag line': result.average_diagonal_line,
                      'average white vertical line': result.average_white_vertical_line,
                      'Trapping time': result.trapping_time,
                      'Divergence': result.divergence,
                      'Longest diag line': result.longest_diagonal_line,
                      'Longest vertical line': result.longest_vertical_line,
                      # 'min diagonal line length': result.min_diagonal_line_length,
                      # 'min vertical line length': result.min_vertical_line_length,
                      'mean isi time': np.mean(isi),
                      'std isi time': np.std(isi),
                      'mean diagonal frequency': np.mean(result.diagonal_frequency_distribution),
                      'median diagonal frequency': np.median(non_zero_diag_freq),
                      'mean vertical frequency': np.mean(result.vertical_frequency_distribution),
                      'median vertical frequency': np.median(non_zero_vert_freq),
                      'number of spikes': np.sum(spikes[i])
                      }

        metrics.append(rec_result)
        freq_dist.append(result.diagonal_frequency_distribution)

    return metrics
