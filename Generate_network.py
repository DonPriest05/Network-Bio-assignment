import random
import numpy as np
import networkx as nx
from Generate_neuron import Neuron

class generate_network():
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
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        
    def get_network(self, num_nodes, k, p, inhibitory_ratio, inhibitory_strength):
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


        Returns
        -------
        weight_matrix: 2d array of ints
            The weight matrix of the resulting network

        directed_graph: networkx object
            The network object of the resulting network

        """

        # Create an initial undirected network using the Watts-Strogatz small-world model
        undirected_graph = nx.watts_strogatz_graph(
            num_nodes, k, p, seed=self.seed)

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
        clustering_directed = nx.average_clustering(
            directed_graph, count_zeros=True)
        largest_scc = max(nx.strongly_connected_components(
            directed_graph), key=len)
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

        random_graph = nx.fast_gnp_random_graph(
            num_nodes, p, directed=True, seed=self.seed)
        clustering_random = nx.average_clustering(
            random_graph, count_zeros=True)
        largest_scc_random = max(
            nx.strongly_connected_components(random_graph), key=len)
        subgraph_random = random_graph.subgraph(largest_scc_random)
        path_length_random = nx.average_shortest_path_length(subgraph_random) if nx.is_strongly_connected(
            subgraph_random) else float('inf')

        # Create weight matrix
        weight_matrix = nx.to_numpy_array(directed_graph)

        return weight_matrix, directed_graph
    

    def create_interconnected_network(self, num_nodes, k, p, inhibitory_ratio, inhibitory_strength, num_sub_networks, interconnect_ratio):

        """
        Create a network of interconnected small-world sub-networks
        ----------
        num_nodes: int
            number of nodes in each sub-network

        k: float
            Determines the number of neighbours in the initial lattice structure of each sub-network

        p: float
            The rewiring probability of each sub-network

        inhibitory_ratio: float
            Determines what fraction of the nodes in each sub-network will be inhibitory

        inhibitory_strength: float
            The weight of the inhibitory connections in each sub-network

        num_sub_networks: int
            Number of small-world sub-networks to create

        interconnect_ratio: float
            Fraction of nodes to use for interconnections between sub-networks

        Returns
        -------
        combined_weight_matrix: 2D array of ints
            The weight matrix of the combined network

        combined_graph: networkx object
            The combined network object
        """
        sub_networks = []
        
        # Generate small-world sub-networks
        for _ in range(num_sub_networks):
            weight_matrix, directed_graph = self.get_network(num_nodes, k, p, inhibitory_ratio, inhibitory_strength)
            sub_networks.append(directed_graph)
        
        # Combine the sub-networks into a single network
        combined_graph = nx.DiGraph()
        node_offset = 0
        
        for graph in sub_networks:
            mapping = {node: node + node_offset for node in graph.nodes()}
            nx.relabel_nodes(graph, mapping, copy=False)
            combined_graph = nx.compose(combined_graph, graph)
            node_offset += num_nodes
        
        # Ensure at least one connection between each pair of subclusters
        for i in range(len(sub_networks) - 1):
            nodes_a = list(sub_networks[i].nodes())
            nodes_b = list(sub_networks[i + 1].nodes())

            # Select nodes with high centrality but low degree
            centralities_a = nx.betweenness_centrality(sub_networks[i])
            degrees_a = dict(sub_networks[i].degree())
            sorted_nodes_a = sorted(nodes_a, key=lambda x: (-centralities_a[x], degrees_a[x]))
            
            centralities_b = nx.betweenness_centrality(sub_networks[i + 1])
            degrees_b = dict(sub_networks[i + 1].degree())
            sorted_nodes_b = sorted(nodes_b, key=lambda x: (-centralities_b[x], degrees_b[x]))
            
            # Create at least one connection
            combined_graph.add_edge(sorted_nodes_a[0], sorted_nodes_b[0], weight=1)

            # Create additional connections based on interconnect_ratio
            num_additional_connections = int(interconnect_ratio * num_nodes) - 1
            for j in range(num_additional_connections):
                if j < len(sorted_nodes_a) and j < len(sorted_nodes_b):
                    combined_graph.add_edge(sorted_nodes_a[j + 1], sorted_nodes_b[j + 1], weight=1)
        
        combined_weight_matrix = nx.to_numpy_array(combined_graph)
        
        return combined_weight_matrix, combined_graph
