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
        
        
    def get_network(self, n, k, P):
        """
        Generate a directed binary stochastic block model graph.

        Parameters:
        n : int
            Total number of nodes.
        k : int
            Number of communities.
        P : 2D array
            k x k matrix of edge probabilities between communities.

        Returns:
        G : NetworkX DiGraph
            Generated directed binary SBM graph.
        weight_matrix : 2D array
            n x n matrix of edge weights (0 or 1).
        """
        # Assign nodes to communities
        community_sizes = [n // k] * k
        for i in range(n % k):
            community_sizes[i] += 1

        node_community = []
        for i, size in enumerate(community_sizes):
            node_community.extend([i] * size)
        np.random.shuffle(node_community)

        # Initialize weight matrix
        weight_matrix = np.zeros((n, n))

        # Create directed graph
        G = nx.DiGraph()
        G.add_nodes_from(range(n))

        # Generate edges and weights based on P
        for i in range(n):
            for j in range(n):
                if i != j:
                    ci = node_community[i]
                    cj = node_community[j]
                    if np.random.rand() < P[ci, cj]:
                        G.add_edge(i, j, weight=1)
                        weight_matrix[i, j] = 1

        return weight_matrix,G


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


    def calculate_edge_lengths(self,G):
        """
        Calculates the length of the edges in the graph G based on a spring layout.

        Parameters:
        -----------
        G : networkx.Graph
            The input graph.
        seed : int, optional
            Seed for the random number generator (default is None).

        Returns:
        --------
        edge_lengths : dict
            A dictionary with edges as keys and their lengths as values.
        """
        # Compute the spring layout positions for each node
        pos = nx.spring_layout(G, seed=self.seed)

        # Calculate the Euclidean distance for each edge based on node positions
        edge_lengths = {edge: np.linalg.norm(np.array(pos[edge[0]]) - np.array(pos[edge[1]])) for edge in G.edges()}

        return edge_lengths