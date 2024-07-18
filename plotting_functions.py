import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class plotting:
    def __init__(self, seed):
        self.seed = seed
    def plot_network(self,graph, title="Network"):
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
        if max_degree == min_degree:
            node_sizes = [min_size for degree in degrees.values()]  # or any other fixed size you prefer
        else:
            node_sizes = [min_size + (degree - min_degree) * (max_size - min_size) / (max_degree - min_degree) for degree in degrees.values()]

        # Set up the graph layout
        pos = nx.spring_layout(graph, seed=self.seed, k=1)

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

        # Add colorbar for betweenness centrality
        sm = plt.cm.ScalarMappable(cmap='viridis',
                                norm=plt.Normalize(vmin=min(centrality_values), vmax=max(centrality_values)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), label='Betweenness Centrality')
        plt.show(block=False)
        

    def visualize_weight_matrix(self,weight, title=' '):
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
        plt.show(block=False)
