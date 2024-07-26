import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk


class plotting:
    def __init__(self, seed):
        self.seed = seed

    
    def plot_network(self, G,W, title="Network"):
        """
        Plots a network with nodes labeled by their numbers and sizes representing their degrees.

        Parameters
        ----------
        G: networkx DiGraph object
            Networkx graph object.

        title: string
            Title of the plot.

        Returns
        -------
        void
        """
        plt.figure(figsize=(10, 8))
        plt.title(title)

    
        # Calculate degrees from the weight matrix
        degrees = np.sum(W, axis = 1) + np.sum(W, axis = 0) 


        # Normalize node sizes based on degree
        min_size = 200
        max_size = 1000
        min_degree = degrees.min()
        max_degree = degrees.max()
        if max_degree == min_degree:
            node_sizes = [min_size for _ in degrees]  # or any other fixed size you prefer
        else:
            node_sizes = [min_size + (degree - min_degree) * (max_size - min_size) / (max_degree - min_degree) for degree in degrees]

        # Set up the graph layout
        pos = nx.spring_layout(G, seed=42)

        # Draw nodes with sizes based on degree
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue')

        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray')

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')
        plt.show(block = False)

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


    def plot_traces(self, V_all_soma, T_pre, dt):
        num_neurons = len(V_all_soma)

        # Create a tkinter window
        root = tk.Tk()
        root.title("Scrollable Plot")

        # Set the window size to fit a reasonable number of subplots
        window_height = 600
        window_width = 800
        root.geometry(f"{window_width}x{window_height}")

        # Create a canvas and a vertical scrollbar for it
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=1)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        # Create a frame inside the canvas
        plot_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=plot_frame, anchor="nw")

        # Create the figure and axes
        fig, axes = plt.subplots(num_neurons, 1, figsize=(10, 2 * num_neurons), sharex=True)
        for i in range(num_neurons):

            axes[i].plot(V_all_soma[i][int(T_pre/dt):])
            axes[i].set_title(f'Neuron {i} Soma Voltage')
            axes[i].set_ylabel('Voltage (mV)')
        
        axes[-1].set_xlabel('Timesteps')
        plt.tight_layout()

        # Embed the figure in the tkinter window
        canvas_fig = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas_fig.draw()
        canvas_fig.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Update the scroll region
        plot_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

        # Start the tkinter main loop
        root.mainloop()

    def plot_combined_subnetworks(self, graph,num_nodes, num_sub_networks , sigma=None, title="Network"):
        """
        Plots network objects with improved separation between subclusters
        ----------
        graph: networkx object
            Networkx object representing the graph
        subclusters: list of lists
            List containing subclusters, each subcluster is a list of nodes
        sigma: float
            Sigma value of the network (measure of small-worldness)
        title: string
            Title of the plot

        Returns
        -------
        void
        """
        plt.figure(figsize=(10, 8))
        subclusters = [list(range(i * int(num_nodes/num_sub_networks), (i + 1) * int(num_nodes/num_sub_networks))) for i in range(num_sub_networks)]
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

        # Colors for different subclusters
        colors = plt.cm.rainbow(np.linspace(0, 1, len(subclusters)))

        # Draw nodes and edges for each subcluster
        for idx, subcluster in enumerate(subclusters):
            nx.draw_networkx_nodes(graph, pos, nodelist=subcluster, node_size=[node_sizes[node] for node in subcluster],
                                node_color=[colors[idx]], label=f'Subcluster {idx + 1}')
            subcluster_edges = [edge for edge in graph.edges(subcluster) if edge[1] in subcluster]
            nx.draw_networkx_edges(graph, pos, edgelist=subcluster_edges, edge_color=[colors[idx]])

        # Draw inter-cluster edges in black
        inter_cluster_edges = [edge for edge in graph.edges() if edge not in subcluster_edges]
        nx.draw_networkx_edges(graph, pos, edgelist=inter_cluster_edges, edge_color='black', style='dashed')

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
        plt.show(block=False)

    def plot_degree(self, full_graph, title = ''):
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
        plt.show(block=False)

        return degrees
