
# Author - Andrei Bobu
# Last updated 28 April 2020
# This code provides several algorithms that make clustering on the GBM model 


import pandas as pd
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cluster import	SpectralClustering
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.cluster import KMeans
from scipy import sparse 
from scipy import special
import copy
import tqdm
import itertools
import time
import seaborn as sns 
from random import choice

class GBM_graph(nx.Graph):

	# Procedure to generate points in 1-dimensional torus

    def add_node(self, ground_label):
        coordinate = np.random.uniform(0,1)
        super().add_node(len(self.nodes), coordinate = coordinate, ground_label=ground_label, label = ground_label)
    
    # Function returns the distance between vertices i and j

    def distance(self, i,j):
        return min(abs(self.nodes[i]["coordinate"] - self.nodes[j]["coordinate"]), 1 - abs(self.nodes[i]["coordinate"] - self.nodes[j]["coordinate"]))

   #  def set_node_attributes(self, labels, label_name):
   #  	"""
			# This function sets the vertices labels. 
			
			# Parameters: 
			# -------------

			# labels --- array-like 
   #  	"""

   #  	labels_dict = dict(zip(list(self.nodes), labels))
   #  	nx.set_node_attributes(self, labels_dict, label_name)

    # def makeEdgesfast(self, n, r_in, r_out, labels_nodes):
    #     edges = []
        
    #     return edges


    def subgraph(self, nodes):
    	"""
			This function returns a copy of the graph taking only vertices mentioned in node array

			Parameters:
			-------------
			nodes : list or array of nodes 

			Returns:
			-------------
			new_G: networkx.Graph
    	"""

    	new_G = copy.deepcopy(self)
    	new_G.remove_nodes_from([node for node in new_G.nodes if node not in nodes]) 
    	# new_G.adj_matrix = nx.adjacency_matrix(new_G)
    	new_G.ground_labels = [new_G.nodes[node]["ground_label"] for node in new_G.nodes]

    	return new_G 

    def something(self):
        return 1

    def __init__(self, n_1=1, n_2=1, a=1, b=1, disp = False):
        super().__init__()
        self.n_1 = n_1
        self.n_2 = n_2
        self.a = a
        self.b = b
        self.r_in = a * np.log(n_1 + n_2) / (n_1 + n_2)
        self.r_out = b * np.log(n_1 + n_2) / (n_1 + n_2)

        # Add nodes from the community "0"... 
        for i in range(self.n_1):
            self.add_node(0)
        # and nodes from the community "1"
        for i in range(self.n_2):
            self.add_node(1)

        self.ground_labels = [self.nodes[node]["ground_label"] for node in self.nodes]
        
        # Add edges depending on the distance between nodes and their communities 
        # for i in range(len(self.nodes)):
        #     for j in range(i+1, len(self.nodes)):
        #         if(self.nodes[i]["ground_label"] == self.nodes[j]["ground_label"] and 
        #           self.distance(i,j) < self.r_in):
        #             self.add_edge(i,j)
        #         if(self.nodes[i]["ground_label"] != self.nodes[j]["ground_label"] and 
        #           self.distance(i,j) < self.r_out):
        #             self.add_edge(i,j)

        # bla = self.something()
        # self.makeEdgesfast(self.n_1 + self.n_2, self.r_in, self.r_out, self.ground_labels)
        cutoffMatrix = np.array([[self.r_in,self.r_out], [self.r_out,self.r_in]])
        positions = nx.get_node_attributes(self, 'coordinate')
        nodesIndexSortedByGeography = np.argsort( [ position for position in positions.values() ] )
   	    
        n = self.n_1 + self.n_2
        edges = []
        for i in range(n):
            j = (i+1) % n
            while ( torusDistance( positions[ nodesIndexSortedByGeography[i] ], positions[ nodesIndexSortedByGeography[j] ] ) < max(self.r_in, self.r_out) ) :
                if( torusDistance( positions[ nodesIndexSortedByGeography[i] ], positions[ nodesIndexSortedByGeography[j] ] ) < cutoffMatrix[self.ground_labels[ nodesIndexSortedByGeography[i] ] - 1, self.ground_labels[ nodesIndexSortedByGeography[j] ] - 1  ] ) :
                    edges.append( ( nodesIndexSortedByGeography[i], nodesIndexSortedByGeography[j] ) )
                j = (j+1) % n
        self.add_edges_from(edges)     

        # Define some useful attributes
        self.av_deg_in = 2 * self.r_in * n_1
        self.av_deg_out = 2 * self.r_out * n_1
        self.adj_matrix = nx.adjacency_matrix(self)
        

        if disp:
        	print("average degree inside:", self.av_deg_in)
        	print("average degree outside:", self.av_deg_out)

    # The function to plot a graph. Communities get different colors 

    def plot(self):
        fig, ax1 = plt.subplots(1, 1, sharey = True, figsize=(14, 7))
        pos = nx.spring_layout(self)
        ground_colors = [self.nodes[node]["ground_label"] for node in self.nodes]
        nx.draw(self, pos, ax1, with_labels=True, node_color=ground_colors)
        plt.show()
        plt.close()



#### The spectral clustering is taken from Max's file with slight modifications

def torusDistance(x,y):
	return min(abs(x-y), 1-abs(x-y))

def eigenvectorAnalysis (adjacencyMatrix, k, val, delta):
	    #detect communities from the sign of the second eigenvector of the adjacency matrix
	    vals, vecs = sparse.linalg.eigs(adjacencyMatrix.asfptype() , k=k, which = 'SM')
	    print(val)
	    idx = min_list(vals, val) + delta
	    print(vals)
	    print(vals[idx])
	    secondVector = vecs[:,idx]
	    secondVector = secondVector.astype('float64')
	    print(sum(np.sign(secondVector)))
	    labels_pred_spectral = checkSign(secondVector)
	    return labels_pred_spectral

def min_list(x, val):
	for i in range(len(x)):
		if x[i] < val and x[i+1] > val:
			return i
	return 1

def checkSign (vector):
    labels_pred = np.zeros(len(vector))
    for i in range(len(vector)):
        if (vector[i]<0):
            labels_pred[i] = 1
    return labels_pred

def max_1(x):
	return max(x, 1-x)

class Spectral_Clustering:
	def __init__(self, G, n_clusters = 2):
		adj_matrix = nx.adjacency_matrix(G)
		# self.prediction = eigenvectorAnalysis(adj_matrix)
		sc = SpectralClustering ( n_clusters = n_clusters, affinity='precomputed', assign_labels='discretize' )
		self.prediction = sc.fit_predict (adj_matrix)
		labels_dict = dict(zip(list(G.nodes), self.prediction))
		nx.set_node_attributes(G, labels_dict, "label")
		self.accuracy = max(accuracy_score(self.prediction, G.ground_labels), 1 - accuracy_score(self.prediction, G.ground_labels))

class Spectral_Clustering_choice:
	def __init__(self, G, n_clusters = 2, k = 2, delta = 0):
		adj_matrix = nx.normalized_laplacian_matrix(G)
		val = 2*G.b/(G.a + G.b)
		self.prediction = eigenvectorAnalysis(adj_matrix, 50, val, delta)
		# sc = SpectralClustering ( n_clusters = n_clusters, affinity='precomputed', assign_labels='discretize' )
		# self.prediction = sc.fit_predict (adj_matrix)
		labels_dict = dict(zip(list(G.nodes), self.prediction))
		nx.set_node_attributes(G, labels_dict, "label")
		self.accuracy = max(accuracy_score(self.prediction, G.ground_labels), 1 - accuracy_score(self.prediction, G.ground_labels))


#### The motif-counting is practically taken from Max's file

class Motif_Counting:
	def motif_counting_analysis(self, G, r_in, r_out):
		N = nx.number_of_nodes(G)
		if(r_in>0.5):
		    return print("r_in  cannot be bigger than 0.5")
		size1 = N/2
		size2 = N/2

		# print("bla")
		Gc=max(nx.connected_component_subgraphs(G), key=len)
		G_mst=nx.Graph()

		G_mst.add_nodes_from(Gc)
		for ed in Gc.edges():
		    if (calcmot1(Gc, ed)>=motif1(r_in,r_out,size1,size2)):
		        G_mst.add_edge(ed[0],ed[1])


		# print ("Number of connected components %d" % nx.number_connected_components(G_mst))
		comp = list(nx.connected_components(G_mst))
		labels_pred = np.zeros(N)
		for i in range(nx.number_connected_components(G_mst)):
		    for elt in comp[i]:
		        labels_pred[elt] = i
		self.accuracy = max(accuracy_score(labels_pred, G.ground_labels), 1 - accuracy_score(labels_pred, G.ground_labels))
		self.prediction = labels_pred

	def __init__(self, G):
		self.accuracy = 0
		self.motif_counting_analysis(G, G.r_in, G.r_out)

#### Some auxuliary functions for the motif-counting algo

def motif1(rs,rd,size1,size2):
    SameExpectation = 0.0;DiffExpectation = 0.0
    if(rs >= 2*rd):
        SameExpectation = 3*rs*0.5*(size1) + size2* 2*rd*rd*1.0/rs
        DiffExpectation = (size1 + size2)*2*rd
    else:
        SameExpectation = 3*rs*0.5*(size1) + size2* (2*rd - rs*0.5)
        DiffExpectation = (size1+size2)*(2*rs - rs*rs*0.5*1/rd)
    return (SameExpectation + DiffExpectation)*0.5

def calcmot1(Gc, edge):
    return len(set(Gc.neighbors(edge[0])).intersection(set(Gc.neighbors(edge[1]))))

#### The algorithm based on cliques 

class Clique_Partitioning:

	def clique_partitioning_analysis(self, G):

		# List of maximum cliques 
		cliques = nx.find_cliques(G)
		# print("bla")

		# Make a copy of nodes of the original graph. Put initially weight 0 to every edge 
		G_c = nx.Graph()
		G_c.add_nodes_from(G)
		G_c.add_edges_from(itertools.combinations(G_c.nodes, 2))
		for u,v,e in G_c.edges(data = True):
			e['weight'] = 0
		self.G_c = G_c 

		# Make weighted matrix based on the definition of Lu, Wahlstorm and Nehorai
		for C in cliques:
			for u,v in itertools.combinations(C, 2):
				G_c[u][v]['weight'] += len(C)

		# Apply Spectral clustering algorithm based on the weighted matrix 
		clustering = SpectralClustering(n_clusters=2, affinity = 'precomputed', assign_labels="discretize", random_state=42).fit(nx.adjacency_matrix(G_c))
		# self.prediction = eigenvectorAnalysis(nx.adjacency_matrix(G_c))
		self.prediction = clustering.ground_labels_
		self.accuracy = max(accuracy_score(self.prediction, G.ground_labels), 1 - accuracy_score(self.prediction, G.ground_labels))

	def __init__(self, G):
		self.accuracy = 0
		self.clique_partitioning_analysis(G)

#### The new algorithm of guys from UMAS 

class Motif_Counting_New:
	def motif_counting_analysis(self, G, a, b):
		t1 = np.real(2*b*(np.exp(special.lambertw((1-2*b)/(2*b*np.exp(1))) + 1) - 1))
		t2 = -np.real(2*b*(np.exp(special.lambertw((1-2*b)/(2*b*np.exp(1)), k = -1) + 1) - 1))
		n = G.number_of_nodes()
		Es = (2*b + t1) * np.log(n) / n
		Ed = (2*b - t2) * np.log(n) / n
		print(t2, (2*b-t2)*np.log(n))
		Gc = G.copy()
		del_edges = 0
		for edge in G.edges:
		    if not process(G, edge, Es, Ed):
		        Gc.remove_edge(edge[0], edge[1])
		       	del_edges += 1
		# Gc.plot()
		print(del_edges)
		
		labels_pred = np.zeros(n, dtype = int)
		k=0
		for connected_component in nx.connected_components(Gc):
		    for node in connected_component:
		        labels_pred[node] = k
		    k = k+1
		self.prediction = labels_pred
		self.accuracy = max(accuracy_score(self.prediction, G.ground_labels), 1 - accuracy_score(self.prediction, G.ground_labels))

	def __init__(self, G):
		self.accuracy = 0
		self.motif_counting_analysis(G, G.a, G.b)

def neg(x):
	if x == 0:
		return 1
	else:
		return 0

class Partition:
	def __init__(self, G, n_clusters = 2, intercluster_edge_min = 100, plot = False):

		if n_clusters == 1:
			sc = Spectral_Clustering(G, 2)
			self.cluster_accuracies = sc.accuracy
			self.accuracy = sc.accuracy
			self.prediction = sc.prediction
			return

		Spectral_Clustering(G, n_clusters)

		cluster_labels = list(set([G.nodes[node]["label"] for node in G.nodes]))
		cluster_nodes = {}

		for l in cluster_labels:
			cluster_nodes.update({l: [node for node in G.nodes if G.nodes[node]['label'] == l]})
			if plot: 
				sns.distplot([G.nodes[node]["coordinate"] for node in G.nodes if G.nodes[node]['label'] == l], label = "Cluster " + str(l)) 
				plt.legend()

		self.cluster_accuracies = []
		for l, cluster in cluster_nodes.items():
			sub_G = G.subgraph(cluster)
			# print(sub_G.nodes)
			sc = Spectral_Clustering(sub_G, 2)
			self.cluster_accuracies.append(accuracy_score(sc.prediction, sub_G.ground_labels))
			labels_dict = dict(zip(list(sub_G.nodes), sc.prediction))
			nx.set_node_attributes(G, labels_dict, "label")

		# print(nx.get_node_attributes(G, "label"))
		current_cluster = copy.deepcopy(cluster_nodes[0])
		del cluster_nodes[0]
		for i in range(len(cluster_labels)-1):
			for l, cluster in cluster_nodes.items():
				number_of_edges_same = 0
				number_of_edges_different = 0

				for (u,v) in itertools.product(current_cluster, cluster):
					if G.has_edge(u,v) and G.nodes[u]['label'] == G.nodes[v]['label']:
						number_of_edges_same += 1
					if G.has_edge(u,v) and G.nodes[u]['label'] != G.nodes[v]['label']:
						number_of_edges_different += 1

				if number_of_edges_same + number_of_edges_different > intercluster_edge_min:
					sub_G = G.subgraph(cluster)
					labels_dict = nx.get_node_attributes(sub_G, 'label')
					if number_of_edges_different > number_of_edges_same:
						labels_dict = {k: neg(v) for k, v in labels_dict.items()}
					nx.set_node_attributes(G, labels_dict, "label")
					current_cluster = copy.deepcopy(cluster)
					del cluster_nodes[l]

					break


			self.prediction = [v for v in nx.get_node_attributes(G, 'label').values()]
			self.accuracy = max_1(accuracy_score(G.ground_labels, self.prediction))


def process(G, edge, Es, Ed):
    count = calcmot1(G, edge)
    if (count/G.number_of_nodes() >= Es or count/G.number_of_nodes() <= Ed):
        return True
    else:
        return False

#### The simulation with  some of presented algorithms 

def simulation(algorithm, n_1 = 100, n_2 = 100, a = 1, b = 1, n_trials = 1, param_flg = 0, k = 2):
	avg_accuracy = 0
	graphs_array = []
	for i in range(n_trials):
		G = GBM_graph(n_1, n_2, a, b, disp = False)
		graphs_array += [G]
		if algorithm.__name__ == "Partition":
			d = 0.75*b + a**2/(4*b)	
			r = d * np.log(n_1)/n_1
			n_clusters = int(max(1,1/r))
			A = algorithm(G, n_clusters = n_clusters)
		else: 
			A = algorithm(G)
		avg_accuracy += A.accuracy / n_trials
	dict_temp = {"accuracy": avg_accuracy, "graphs": graphs_array}
	return dict_temp

#### The simulation with all algortihms for one value of b

def full_simulation2(a_start = 1, a_finish = 2, a_step = 1, b = 1, n_1 = 100, n_2 = 100, n_trials = 10):
	# This for total time for all algorithms 
	# total_time_sc = 0 
	# total_time_mc = 0 
	# total_time_cp = 0

	# Create an array with possible values of a
	a_array = np.arange(a_start, a_finish, a_step)
	sc_array = []
	mc_array = []
	cp_array = []

	for a in a_array:
	    time_sc = 0 
	    time_mc = 0 
	    time_cp = 0

	    # Launch algorithms and mesure the time of execution
	    start_time = time.time()
	    sc_array.append(simulation(Spectral_Clustering, n_1, n_2, a, b, n_trials))
	    time_sc = time.time() - start_time
	    mc_array.append(simulation(Motif_Counting, n_1, n_2, a, b, n_trials))
	    time_mc = time.time() - start_time - time_sc 
	    cp_array.append(simulation(Clique_Partitioning, n_1, n_2, a, b, n_trials))
	    time_cp = time.time() - start_time - time_sc - time_mc 
	    print("a = %.1f, b = %.1f, Spectral clustering = %d sec, Motif counting = %d sec, Clique Partitioning = %d sec" % (a,b,time_sc, time_mc, time_cp))

	return {"a": a_array, "sc": sc_array, "mc": mc_array, "cp": cp_array}

def full_simulation(algo_list, a_start = 1, a_finish = 2, a_step = 1, b = 1, n_1 = 100, n_2 = 100, n_trials = 10):
	# This for total time for all algorithms 
	# total_time_sc = 0 
	# total_time_mc = 0 
	# total_time_cp = 0

	# Create an array with possible values of a
	acc_array = []
	# G_dict = {}

	for a in np.arange(a_start, a_finish, a_step):
		cur_step = {"a": a}
		s = 'a = ' + str(a) + ', b = ' + str(b) + ', '
		for A in algo_list: 
			start_time = time.time() 
			sim = simulation(A, n_1, n_2, a, b, n_trials)
			acc = sim['accuracy']
			# print(acc)
			# G_dict.update({(a,b,n_1+n_2): sim['graphs']})
			time_sec = time.time() - start_time 
			cur_step.update({A.__name__ : acc}) 
			s += A.__name__ + ' = ' + str(np.around(acc*100)) + '% (' + str(int(time_sec)) + 'sec), '
			acc_array.append(cur_step) 
		print(s)

	a_array = [x.get('a') for x in acc_array]
	legend = []
	for A in algo_list:
		v_array = [x.get(A.__name__) for x in acc_array]
		plt.plot(a_array, v_array)
		legend.append(A.__name__)

	plt.legend(legend)
	plt.xlabel('$a$')
	plt.ylabel('$accuracy$')
	plt.grid(True)
	plt.show()

	return acc_array

def khren(G):

	result_s = []
	result_d = []
	passed_set = []

	list_neighbrs = {}

	for v in G.nodes:
		list_neighbrs.update({v: set(nx.neighbors(G, v))})

	for u in G.nodes:
		passed_set.append(u)
		for v in nx.non_neighbors(G, u):
			if not v in passed_set:
				cmn_nmbr = len(list_neighbrs[u] & list_neighbrs[v])
				# dist = nx.shortest_path_length(G,u,v)
				# if dist == 2:
				# cmn_nmbr = G.distance(u,v)
				if G.nodes[u]["ground_label"] == G.nodes[v]['ground_label']:
					result_s.append(cmn_nmbr)
				else:
					result_d.append(cmn_nmbr)

	max_s = max(result_s)
	max_d = max(result_d)

	print(max_s, max_d)

	return (result_s, result_d)	

	# for u in G.nodes:
	# 	max_number1 = 0
	# 	max_number2 = 0
	# 	for v in nx.non_neighbors(G, u):
	# 		cmn_nmbr = len([i for i in nx.common_neighbors(G,u,v)])
	# 		if cmn_nmbr > max_number2:
	# 			if cmn_nmbr > max_number1:
	# 				max_number1 = cmn_nmbr
	# 				if max_number1 + max_number2 > max_number:
	# 					v_max = v
	# 					u_max = u
	# 					max_number = max_number1 + max_number2
	# 			else:
	# 				max_number2 = cmn_nmbr
	# 				if max_number1 + max_number2 > max_number:
	# 					w_max = v
	# 					u_max = u
	# 					max_number = max_number1 + max_number2

	# print(max_number1, max_number2, G.nodes[v_max], G.nodes[w_max], len([i for i in nx.common_neighbors(G,G.nodes[v_max],G.nodes[w_max])]))
	# return (G.nodes[v_max]['ground_label'], G.nodes[w_max]['ground_label'])	
				

	# for (u,v) in itertools.combinations(G.nodes, 2):
	# 	if not G.has_edge(u,v) and G.nodes[u]["ground_label"] == G.nodes[v]['ground_label']:
	# 		result += [i for i in nx.common_neighbors(G,u,v)]

	# result1 = set(result)
	# return len(result1)

def khren2(G):

	result_s = {}
	result_d = {}
	passed_set = []

	list_neighbrs = {}

	for v in G.nodes:
		list_neighbrs.update({v: set(nx.neighbors(G, v))})

	for u in G.nodes:
		passed_set.append(u)
		for v in nx.non_neighbors(G, u):
			if not v in passed_set:
				cmn_nmbr = list_neighbrs[u] & list_neighbrs[v]
				# dist = nx.shortest_path_length(G,u,v)
				# if dist == 2:
				# cmn_nmbr = G.distance(u,v)
				if G.nodes[u]["ground_label"] == G.nodes[v]['ground_label']:
					result_s.update({(u,v): cmn_nmbr})
				else:
					result_d.update({(u,v): cmn_nmbr})

	# max_s = max(len(result_s.values()))
	max_s = len(max(result_s.values(), key = len)) 
	max_d = len(max(result_d.values(), key = len)) 
	print(max_s, max_d)

	potential = 0
	potential_set = set()
	result_s = {key:val for key, val in result_s.items() if len(val) > max_d}

	G_s = G.subgraph(list(G.nodes))
	edges = list(G_s.edges())
	G_s.remove_edges_from(edges)

	for (u,v) in result_s:
		G_s.add_edge(u,v)

	print(G_s.number_of_edges())
	print(nx.number_connected_components(G_s))
	fig, ax1 = plt.subplots(1, 1, sharey = True, figsize=(14, 7))
	pos = nx.spring_layout(G_s)
	nx.draw(G_s, pos, ax1, with_labels=False, node_color = 'black', node_size = 20, width = 1)
	plt.show()
	plt.close()
    # ground_colors = [G_s.nodes[node]["ground_label"] for node in self.nodes




	for pair in result_s:
		potential += 1
		potential_set = potential_set | result_s[pair]
	print("Potential pairs = %d" % potential)
	print("Potential vertices = %d" % len(potential_set))

	for (pair, vertex_list) in result_s.items():
		if len(vertex_list) == max_s:
			max_pair = pair 
			break

	nx.set_node_attributes(G, {max_pair[0]: 0}, "label")
	nx.set_node_attributes(G, {max_pair[1]: 0}, "label")
	marked_nmbr = 0
	marked_vertices = {max_pair[0], max_pair[1]}
	passed_pairs = set()
	# print(result_s[max_pair])
	it = 0
	prev_len = 0

	while len(marked_vertices) < 500 and len(marked_vertices) > marked_nmbr:
		marked_nmbr = len(marked_vertices)
		marked_list = list(marked_vertices)
		marked_list.reverse()
		print(marked_list)
		for marked_pair in itertools.combinations(marked_list, 2):
			if marked_pair[0] > marked_pair[1]:
				rev_marked_pair = (marked_pair[1], marked_pair[0])
			else:
				rev_marked_pair = marked_pair
			if rev_marked_pair in result_s:
				for pair in result_s:
					if len(result_s[pair]) > max_d:
						if len(result_s[pair] & result_s[rev_marked_pair]) > 0 or len([pair[0]]):
							pred_labels = dict(zip(list(result_s[pair]), [0 for i in result_s[pair]]))
							nx.set_node_attributes(G, pred_labels, "label")
							marked_vertices.update(result_s[pair])
							marked_vertices.update([pair[0], pair[1]])
				if len(marked_vertices) > prev_len:
				 	print(len(marked_vertices))
				 	prev_len = len(marked_vertices)
				if len(marked_vertices) >= 500:
					break
		if len(marked_vertices) >= 500:
				break
		print("iteration = %d" % it)
		it += 1

	print(len(marked_vertices))
	one_pred = list(set(G.nodes) - marked_vertices)
	pred_labels = dict(zip(one_pred, [1 for i in one_pred]))
	nx.set_node_attributes(G, pred_labels, "label")
	prediction = [v for v in nx.get_node_attributes(G, 'label').values()]
	# print([G.nodes[node]['ground_label'] for node in G.nodes if node not in marked_vertices])
	# print([G.nodes[node]['label'] for node in G.nodes if node not in marked_vertices])
	print(max_1(accuracy_score(G.ground_labels, prediction)))

	print(it)

	return (result_s, result_d)	

	# for u in G.nodes:
	# 	max_number1 = 0
	# 	max_number2 = 0
	# 	for v in nx.non_neighbors(G, u):
	# 		cmn_nmbr = len([i for i in nx.common_neighbors(G,u,v)])
	# 		if cmn_nmbr > max_number2:
	# 			if cmn_nmbr > max_number1:
	# 				max_number1 = cmn_nmbr
	# 				if max_number1 + max_number2 > max_number:
	# 					v_max = v
	# 					u_max = u
	# 					max_number = max_number1 + max_number2
	# 			else:
	# 				max_number2 = cmn_nmbr
	# 				if max_number1 + max_number2 > max_number:
	# 					w_max = v
	# 					u_max = u
	# 					max_number = max_number1 + max_number2

	# print(max_number1, max_number2, G.nodes[v_max], G.nodes[w_max], len([i for i in nx.common_neighbors(G,G.nodes[v_max],G.nodes[w_max])]))
	# return (G.nodes[v_max]['ground_label'], G.nodes[w_max]['ground_label'])	
				

	# for (u,v) in itertools.combinations(G.nodes, 2):
	# 	if not G.has_edge(u,v) and G.nodes[u]["ground_label"] == G.nodes[v]['ground_label']:
	# 		result += [i for i in nx.common_neighbors(G,u,v)]

	# result1 = set(result)
	# return len(result1)

def khren3(G):

	result_s = {}
	result_d = {}
	passed_set = []

	list_neighbrs = {}

	for v in G.nodes:
		list_neighbrs.update({v: set(nx.neighbors(G, v))})

	for u in G.nodes:
		passed_set.append(u)
		for v in nx.neighbors(G, u):
			if not v in passed_set:
				cmn_nmbr = list_neighbrs[u] & list_neighbrs[v]
				# dist = nx.shortest_path_length(G,u,v)
				# if dist == 2:
				# cmn_nmbr = G.distance(u,v)
				if G.nodes[u]["ground_label"] == G.nodes[v]['ground_label']:
					result_s.update({(u,v): cmn_nmbr})
				else:
					result_d.update({(u,v): cmn_nmbr})

	# max_s = max(len(result_s.values()))
	min_s = len(min(result_s.values(), key = len)) 
	min_d = len(min(result_d.values(), key = len)) 
	max_d = len(max(result_d.values(), key = len)) 

	for (pair, vertex_list) in result_d.items():
		if len(vertex_list) == max_d:
			max_pair = pair 
			break

	print(min_s, min_d)


	adj_matrix = nx.adjacency_matrix(G).toarray()
	labels = [-1 for node in G.nodes]
	true_labels = [G.nodes[node]['ground_label'] for node in G.nodes]
	# labels[[0]] = 0
	labels[max_pair[0]] = 0
	labels[max_pair[1]] = 1
	# labels[0:10] = [0 for i in range(10)]
	# labels[900:910] = [1 for i in range(10)]

	lp = LabelPropagation(kernel = 'rbf', gamma=0.7, max_iter = 1000)
	lp.fit(adj_matrix, labels)
	print(lp.score(adj_matrix, true_labels))

	return (result_s, result_d)	

	# for u in G.nodes:
	# 	max_number1 = 0
	# 	max_number2 = 0
	# 	for v in nx.non_neighbors(G, u):
	# 		cmn_nmbr = len([i for i in nx.common_neighbors(G,u,v)])
	# 		if cmn_nmbr > max_number2:
	# 			if cmn_nmbr > max_number1:
	# 				max_number1 = cmn_nmbr
	# 				if max_number1 + max_number2 > max_number:
	# 					v_max = v
	# 					u_max = u
	# 					max_number = max_number1 + max_number2
	# 			else:
	# 				max_number2 = cmn_nmbr
	# 				if max_number1 + max_number2 > max_number:
	# 					w_max = v
	# 					u_max = u
	# 					max_number = max_number1 + max_number2

	# print(max_number1, max_number2, G.nodes[v_max], G.nodes[w_max], len([i for i in nx.common_neighbors(G,G.nodes[v_max],G.nodes[w_max])]))
	# return (G.nodes[v_max]['ground_label'], G.nodes[w_max]['ground_label'])	
				

	# for (u,v) in itertools.combinations(G.nodes, 2):
	# 	if not G.has_edge(u,v) and G.nodes[u]["ground_label"] == G.nodes[v]['ground_label']:
	# 		result += [i for i in nx.common_neighbors(G,u,v)]

	# result1 = set(result)
	# return len(result1)

def max_neighbourhood(G):

	def interval_u(x):
		return np.exp(special.lambertw((1/x-1)/np.exp(1)) + 1)
	threshold = 2*G.b*interval_u(2*G.b)*np.log(G.n_1+G.n_2)
	print(threshold)

	list_neighbrs = {}
	for v in G.nodes:
		list_neighbrs.update({v: set(nx.neighbors(G, v))})

	current_node = choice(list(G))
	passed_set = [current_node]
	nodes_on_step = []
	update_flg = 1

	while len(passed_set) < G.n_1 and update_flg == 1:
		nodes_on_step = []
		for u in list(set(list_neighbrs[current_node]) - set(passed_set)):
			cmn_nmbr = list_neighbrs[u] & list_neighbrs[current_node]
			if len(cmn_nmbr) > threshold:
				nodes_on_step.append(u)
				passed_set.append(u)

		if len(nodes_on_step) > 0:
			list_cmn_neighb = {}
			for u in nodes_on_step:
				list_cmn_neighb.update({u: len(list_neighbrs[u] & list_neighbrs[current_node])})
			current_node = min(list_cmn_neighb, key=list_cmn_neighb.get)
		else:
			update_flg = 0

		# print(passed_set)
		res = [G.nodes[u]['ground_label'] for u in passed_set]
		print("Length = %d, avg_label = %.2f" % (len(res), sum(res)/len(res)))

	print(len(res))
	print(sum(res)/len(res))
	
	return [G.nodes[u]['coordinate'] for u in passed_set]
	
	# for u in G.nodes:
	# 	passed_set.append(u)
	# 	for v in nx.non_neighbors(G, u):
	# 		if not v in passed_set:
	# 			cmn_nmbr = list_neighbrs[u] & list_neighbrs[v]
	# 			# dist = nx.shortest_path_length(G,u,v)
	# 			# if dist == 2:
	# 			# cmn_nmbr = G.distance(u,v)
	# 			if G.nodes[u]["ground_label"] == G.nodes[v]['ground_label']:
	# 				result_s.update({(u,v): cmn_nmbr})
	# 			else:
	# 				result_d.update({(u,v): cmn_nmbr})

	# # max_s = max(len(result_s.values()))
	# max_s = len(max(result_s.values(), key = len)) 
	# max_d = len(max(result_d.values(), key = len)) 
	# print(max_s, max_d)

	# potential = 0
	# potential_set = set()
	# result_s = {key:val for key, val in result_s.items() if len(val) > max_d}

	# G_s = G.subgraph(list(G.nodes))
	# edges = list(G_s.edges())
	# G_s.remove_edges_from(edges)

	# for (u,v) in result_s:
	# 	G_s.add_edge(u,v)

	# print(G_s.number_of_edges())
	# print(nx.number_connected_components(G_s))
	# fig, ax1 = plt.subplots(1, 1, sharey = True, figsize=(14, 7))
	# pos = nx.spring_layout(G_s)
	# nx.draw(G_s, pos, ax1, with_labels=False, node_color = 'black', node_size = 20, width = 1)
	# plt.show()
	# plt.close()
 #    # ground_colors = [G_s.nodes[node]["ground_label"] for node in self.nodes




	# for pair in result_s:
	# 	potential += 1
	# 	potential_set = potential_set | result_s[pair]
	# print("Potential pairs = %d" % potential)
	# print("Potential vertices = %d" % len(potential_set))

	# for (pair, vertex_list) in result_s.items():
	# 	if len(vertex_list) == max_s:
	# 		max_pair = pair 
	# 		break

	# nx.set_node_attributes(G, {max_pair[0]: 0}, "label")
	# nx.set_node_attributes(G, {max_pair[1]: 0}, "label")
	# marked_nmbr = 0
	# marked_vertices = {max_pair[0], max_pair[1]}
	# passed_pairs = set()
	# # print(result_s[max_pair])
	# it = 0
	# prev_len = 0

	# while len(marked_vertices) < 500 and len(marked_vertices) > marked_nmbr:
	# 	marked_nmbr = len(marked_vertices)
	# 	marked_list = list(marked_vertices)
	# 	marked_list.reverse()
	# 	print(marked_list)
	# 	for marked_pair in itertools.combinations(marked_list, 2):
	# 		if marked_pair[0] > marked_pair[1]:
	# 			rev_marked_pair = (marked_pair[1], marked_pair[0])
	# 		else:
	# 			rev_marked_pair = marked_pair
	# 		if rev_marked_pair in result_s:
	# 			for pair in result_s:
	# 				if len(result_s[pair]) > max_d:
	# 					if len(result_s[pair] & result_s[rev_marked_pair]) > 0 or len([pair[0]]):
	# 						pred_labels = dict(zip(list(result_s[pair]), [0 for i in result_s[pair]]))
	# 						nx.set_node_attributes(G, pred_labels, "label")
	# 						marked_vertices.update(result_s[pair])
	# 						marked_vertices.update([pair[0], pair[1]])
	# 			if len(marked_vertices) > prev_len:
	# 			 	print(len(marked_vertices))
	# 			 	prev_len = len(marked_vertices)
	# 			if len(marked_vertices) >= 500:
	# 				break
	# 	if len(marked_vertices) >= 500:
	# 			break
	# 	print("iteration = %d" % it)
	# 	it += 1

	# print(len(marked_vertices))
	# one_pred = list(set(G.nodes) - marked_vertices)
	# pred_labels = dict(zip(one_pred, [1 for i in one_pred]))
	# nx.set_node_attributes(G, pred_labels, "label")
	# prediction = [v for v in nx.get_node_attributes(G, 'label').values()]
	# # print([G.nodes[node]['ground_label'] for node in G.nodes if node not in marked_vertices])
	# # print([G.nodes[node]['label'] for node in G.nodes if node not in marked_vertices])
	# print(max_1(accuracy_score(G.ground_labels, prediction)))

	# print(it)

	# return (result_s, result_d)	

	# for u in G.nodes:
	# 	max_number1 = 0
	# 	max_number2 = 0
	# 	for v in nx.non_neighbors(G, u):
	# 		cmn_nmbr = len([i for i in nx.common_neighbors(G,u,v)])
	# 		if cmn_nmbr > max_number2:
	# 			if cmn_nmbr > max_number1:
	# 				max_number1 = cmn_nmbr
	# 				if max_number1 + max_number2 > max_number:
	# 					v_max = v
	# 					u_max = u
	# 					max_number = max_number1 + max_number2
	# 			else:
	# 				max_number2 = cmn_nmbr
	# 				if max_number1 + max_number2 > max_number:
	# 					w_max = v
	# 					u_max = u
	# 					max_number = max_number1 + max_number2

	# print(max_number1, max_number2, G.nodes[v_max], G.nodes[w_max], len([i for i in nx.common_neighbors(G,G.nodes[v_max],G.nodes[w_max])]))
	# return (G.nodes[v_max]['ground_label'], G.nodes[w_max]['ground_label'])	
				

	# for (u,v) in itertools.combinations(G.nodes, 2):
	# 	if not G.has_edge(u,v) and G.nodes[u]["ground_label"] == G.nodes[v]['ground_label']:
	# 		result += [i for i in nx.common_neighbors(G,u,v)]

	# result1 = set(result)
	# return len(result1)


class cmn_nbr_ssl_algo:
	def cmn_nbr_ssl_algo_analysis(self, G, eta):
		N = nx.number_of_nodes(G)
		labeled_set = random.choices(list(G), k = int(N*self.eta))
		unlabeled_set = list(set(list(G)) - set(labeled_set))
		for v in labeled_set:
			G.nodes[v]['label'] = G.nodes[v]['ground_label']

		list_labeled_neighbrs = {}
		for v in labeled_set:
			list_labeled_neighbrs.update({v: set(G.neighbors(v)) & set(labeled_set) })

		# c0 = []
		# c1 = []
		# for v in labeled_set:
		# 	if G.nodes[v]['label'] == 0:
		# 		c0 += [G.nodes[v]['coordinate']]
		# 	else:
		# 		c1 += [G.nodes[v]['coordinate']]
		# c0.sort()
		# c1.sort()
		# print(c0)
		# print(c1)

		for u in unlabeled_set:
			p_1 = 0
			p_0 = 0
			for (v,w) in itertools.combinations(set([v for v in G.neighbors(u)]) & set(labeled_set), 2):
				# print(v,w)
				if G.nodes[v]['label'] == 0 and G.nodes[w]['label'] == 0:
					if len(list_labeled_neighbrs[v] & list_labeled_neighbrs[w]) == 0:
						p_0 += 1
					elif sum([G.nodes[z]['label'] for z in (list_labeled_neighbrs[v] & list_labeled_neighbrs[w])])/ len(list_labeled_neighbrs[v] & list_labeled_neighbrs[w]) <= 0:
						p_0 += 1
				if G.nodes[v]['label'] == 1 and G.nodes[w]['label'] == 1:
					if len(list_labeled_neighbrs[v] & list_labeled_neighbrs[w]) == 0: 
						p_1 += 1
					elif sum([G.nodes[z]['label'] for z in (list_labeled_neighbrs[v] & list_labeled_neighbrs[w])]) / len(list_labeled_neighbrs[v] & list_labeled_neighbrs[w]) >= 1:
						p_1 += 1
			if p_1 > p_0:
				G.nodes[u]['label'] = 1
			else:
				G.nodes[u]['label'] = 0
			# print(G.nodes[u]['ground_label'], p_0, p_1, [G.nodes[v]['label'] for v in (set([v for v in G.neighbors(u)]) & set(labeled_set))], G.nodes[u]['coordinate'], [G.nodes[v]['coordinate'] for v in (set([v for v in G.neighbors(u)]) & set(labeled_set))])

		labels_pred = [G.nodes[v]['label'] for v in unlabeled_set]
		self.accuracy = max(accuracy_score(labels_pred, G.subgraph(unlabeled_set).ground_labels), 1 - accuracy_score(labels_pred, G.subgraph(unlabeled_set).ground_labels))
		self.prediction = labels_pred
		# print(self.accuracy)


	def __init__(self, G, eta = 0.05):
		self.accuracy = 0
		self.eta = eta 
		self.cmn_nbr_ssl_algo_analysis(G, self.eta)

def neg(x):
	if x == 1:
		return 0
	else:
		return 1

class Common_neigbours_labeling:
	def Common_neigbours_labeling_analysis(self, G, labeled_set):
		N = nx.number_of_nodes(G)
		unlabeled_set = list(set(list(G)) - set(labeled_set))

		cluster = {}
		cluster.update({0: set([x for x,y in G.nodes(data = True) if y['label'] ==0])})
		cluster.update({1: set([x for x,y in G.nodes(data = True) if y['label'] ==1])})

		list_neighbrs = {}
		for v in labeled_set:
			list_neighbrs.update({ v: set(G.neighbors(v)) & cluster[neg(G.nodes[v]['label'])] })

		list_cmn_neighb = {}
		for (v,w) in itertools.combinations(cluster[0], 2):
			list_cmn_neighb.update({(v,w): len(list_neighbrs[v] & list_neighbrs[w])}) 
		for (v,w) in itertools.combinations(cluster[1], 2):
			list_cmn_neighb.update({(v,w): len(list_neighbrs[v] & list_neighbrs[w])}) 

		# print(list_cmn_neighb[(1, 128)])
		i = 0
		shuffled_nodes = list(G.nodes)
		random.shuffle(shuffled_nodes)
		relabeled_amt = 0 
		zero_amt = 0
		good_points = []
		bad_points = []
		good_set = []
		for u in shuffled_nodes:
			p_1 = 0
			p_0 = 0
			for (v,w) in itertools.combinations(sorted(set(G.neighbors(u)) & cluster[0]), 2):
				# print(v,w)
				if list_cmn_neighb[(v,w)] <= 1:
					p_0 += 1
			for (v,w) in itertools.combinations(sorted(set(G.neighbors(u)) & cluster[1]), 2):
				# print(v,w)
				if list_cmn_neighb[(v,w)] <= 1:
					p_1 += 1
			if p_1 > p_0:
				if G.nodes[u]['label'] == 0:
					relabeled_amt += 1 
				G.nodes[u]['label'] = 1
			if p_1 < p_0:
				if G.nodes[u]['label'] == 1:
					relabeled_amt += 1 
				G.nodes[u]['label'] = 0
			if p_1 == p_0:
				zero_amt += 1
			i+= 1
			# if i < 50:
			# 	print(G.nodes[u]['label'], G.nodes[u]['ground_label'], p_0, p_1, len(set(G.neighbors(u)) & cluster[0]), len(set(G.neighbors(u)) & cluster[1]))
				# print(G.nodes[u]['ground_label'], p_0, p_1, [G.nodes[v]['label'] for v in (set([v for v in G.neighbors(u)]) & set(labeled_set))], G.nodes[u]['coordinate'], [G.nodes[v]['coordinate'] for v in (set([v for v in G.neighbors(u)]) & set(labeled_set))])
			if p_1 + p_0 > 800:
				good_set += [u]
			if G.nodes[u]['label'] == G.nodes[u]['ground_label']:
				good_points += [p_1 + p_0]
			else: 
				bad_points += [p_1 + p_0]

		self.good_points = good_points
		self.bad_points = bad_points

		adj_matrix = nx.adjacency_matrix(G).toarray()
		labels = [-1 for node in G.nodes]
		for v in good_set:
			labels[v] = G.nodes[v]['label'] 
		true_labels = [G.nodes[node]['ground_label'] for node in G.nodes]

		lp = LabelSpreading(kernel = 'knn', n_neighbors = 20, max_iter = 1000, alpha = 0.8)
		lp.fit(adj_matrix, labels)
		print("LP score = %.3f" % max(lp.score(adj_matrix, true_labels), 1- lp.score(adj_matrix, true_labels)))

		labels_pred = [G.nodes[v]['label'] for v in G.nodes]
		self.accuracy = max(accuracy_score(labels_pred, G.ground_labels), 1 - accuracy_score(labels_pred, G.ground_labels))
		self.prediction = labels_pred
		print(self.accuracy)
		print(relabeled_amt, zero_amt)


	def __init__(self, G, labeled_set):
		self.accuracy = 0
		self.Common_neigbours_labeling_analysis(G, labeled_set)

class Spectral_Clustering_analysis:
	def __init__(self, n = 2000, a = 15, b = 6, n_clusters = 2, portion = 0.01, n_iter = 10):
		iters = []
		spectrum = []
		accs = []
		number_of_edges = []

		for iter in tqdm.tqdm(range(n_iter)):
			G = GBM_graph(n_1 = int(n/2), n_2 = int(n/2), a = a, b = b, disp=False)
			# print(len(G.edges))
			# G.remove_edges_from([(1,i) for i in range(2,n)])
			# print(len(G.edges))
			laplacian_matrix = nx.normalized_laplacian_matrix(G)
			vals, vecs = sparse.linalg.eigs(laplacian_matrix.asfptype() , k=int(portion * (G.n_1 + G.n_2)), which = 'SM')
			ground_labels = nx.get_node_attributes(G, 'ground_label')
			optimal_val = 2*G.b/(G.a + G.b)

			for i in range(1, int(portion * n)):
				vector = vecs[:,i]
				vector = vector.astype('float64')
				labels_pred_spectral = checkSign(vector)
				accuracy = max(accuracy_score(labels_pred_spectral, G.ground_labels), 1 - accuracy_score(labels_pred_spectral, G.ground_labels))
				accs += [accuracy]
				iters += [iter]
				spectrum += [vals[i]]
				cluster0 = [j for j in range(n) if labels_pred_spectral[j] == 0]
				number_of_edges += [nx.cut_size(G,cluster0)]
			
				# if i >= 11 and i <= 17:
				# 	labels_dict = dict(zip(list(G.nodes), labels_pred_spectral))
				# 	nx.set_node_attributes(G, labels_dict, "label")
				# 	sns.distplot([G.nodes[node]["coordinate"] for node in G.nodes if G.nodes[node]['label'] == 0], label = "Cluster 0", kde = False, bins = 50)
				# 	sns.distplot([G.nodes[node]["coordinate"] for node in G.nodes if G.nodes[node]['label'] == 1], label = "Cluster 1", kde = False, bins = 50)
				# 	plt.title("i = " + str(i))
				# 	plt.show()
				# if i >= 11 and i <= 13:
				# 	print(vector[:20])
				# 	coordinates = [G.nodes[node]["coordinate"] for node in G.nodes]
				# 	plt.scatter(coordinates, vector, marker='o', facecolors='none', edgecolors='b')
				# 	plt.show()
			
			# closest_vectors = [vecs[:,i] for i in range(1, int(portion * n)) if val[i] > optimal_val - 0.01 and val[i] < optimal_val + 0.05]
			closest_vectors = [vecs[:,i] for i in range(13, 15)]
			labels_pred_spectral = k_means(closest_vectors, n)
			accuracy = max(accuracy_score(labels_pred_spectral, G.ground_labels), 1 - accuracy_score(labels_pred_spectral, G.ground_labels))
			print(accuracy)


		self.n_edges = number_of_edges
		self.iters = iters
		self.spectrum = spectrum
		self.accs = accs

		sns.set()
		plt.rcParams['figure.figsize'] = [14, 7]

		plt.scatter(self.spectrum, self.iters, marker='o', facecolors='none', edgecolors='b')
		plt.axvline(x = float(12/21), linewidth = 2, color='black')
		plt.xlabel(r"$spectrum$")
		plt.ylabel(r"$iterations$")
		plt.show()

		cmap = plt.get_cmap("tab10")
		for i in range(5):
		    spectrum = [self.spectrum[j] for j in range(len(self.iters)) if self.iters[j] == i]
		    accuracy = [self.accs[j] for j in range(len(self.iters)) if self.iters[j] == i]
		#     plt.scatter(spectrum, accuracy, marker='o', facecolors='none', edgecolors = cmap(i))
		    plt.plot(spectrum, accuracy, marker='o', label = 'Iteration ' + str(i))
		plt.axvline(x = float(12/21), linewidth = 2, color='black')
		plt.xlabel(r"$spectrum$")
		plt.ylabel(r"$accuracy$")
		plt.legend()
		plt.title("Accuracy vs spectrum for n = " + str(n) + ", a = " + str(a) + ", b = " + str(b))
		plt.show()

def k_means(eigenvectors, n):
	m = len(eigenvectors)
	X = np.real(np.array([[v[i] for v in eigenvectors] for i in range(n)]))

	# if m == 2:
	# 	plt.scatter(eigenvectors[0], eigenvectors[1], facecolors = 'none', edgecolors = 'b')
	# 	plt.show()

	kmeans = KMeans(n_clusters=2, random_state=42).fit(X)
	c0 = kmeans.cluster_centers_[0]
	c1 = kmeans.cluster_centers_[1]
	norm_vector = np.array((c1 - c0)/np.linalg.norm(c1 - c0, ord = 2))
	p = sum(c1 + c0)/(2*np.linalg.norm(c1 - c0, ord = 2))
	# print(n)
	# print(p)
	# print(np.dot(n,X[0]))

	dists = [abs(np.dot(norm_vector,X[i]) + p) for i in range(n)]
	# print(min(dists))
	# print(sum(dists))
	# print("Clusters centers:")
	# print(kmeans.cluster_centers_)
	# print(np.linalg.norm(kmeans.cluster_centers_, ord = 2))
	# print(np.linalg.norm(sum(kmeans.cluster_centers_)))
	return {"labels": kmeans.labels_, "centers": kmeans.cluster_centers_, "inertia": kmeans.inertia_, "dists": dists}

class k_means_analysis:
	def __init__(self, G, n = 2000, portion = 0.01, vectors = [13], spectrum_disp = False, cut_disp = False, c_norm_disp = False, k = 0):
		iters = []
		spectrum = []
		accs = []
		number_of_edges = []
		n = G.n_1 + G.n_2

		laplacian_matrix = nx.normalized_laplacian_matrix(G)
		vals, vecs = sparse.linalg.eigs(laplacian_matrix.asfptype() , k=int(portion * (G.n_1 + G.n_2)), which = 'SM')
		ground_labels = nx.get_node_attributes(G, 'ground_label')
		optimal_val = 2*G.b/(G.a + G.b)

		if spectrum_disp:
			sns.set()
			plt.rcParams['figure.figsize'] = [14, 7]

			plt.scatter(vals, [1 for i in range(len(vals))], marker='o', facecolors='none', edgecolors='b')
			plt.axvline(x = optimal_val, linewidth = 2, color='black')
			plt.xlabel(r"spectrum")
			plt.ylabel(r"iterations")
			plt.show()

		if cut_disp:
			for i in vectors:
				vector = vecs[:,i]
				vector = vector.astype('float64')
				labels_pred_spectral = checkSign(vector)
				accuracy = max(accuracy_score(labels_pred_spectral, G.ground_labels), 1 - accuracy_score(labels_pred_spectral, G.ground_labels))
				accs += [accuracy]
				labels_dict = dict(zip(list(G.nodes), labels_pred_spectral))
				nx.set_node_attributes(G, labels_dict, "label")

				sns.distplot(vector, kde = False, bins = 50)
				plt.show()

				sns.distplot([G.nodes[node]["coordinate"] for node in G.nodes if G.nodes[node]['label'] == 0], label = "Cluster 0", kde = False, bins = 50)
				sns.distplot([G.nodes[node]["coordinate"] for node in G.nodes if G.nodes[node]['label'] == 1], label = "Cluster 1", kde = False, bins = 50)
				plt.title("i = " + str(i) + ", eigenvalue = " + str(vals[i]) + ", accuracy = " + str(accuracy))
				plt.show()

		if c_norm_disp:
			accs = []
			c_norms = []
			spectra = []
			for i in vectors:
				vector = vecs[:,i]
				vector = vector.astype('float64')
				km = k_means([vector], n)
				accuracy = max(accuracy_score(km['labels'], G.ground_labels), 1 - accuracy_score(km['labels'], G.ground_labels))
				accs += [accuracy]
				spectra += [vals[i]] 
				c_norms += [np.linalg.norm(sum(km['centers']))]

			plt.plot(vectors, accs, marker='o', label = 'Iteration ' + str(i))
			plt.xlabel("Order of eigenvector")
			plt.ylabel("Accuracy")
			plt.show()
			plt.plot(vectors, c_norms, marker='o', label = 'Iteration ' + str(i))
			plt.show()
			plt.plot(vectors, spectra, marker='o')
			plt.axhline(y = optimal_val, linewidth = 2, color='black')
			plt.show()

		if k > 0:
			k = min(k, len(vectors))
			accs = []
			c_norms = []
			inerts = []
			min_dists = []
			sum_dists = []
			balances = []

			for j in range(1,k+1):
				# print([vectors[i] for i in range(j)])
				# km_vectors = [vecs[:,vectors[1]], vecs[:,vectors[j]]]
				km_vectors = [vecs[:,vectors[i]] for i in range(j)]
				km = k_means(km_vectors, n)
				accuracy = max(accuracy_score(km['labels'], G.ground_labels), 1 - accuracy_score(km['labels'], G.ground_labels))
				accs += [accuracy]
				c_norms += [np.linalg.norm(sum(km['centers']), ord = 2)]
				inerts += [km['inertia']/j]
				min_dists += [min(km['dists'])]
				sum_dists += [sum(km['dists'])]
				balances += [abs(sum(km['labels']) - n/2)]

			plt.plot(vectors[:k], accs, marker='o')
			plt.show()
			plt.plot(vectors[:k], c_norms, marker='o', color = 'red')
			plt.show()
			# plt.plot(vectors[:k], min_dists, marker='o', color = 'red')
			# plt.show()		
			# plt.plot(vectors[:k], sum_dists, marker='o', color = 'green')
			# plt.show()		
			plt.plot(vectors[:k], balances, marker='o', color = 'orange')
			plt.show()		


		k_means_vectors = [vecs[:,i] for i in vectors]
		labels_k_means = k_means(k_means_vectors, n)['labels']
		# print(k_means(k_means_vectors, n)['labels'][:100])
		accuracy = max(accuracy_score(labels_k_means, G.ground_labels), 1 - accuracy_score(labels_k_means, G.ground_labels))
		print("Total accuracy after k-means = %.3f" % accuracy)

		self.n_edges = number_of_edges
		self.spectrum = vals
		self.accs = accs
		self.accuracy = accuracy

class Spectral_k_means:
	def __init__(self, G, portion = 0.02):
		n = G.n_1 + G.n_2 
		optimal_val = 2*G.b/(G.a + G.b)
		step = 2*(G.a**2 + G.b**2)*np.log(n)/(G.a+G.b)/n

		laplacian_matrix = nx.normalized_laplacian_matrix(G)
		vals, vecs = sparse.linalg.eigs(laplacian_matrix.asfptype() , k=int(portion * n), which = 'SM')
		optimal_val = 2*G.b/(G.a + G.b)

		optimal_vectors = [vecs[:,i] for i in range(int(portion*n)) if vals[i] > optimal_val - 2*step/3 and vals[i] < optimal_val + step]
		print([i for i in range(int(portion*n)) if vals[i] > optimal_val - 2*step/3 and vals[i] < optimal_val + step])
		k = len(optimal_vectors)

		min_c_norm = 1
		min_sum_labels = n
		j_min = -1
		for j in range(1,k+1):
			# for subset in itertools.combinations(range(1,min(8,k+1)), j):
			km = k_means(optimal_vectors[:j+1], n)
			# if np.linalg.norm(sum(km['centers']), ord = 2) < min_c_norm:
			# 	min_c_norm = np.linalg.norm(sum(km['centers']), ord = 2)
			# 	labels_pred = km['labels']
			# 	j_min = j

			if abs(sum(km['labels']) - n/2) < min_sum_labels:
				min_sum_labels = abs(sum(km['labels']) - n/2)
				labels_pred = km['labels']
				j_min = j
				# subset_min = subset 
		
			if min_sum_labels <= 0:
				break

		accuracy = max(accuracy_score(labels_pred, G.ground_labels), 1 - accuracy_score(labels_pred, G.ground_labels))
		print("Total accuracy after k-means = %.3f" % accuracy)
		print("Took %d vectors" % (j_min + 1))
		print("metric_min = %d" % min_sum_labels)
		# print("Optimal vectors: ")
		# print(subset_min)

		self.accuracy = accuracy