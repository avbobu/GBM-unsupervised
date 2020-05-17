
# Author - Andrei Bobu
# Last updated 22 November 2019
# This code provides several algorithms that make clustering on GBM model 


import pandas as pd
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cluster import	SpectralClustering
from sklearn.semi_supervised import LabelPropagation
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
    	new_G.adj_matrix = nx.adjacency_matrix(new_G)
    	new_G.ground_labels = [new_G.nodes[node]["ground_label"] for node in new_G.nodes]

    	return new_G 

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
        
        # Add edges depending on the distance between nodes and their communities 
        for i in range(len(self.nodes)):
            for j in range(i+1, len(self.nodes)):
                if(self.nodes[i]["ground_label"] == self.nodes[j]["ground_label"] and 
                  self.distance(i,j) < self.r_in):
                    self.add_edge(i,j)
                if(self.nodes[i]["ground_label"] != self.nodes[j]["ground_label"] and 
                  self.distance(i,j) < self.r_out):
                    self.add_edge(i,j)

        # Define some useful attributes
        self.av_deg_in = 2 * self.r_in * n_1
        self.av_deg_out = 2 * self.r_out * n_1
        self.adj_matrix = nx.adjacency_matrix(self)
        self.ground_labels = [self.nodes[node]["ground_label"] for node in self.nodes]

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

def eigenvectorAnalysis (adjacencyMatrix):
	    #detect communities from the sign of the second eigenvector of the adjacency matrix
	    vals, vecs = sparse.linalg.eigs(adjacencyMatrix.asfptype() , k=2)
	    secondVector = vecs[:,1]
	    secondVector = secondVector.astype('float64')
	    labels_pred_spectral = checkSign(secondVector)
	    return labels_pred_spectral

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

def simulation(algorithm, n_1 = 100, n_2 = 100, a = 1, b = 1, n_trials = 1, param_flg = 0):
	avg_accuracy = 0
	for i in range(n_trials):
	    G = GBM_graph(n_1, n_2, a, b, disp = False)
	    if algorithm.__name__ == "Partition":
	    	d = 0.75*b + a**2/(4*b)	
	    	r = d * np.log(n_1)/n_1
	    	n_clusters = int(max(1,1/r))
	    	A = algorithm(G, n_clusters = n_clusters)
	    else: 
	    	A = algorithm(G)
	    avg_accuracy += A.accuracy / n_trials
	return avg_accuracy

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

	for a in np.arange(a_start, a_finish, a_step):
		cur_step = {"a": a}
		s = 'a = ' + str(a) + ', b = ' + str(b) + ', '
		for A in algo_list: 
			start_time = time.time() 
			acc = simulation(A, n_1, n_2, a, b, n_trials) 
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
