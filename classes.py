
# Author - Andrei Bobu
# Last updated 17 May 2020
# This code provides several algorithms that make clustering on the GBM model 


import pandas as pd
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cluster import SpectralClustering
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.cluster import KMeans
from scipy import sparse 
from scipy import special
from scipy import optimize
import copy
import tqdm
import itertools
import time
import seaborn as sns 
from random import choice

### The basic class with a GBM random realization 

class GBM_graph(nx.Graph):

	# Procedure to generate points in 1-dimensional torus

	def add_node(self, ground_label):
		coordinate = np.random.uniform(0,1)
		super().add_node(len(self.nodes), coordinate = coordinate, ground_label=ground_label, label = 0)
	
	# Function returns the distance between vertices i and j

	def distance(self, i,j):
		return min(abs(self.nodes[i]["coordinate"] - self.nodes[j]["coordinate"]), 1 - abs(self.nodes[i]["coordinate"] - self.nodes[j]["coordinate"]))

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

	def __init__(self, n_1=1, n_2=1, a=1, b=1, eta = 0, disp = False):
		super().__init__()

		# Establish the basic parameters of the model
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

		# self.ground_labels = [self.nodes[node]["ground_label"] for node in self.nodes]
		self.ground_labels = list(nx.get_node_attributes(self, 'ground_label').values())

		# A fast way to create the edges in a graph
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

	def GetAccuracy(self, labels_pred):
		if isinstance(labels_pred, list) or isinstance(labels_pred, np.ndarray):
			return max_1(accuracy_score(labels_pred, self.ground_labels))

		if isinstance(labels_pred, dict):
			ground_labels_dict = nx.get_node_attributes(self, 'ground_label')
			ground_labels_dict = {x: ground_labels_dict[x] for x in labels_pred.keys()}
			return max_1(accuracy_score( list(labels_pred.value()), list(ground_labels_dict.value()) ))

### Useful functions

# Distance on the torus [0,1[

def torusDistance(x,y):
	return min(abs(x-y), 1-abs(x-y))

def max_1(x):
	return max(x, 1-x)

def checkSign (vector):
	labels_pred = np.zeros(len(vector))
	for i in range(len(vector)):
		if (vector[i]<0):
			labels_pred[i] = 1
	return labels_pred

def eigenvectorAnalysis (matrix, eig_max_order, val):
		#detect communities from the sign of the second eigenvector of the adjacency matrix
		vals, vecs = sparse.linalg.eigs(matrix.asfptype() , k=eig_max_order, which = 'SM')
		idx = min_list(vals, val)
		secondVector = vecs[:,idx]
		secondVector = secondVector.astype('float64')
		labels_pred_spectral = checkSign(secondVector)
		return labels_pred_spectral

def neg(x):
	if x == 0:
		return 1
	else:
		return 0

def interval_u(x):
		return np.exp(special.lambertw((1/x-1)/np.exp(1)) + 1)

def k_means(eigenvectors, n):
	m = len(eigenvectors)
	X = np.real(np.array([[v[i] for v in eigenvectors] for i in range(n)]))

	# if m == 2:
	#   plt.scatter(eigenvectors[0], eigenvectors[1], facecolors = 'none', edgecolors = 'b')
	#   plt.show()

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

#### The new algorithm of guys from UMASS

class Motif_Counting_second_paper:
	def analysis(self, G, a, b):
		# recall  rs = a logn /n, rd = b log n / n
		t1 = optimize.bisect(f1, 0, 2*b, maxiter=5000, args = np.array([b]) )
		t2 = optimize.bisect(f2, 0, 2*b - 0.1, args = np.array([b]) , maxiter = 5000 )
		#t2 = optimize.newton(g1, b, maxiter=5000,  args = np.array([b]) ) #other method

		n = G.number_of_nodes()
		Es = (2*b + t1) * np.log(n) / n
		Ed = (2*b - t2) * np.log(n) / n
		Gc = G.subgraph(G.nodes)
		count = 0
		for edge in G.edges:
			if not process(G, edge, Es, Ed):
				Gc.remove_edge(*edge)
				count = count + 1
		
		# print( "Number of edges removed : ", count )
		
		labels_pred = np.zeros(n, dtype = int)
		k=0
		# print("The remaining graph is connected : ", nx.is_connected(Gc) )
		# print("Number of connected components : ", nx.number_connected_components(Gc) )
		for connected_component in nx.connected_components(Gc):
			#print("Connected component number ", k)
			for node in connected_component:
				labels_pred[node] = k
			k = k+1
		
		self.labels = labels_pred
		self.accuracy = G.GetAccuracy(labels_pred)
		# print("Accuracy = %.3f" % self.accuracy)

	def __init__(self, G):
		self.accuracy = 0
		self.analysis(G, G.a, G.b)

def process(G, edge, Es, Ed):
	count = calcmot1(G, edge)
	if (count/G.number_of_nodes() >= Es or count/G.number_of_nodes() <= Ed):
		return True
	else:
		return False
	
def f1(t,b):
	return (2*b+t) * np.log( (2*b+t)/(2*b) ) - t - 1

def f2(t,b):
	return (2*b-t) * np.log((2*b-t)/(2*b)) + t - 1

def g1(t,b):
	return (2*b+t)*np.log( (2*b+t) / (2*b) ) - 1

def g2(t,b):
	return 1 - (2*b-t)* np.log((2*b+t) / (2*b))

### New algorithm based on the expansion of the interval around the points

class Expansion_algorithm:

	def analysis(self, G, a, b):
		n = G.number_of_nodes()
		threshold = 2*G.b*interval_u(2*G.b)*np.log(n)
		# print(threshold)

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
			
		labels_pred = np.zeros(n)
		for i in passed_set:
			labels_pred[i] = 1
		
		self.labels = labels_pred
		self.accuracy = G.GetAccuracy(labels_pred)
		# print(len(passed_set))
		# print(self.accuracy) 

	def __init__(self, G):
		self.accuracy = 0
		self.analysis(G, G.a, G.b)

### Spectral clustering + k-means in the unsupervised mode

class Spectral_k_means:
	def analysis(self, G, a, b, portion):
		n = G.number_of_nodes()
		optimal_val = 2*b/(a + b)
		step = 2*(a**2 + b**2)*np.log(n)/(a+b)/n

		laplacian_matrix = nx.normalized_laplacian_matrix(G)
		vals, vecs = sparse.linalg.eigs(laplacian_matrix.asfptype(), k=int(portion * n), which = 'SM')
		optimal_val = 2*b/(a + b)

		def condition(i):
			return vals[i] > optimal_val - step / 2 and vals[i] < optimal_val + step

		optimal_vectors = [vecs[:, i] for i in range(int(portion * n)) if condition(i)]
		k = len(optimal_vectors)

		min_sum_labels = n
		j_min = -1
		for j in range(1, k+1):
			km = k_means(optimal_vectors[:j+1], n)
			if abs(sum(km['labels']) - n/2) < min_sum_labels:
				min_sum_labels = abs(sum(km['labels']) - n/2)
				labels_pred = km['labels']
				# j_min = j
				# subset_min = subset 

			if min_sum_labels <= 0:
				break

		accuracy = G.GetAccuracy(labels_pred)
		# print("Total accuracy after k-means = %.3f" % accuracy)
		# print("Took %d vectors" % (j_min + 1))
		# print("metric_min = %d" % min_sum_labels)
		# print("Optimal vectors: ")
		# print(subset_min)

		self.accuracy = accuracy
		self.labels = labels_pred

	def __init__(self, G, portion = 0.02):
		self.accuracy = 0
		self.analysis(G, G.a, G.b, portion)


#### SSL algorithm from my theoretical draft

class Common_neigbours_labeling:
	def analysis(self, G, labeled_set):
		unlabeled_set = set(list(G)) - set(labeled_set)
		labeled_set0 = set([v for v in labeled_set if G.nodes[v]['label'] == 0])
		labeled_set1 = set([v for v in labeled_set if G.nodes[v]['label'] == 1])

		# Neighbours in the opposite community (only for labeled nodes)

		list_neighbrs = {}
		for v in labeled_set0:
			list_neighbrs.update({ v: set(G.neighbors(v)) & labeled_set1})
		for v in labeled_set1:
			list_neighbrs.update({ v: set(G.neighbors(v)) & labeled_set0})

		# Number of common neighbours in the opposite community (only for pairs of labeled nodes)

		list_cmn_neighb = {}
		for (v,w) in itertools.combinations(sorted(labeled_set0), 2):
			list_cmn_neighb.update({(v,w): len(list_neighbrs[v] & list_neighbrs[w])}) 
		for (v,w) in itertools.combinations(sorted(labeled_set1), 2):
			list_cmn_neighb.update({(v,w): len(list_neighbrs[v] & list_neighbrs[w])}) 

		# A step of the algorithm is one unlabeled node

		for u in unlabeled_set:
			# p0 and p1 correspond to p1 and p2 from the draft
			p1 = 0
			p0 = 0

			# Count the number of labeled nodes from V_0 without common neighbours in V_1 
			for (v,w) in itertools.combinations(sorted(set(G.neighbors(u)) & labeled_set0), 2):
				if list_cmn_neighb[(v,w)] <= 0:
					p0 += 1

			# Count the number of labeled nodes from V_1 without common neighbours in V_0 
			for (v,w) in itertools.combinations(sorted(set(G.neighbors(u)) & labeled_set1), 2):
				if list_cmn_neighb[(v,w)] <= 1:
					p1 += 1

			# Take the decision like in the draft
			if p1 > p0:
				G.nodes[u]['label'] = 1
			if p1 < p0:
				G.nodes[u]['label'] = 0
			if p1 == p0:
				G.nodes[u]['label'] = random.randint(0,1) 
		
		# Check the accuracy
		labels_pred = [G.nodes[v]['label'] for v in G.nodes]
		self.accuracy = G.GetAccuracy(labels_pred)
		self.prediction = labels_pred
		# print(self.accuracy)

	def __init__(self, G, eta = 0.05):
		self.accuracy = 0
		n = G.number_of_nodes()

		# Make a random sample of labeled nodes
		labeled_set = random.sample(list(G.nodes), int(eta * n))
		for u in labeled_set:
			G.nodes[u]['label'] = G.nodes[u]['ground_label']

		# The algorithm 
		self.analysis(G, labeled_set)

#### The simulation with some of presented algorithms 

def simulation(algorithm, n_1 = 100, n_2 = 100, a = 1, b = 1, n_trials = 1):
	avg_accuracy = 0
	graphs_array = []
	for i in range(n_trials): 
		G = GBM_graph(n_1, n_2, a, b, disp = False)
		graphs_array += [G]
		A = algorithm(G)
		avg_accuracy += A.accuracy / n_trials
	dict_temp = {"accuracy": avg_accuracy, "graphs": graphs_array}
	return dict_temp

#### The simulation with all algortihms for one value of b 

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
		acc_dict = {A.__name__: 0 for A in algo_list}
		time_dict = {A.__name__: 0 for A in algo_list}
		for i in range(n_trials): 
			G = GBM_graph(n_1, n_2, a, b, disp = False)
			for A in algo_list: 
				start_time = time.time() 
				alg_res = A(G)
				# print(acc)
				# G_dict.update({(a,b,n_1+n_2): sim['graphs']})
				time_sec = time.time() - start_time 
				acc_dict[A.__name__] += alg_res.accuracy / n_trials
				time_dict[A.__name__] += time_sec

		for A in algo_list:
			s = s + A.__name__ + ' = ' + str(np.around(acc_dict[A.__name__] * 100)) + '% (' + str(int(time_dict[A.__name__])) + 'sec), '
		print(s)

		cur_step.update(acc_dict)
		acc_array.append(cur_step) 		

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

