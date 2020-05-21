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