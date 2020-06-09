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
		self.accuracy = max(accuracy_score(labels_pred, G.labels), 1 - accuracy_score(labels_pred, G.labels))
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