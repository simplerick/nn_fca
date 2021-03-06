import numpy as np
from collections import deque

class FCA:

    """ 
    Class for formal concept analysis
    Data should be 2-dim np.array with 1,0 or boolean values
    I - context, G - objects, M - attributes
    """

    def __init__(self, data=np.zeros((0,0))):
        self.I = data.copy().astype(dtype='bool')
        self.G = np.arange(self.I.shape[0])
        self.M = np.arange(self.I.shape[1])
        self.lattice = []
        self.adj = {}   #adjacency lists for cover relation
        self.reversed_adj = {}
        self.conf = {} 
        self.support = {}
        self.purity = {}
        self.accuracy = {}
        self.f_measure = {}
        self.stability = {}
        self.partition_to_classes = {}



    def closure(self, attributes):
        """ 
        Calculate closure 
        Params: attributes (iterable data structure)
        Returns:  {attributes}', {attributes}"  in increaing order
        Return type: np.array, np.array  
        """
        objs = np.ones(len(self.G), dtype='bool')
        for m in attributes:
            objs = np.logical_and(objs,self.I[:, m])
        objects = self.G[objs]

        attrs = np.ones(len(self.M), dtype='bool')
        for g in objects:
            attrs = np.logical_and(attrs,self.I[g,:])
        attributes = self.M[attrs]
    
        return objects , attributes



    def build_lattice(self, level=5):
        """ 
        Calculate set of formal concepts (Ganter, Wille algorithm - finding in lectic order)
        Params: level=5 - restriction on maximal number of attributes that can be in formal concept
        Return type: list
        """
        self.lattice = []
        A = []

        while True:
            max_elem = len(self.M)-1
            if len(A) == level:
                max_elem = A[-1]

            for m in reversed(range(0,max_elem+1)):
                if len(A)>0 and A[-1] == m:         # A is always sorted
                    A.pop()
                else:
                    A.append(m)
                    objs, attrs = self.closure(A)

                    if len(attrs) > level:
                        A.pop()
                        continue

                    if len(A) + np.count_nonzero(attrs > m) < len(attrs):
                        A.pop()
                        continue
                    else:
                        A = attrs.tolist()
                        if len(objs) > 0 and len(attrs) > 0:
                            self.lattice.append({'extent': objs, 'intent':attrs})
                        break

            if len(A) == 0:
                break                        

        return self.lattice



    def save_lattice(self, path='data/lattice'):
        """
        Save lattice to file with specified path 
        Params: path
        """
        np.savez(path,I = self.I, M=self.M,G=self.G,lattice=self.lattice)



    def load_lattice(self, path='data/lattice'):
        """
        Load lattice from file with specified path
        Params: path
        Returns: lattice
        """
        data = np.load(path+".npz")
        self.I = data['I']
        self.G = data['G']
        self.M = data['M']
        self.lattice = data['lattice']
        return self.lattice



    def save_properties(self, path='data/properties'):
        """
        Save properties to file with specified path 
        Params: path
        """
        np.savez(path,
        conf = self.conf,
        support = self.support,
        purity = self.purity,
        accuracy = self.accuracy,
        f_measure = self.f_measure,
        stability  = self.stability,
        partition = self.partition_to_classes)



    def load_properties(self, path='data/properties'):
        """
        Load properties from file with specified path 
        Params: path
        """
        data = np.load(path+".npz")
        self.conf = data['conf'].item()
        self.support = data['support'].item()
        self.purity = data['purity'].item()
        self.accuracy = data['accuracy'].item()
        self.f_measure = data['f_measure'].item()
        self.stability = data['stability'].item()
        self.partition_to_classes = data['partition'].item()



    def calculate_conf(self):
        """
        Calculate confidence for every edge in diagram, returns and stores in 'conf', needs 'adj' with cover relation
        Return type: dict with pairs ("i_j":float), where "i_j" means edge 
                     from concept i to concept j, intent(i)<intent(j) i.e. in the downward direction
        """
        assert(len(self.adj)>0)
        self.conf = {}
        assert(len(self.adj)>0)
        for i in self.adj:
            self.conf["in,"+str(i)] = []
            for m in self.M:
                r = np.zeros((len(self.G),), dtype=bool)
                r[self.lattice[i]['extent']] = True
                temp = np.logical_and(r,self.I[:,m])
                self.conf["in,"+str(i)].append(np.count_nonzero(temp)/np.count_nonzero(self.I[:,m]))
            for j in self.adj[i]:
                self.conf[str(i)+"_"+str(j)] = len(self.lattice[j]['extent'])/len(self.lattice[i]['extent'])
        return self.conf



    def calculate_properties(self, classes):
        """
        Calculate support, purity, accuracy and f-measure values
        """
        self.purity = {}
        self.accuracy = {}
        self.f_measure = {}
        self.partition_to_classes = {}
        self.support = {}
        assert(len(self.lattice)>0)
        for index in range(len(self.lattice)):
            concept = self.lattice[index]
            self.support[index] = len(concept['extent'])/self.I.shape[0]
            class_frequency = classes[concept['extent'], :].sum(axis=0)
            top_class = np.argmax(class_frequency)
            tp = class_frequency[top_class]
            fp = len(concept['extent']) - tp
            bool_inverse = np.ones(classes.shape[0], dtype='bool')
            bool_inverse[concept['extent']] = False
            fn = np.logical_and(bool_inverse,classes[:,top_class]).sum()
            tn = classes.shape[0] - tp - fp - fn

            self.purity[index] = class_frequency[top_class] / len(concept['extent'])
            self.accuracy[index] = (tn+tp) / (tn+tp+fp+fn)
            self.f_measure[index] = 2 * tp / (2*tp + fp + fn)
            self.partition_to_classes[index] = top_class



    def calculate_stability(self):
        """
        Calculate "delta" estimate of stability for each concept in lattice. 
        Requires 'adj' with cover relation.
        Return type: dict with pairs (int i : float j), where i - concept index, j - stability 
        """
        assert(len(self.adj)>0)
        self.stability = {}
        for i in self.adj:
            self.stability[i] = 0
            for j in self.adj[i]:
                self.stability[i] -= 0.5**(len(self.lattice[i]['extent'])-len(self.lattice[j]['extent']))
        return self.stability



    def select_concepts(self, lower_supp = 0, upper_supp = 0,  purity_value = 0, accuracy_value = 0, f_measure_value = 0, stability_value = 0):
        """
        Select concepts satisfying at least one of the conditions:
        1) support of concept > upper_supp
        2) lower_supp <= support <= upper_supp and purity, f-measure and accuracy >= than the values in parameters
        Params: upper_supp = 0, lower_supp = 0, purity_value = 0, accuracy_value = 0, f_measure_value = 0
                classes - binary np.array where each column corresponds to class.
        Returns: list of indices of selected concepts
        """
        concepts = []
        
        if (upper_supp == 0) or (purity_value == 0 and accuracy_value == 0 and f_measure_value == 0):
            for index in range(len(self.lattice)):
                if self.support[index] >= lower_supp:
                    concepts.append(index)
            return concepts

        for index in range(len(self.lattice)):
            concept = self.lattice[index]
            if self.support[index] > upper_supp:
                concepts.append(index)
                continue
            if self.support[index] >= lower_supp:
                if (self.purity[index] >= purity_value and self.accuracy[index] >= accuracy_value and self.f_measure[index] >= f_measure_value and self.stability[index] >= stability_value):
                    concepts.append(index)

        return concepts



    def reduce_lattice(self, concept_indices):
        """
        Reduce lattice (change class attribute) to concepts that are in concept_indices
        Params: concept_indices 
        Returns: reduced lattice
        """
        lattice = []
        partition = {}
        new_index = 0
        for index in range(len(self.lattice)):
            if index in concept_indices:
                lattice.append(self.lattice[index])
                partition[new_index] = self.partition_to_classes[index]
                new_index += 1
        self.lattice = lattice
        self.partition_to_classes = partition
        return self.lattice



    def build_cover_relation(self, reverse=False):
        """
        Calculate cover relation given lattice, i.e. iRj iff intent(i)<intent(j)
        Returns: adjacency list or adjacency list of reversed relation if reverse = True
        Return type: dict with pairs (int:set)
        """
        n = len(self.lattice)
        concepts = np.zeros((n, len(self.M)), dtype='bool')
        self.adj = {}

        for i in reversed(range(n)):
            concepts[i, self.lattice[i]['intent']] = True
            self.adj[i] = set()
            for j in range(i+1,n):
                if (concepts[j])[concepts[i]].all():
                    neighbor = True
                    for k in self.adj[i]:
                        if j in self.adj[k]:
                            neighbor = False
                            break
                    if neighbor:
                        self.adj[i].add(j)
        
        self.reversed_adj = dict((i,set()) for i in range(n))
        for i in self.adj:
            for j in self.adj[i]:
                self.reversed_adj[j].add(i)
                
        if reverse:
            return self.reversed_adj
        return self.adj
