import numpy as np
from collections import deque

class FCA:

    """ 
    Class for formal concept analysis
    Data should be 2-dim np.array with 1,0 or boolean values
    I - context, G - objects, M - attributes
    """

    def __init__(self, data):
        self.I = data.copy().astype(dtype='bool')
        self.G = np.arange(self.I.shape[0])
        self.M = np.arange(self.I.shape[1])
        self.lattice = {} 
        self.adj = {}   #adjacency lists for cover relation
        self.conf = {} 



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
        Return type: set
        """
        self.lattice = {}
        A = []
        concept_index = 0

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
                            self.lattice[concept_index] = {'extent': objs, 'intent':attrs}
                            concept_index += 1
                        break

            if len(A) == 0:
                break                        

        return self.lattice



    def save_lattice(self, file_name='lattice'):
        """
        Save lattice to file with path specified by 'file_name'
        Params: file_name = 'lattice'
        """
        f = open(file_name, 'w')
        for index, concept in self.lattice.items():
            f.writelines('%d;%s;%s\n' % (
                index,
                ','.join([str(d) for d in concept['extent']]),
                ','.join([str(d) for d in concept['intent']])
            ))
        f.close()



    def load_lattice(self, file_name='lattice'):
        """
        Load lattice from file with path specified by 'file_name'
        Params: file_name = 'lattice'
        Returns: lattice
        """
        self.lattice = {}
        f = open(file_name, 'r')
        for line in f:
            row = line.strip('\n').split(';')
            self.lattice[int(row[0])] = {
                'extent': np.array([int(s) for s in row[1].split(',')]),
                'intent': np.array([int(s) for s in row[2].split(',')])
            }
        f.close()
        return self.lattice



    def calculate_conf(self):
        """
        Calculate confidence for every edge in diagram, returns and stores in 'conf', needs 'adj' with cover relation
        Return type: dict with pairs ("i_j":float), where "i_j" means edge 
                     from concept i to concept j, i<j i.e. in the downward direction
        """
        self.conf = {}
        for j in self.adj:
            for i in self.adj[j]:
                self.conf[str(i)+"_"+str(j)] = self.lattice[j]['extent']/self.lattice[i]['extent']
        return self.conf



    def select_concepts(self, classes, lower_supp = 0, upper_supp = 0,  purity_value = 0, accuracy_value = 0, f_measure_value = 0):
        """
        Select concepts satisfying at least one of the conditions:
        1) support of concept > upper_supp
        2) lower_supp <= support <= upper_supp and purity, f-measure and accuracy >= than the values in parameters
        Moreover for second case return new concept index (after reducing lattice to selected concepts) 
        and its top class - the most common class in the extent.
        Params: upper_supp = 0, lower_supp = 0, purity_value = 0, accuracy_value = 0, f_measure_value = 0
                classes - binary np.array where each column corresponds to class.
        Returns: list of original indices of selected concepts, dict with pairs (int class, list of  new concept indices)
        """
        concepts = []
        top_classes = dict((c,[]) for c in range(classes.shape[1]))
        new_index = 0

        if (upper_supp == 0) or (purity_value == 0 and accuracy_value == 0 and f_measure_value == 0):
            for index, concept in self.lattice.items():
                if len(concept['extent'])/self.I.shape[0] >= lower_supp:
                    concepts.append(index)
            return concepts, top_classes

        for index, concept in self.lattice.items():
            supp = len(concept['extent'])/self.I.shape[0]
            if supp > upper_supp:
                concepts.append(index)
                new_index += 1
                continue
            if supp >= lower_supp:
                class_frequency = classes[concept['extent'], :].sum(axis=0)
                top_class = np.argmax(class_frequency)
                tp = class_frequency[top_class]
                fp = len(concept['extent']) - tp
                bool_inverse = np.ones(classes.shape[0], dtype='bool')
                bool_inverse[concept['extent']] = False
                fn = np.logical_and(bool_inverse,classes[:,top_class]).sum()
                tn = classes.shape[0] - tp - fp - fn

                purity= class_frequency[top_class] / len(concept['extent'])
                accuracy = (tn+tp) / (tn+tp+fp+fn)
                f_measure = 2 * tp / (2*tp + fp + fn)

                if (purity >= purity_value and accuracy >= accuracy_value and f_measure >= f_measure_value):
                    concepts.append(index)
                    top_classes[top_class].append(new_index)
                    new_index += 1

        return concepts, top_classes



    def reduce_lattice(self, concept_indices):
        """
        Reduce lattice (change class attribute) to concepts that are in concept_indices
        Params: concept_indices 
        Returns: reduced lattice
        """
        lattice = {}
        new_index = 0
        for index, concept in self.lattice.items():
            if index in concept_indices:
                lattice[new_index] = self.lattice[index]
                new_index += 1
        self.lattice = lattice
        return self.lattice



    def build_cover_relation(self):
        """
        Calculate cover relation given lattice
        Returns: adjacency list
        Return type: dict with pairs (int:set)
        """
        n = len(self.lattice)
        concepts = np.zeros((len(self.lattice), len(self.M)), dtype='bool')
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

        return self.adj





