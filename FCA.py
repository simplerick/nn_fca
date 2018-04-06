import numpy as np
from collections import deque

class FCA:

    """ 
    Data should be 2-dim np.array with 1,0 or boolean values
    I - context, G - objects, M - attributes
    """

    def __init__(self, data):
        self.I = data.copy().astype(dtype='bool')
        self.G = np.arange(self.I.shape[0])
        self.M = np.arange(self.I.shape[1])
        self.lattice = {} 
        self.adj = {}   #adjacency lists for cover relation


    def closure(self, attributes):
        """ 
        Calculates closure 
        Arguments: attributes (iterable data structure)
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
        Calculates set of formal concepts (Ganter, Wille algorithm - finding in lectic order)
        Params: restriction on maximal number of attributes that can be in formal concept
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




    def reduce_lattice(self):
        pass



    def build_cover_relation(self):
        """
        Calculates cover relation given lattice
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





