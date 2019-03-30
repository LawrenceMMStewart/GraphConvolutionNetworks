
#  ______         _             _____                  _   _____        _        
# |  ____|       | |           |  __ \                (_) |  __ \      | |       
# | |__   _ __ __| | ___  ___  | |__) |___ _   _ _ __  _  | |  | | __ _| |_ __ _ 
# |  __| | '__/ _` |/ _ \/ __| |  _  // _ \ | | | '_ \| | | |  | |/ _` | __/ _` |
# | |____| | | (_| | (_) \__ \ | | \ \  __/ |_| | | | | | | |__| | (_| | || (_| |
# |______|_|  \__,_|\___/|___/ |_|  \_\___|\__, |_| |_|_| |_____/ \__,_|\__\__,_|
#                                           __/ |                                
#                                          |___/                                 

"""
A library that allows the creation of datasets of erdos reyni graphs, created by sampling p from probability distributions such
as normal or uniform. Requires networkx and dgl:
1) networkx:  https://networkx.github.io/
2) dgl: 	  https://www.dgl.ai/


"""

import math
import networkx as nx
import numpy as np
import dgl
from dgl import DGLGraph
import matplotlib.pyplot as plt
import random
#list all classes that are imported if we use the command from ERdat import *
__all__ = ['ERdat']


class ERdat(object):
    """ the dataset class.
    ..please build in way that is compatible with pytorches dataset class
    Parameters
    ----------
    num_graphs: int
        Number of graphs in this dataset.
    min_num_v: int
        Minimum number of nodes for graphs
    max_num_v: int
        Maximum number of nodes for graphs
    distribution : string
        Selects the distribution to sample p from options: "Uniform" and "Normal"
    metaparams : int list
        A list designating the range of the uniform distribution or the mean and std dev of the 
        normal distribution
    """

    def __init__(self, num_graphs, min_num_v, max_num_v, distribution="Normal",metaparams=[0.5,0.1]):
        super(ERdat, self).__init__()
        self.num_graphs = num_graphs
        self.min_num_v = min_num_v
        self.max_num_v = max_num_v
        self.graphs = []
        self.pvals = []
        self.distribution=distribution
        self.metaparams=metaparams
        self._generate()

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Retrieve the i^th sample from the dataset.
        Parameters
        ---------
        idx : int
            The sample index.
        Returns
        -------
        (dgl.DGLGraph, int)
            The graph and its label.
        """
        return self.graphs[idx],self.pvals[idx]

    def _generate(self):
        """Generates self.num_graphs ER graphs"""
        self._gen_(self.num_graphs) 

        for i in range(self.num_graphs):
            self.graphs[i] = DGLGraph(self.graphs[i])
            # add self edges
            nodes = self.graphs[i].nodes()
            self.graphs[i].add_edges(nodes, nodes)

  

    def _gen_(self, n):
        for _ in range(n):
            
            #generate p value:

            if self.distribution=="Uniform":
                p=np.random.uniform(self.metaparams[0],self.metaparams[1])
                num_v = np.random.randint(self.min_num_v, self.max_num_v)
                g = nx.binomial_graph(num_v,p)
                self.graphs.append(g)
                self.pvals.append(p)
            

            elif self.distribution=="Normal":
                p=min(abs(np.random.normal(self.metaparams[0],self.metaparams[1])) ,1)
                num_v = np.random.randint(self.min_num_v, self.max_num_v)
                g = nx.binomial_graph(num_v,p)
                self.graphs.append(g)
                self.pvals.append(p)

            else:
                raise Exception("Please enter a valid distribution from: Normal, Uniform")



    #shuffle function to mix dataset:
    def shuffle(self):
        """Shuffles the dataset randomly
        """
        temp =  list(zip(self.graphs, self.pvals))
        random.shuffle(temp)
        self.graphs, self.pvals = zip(*temp)

    #plot_graph i 
    def __plotitem__(self,idx):
        """
        Plots a selected graph
         Parameters
        ---------
        idx : int
            The sample index.
        Returns
        -------
        Plot of graph
        """

        dg,p= self.__getitem__(idx)
        nx.draw(dg.to_networkx())
        plt.title("Graph %d with p= %s" % (idx,p))
        plt.show()





