from erdoslib import ERdat
import dgl
import matplotlib.pyplot as plt
import networkx as nx
import torch 
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import pylab
import numpy as np

#throughout this file we aim to test an architecture that can infer the probability componenent of an Erodos-Reyni random graph
#create file for investigating depths of networks
f= open("saved_plots/min_loss.txt","w+")

#Collate function batches graphs together
def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples)) #extract the two as lists:
    batched_graph = dgl.batch(graphs)  #use inbuilt jordan block function from the package
    return batched_graph, torch.tensor(labels) #return both the graph (of all graphs combined together) and a tensor of the labels


#define the message function -- sends out h of source nodes as message m 
msg = fn.copy_src(src='h', out='m')


#create reduce function:

def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}




#create a class node apply module that updates node features with relu:
class NodeApplyModule(nn.Module):
    """Update the node embeddings hv with ReLU(Whv+b).""" 
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}




#GCN class that runs through apply module
class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        # Initialize the node features with h.
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

"""
Initial node features to be their degrees.]
After two rounds of graph convolution, perform a graph readout by averaging over all 
node features for each graph in the batch

In DGL, dgl.mean_nodes() handles this task for a batch of graphs with variable
size. We then feed our graph representations into a classifier that predicts the value of p

"""




                                       

#classifier 1 is a 2 layer (or 1 hidden layer) classical GCN.

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Classifier, self).__init__()

        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu)])
        self.classify = nn.Linear(hidden_dim,1)


    def forward(self, g):

        #aggregation stage --- through graph
        h = g.in_degrees().view(-1, 1).float()
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        #hg is the aggreation function - average 
        hg = dgl.mean_nodes(g, 'h')

        #linear layer
        lin_out =torch.sigmoid(self.classify(hg))
                            
        return lin_out


#define the maximum likelihood estimator for p from a graph:
def max_likelihood_estimator(g):
    """
    Input: nx.graph
    Output: maximum likelihood estimate for p    
    """
    adj_mat=nx.to_numpy_array(g)
    n=adj_mat.shape[0]
    sliding_row_sums=[np.sum(adj_mat[i][i+1:]) for i in range(n)]
    return 1/(n*(n-1)/2)*np.sum(sliding_row_sums)

#control panel- dictionary of all parameters and booleans for training and testing -- modify to change tests:
control= {
    
    'epochs': 75, #number of training epochs 
    'batch_size':64, 
    'hidden_layer_size':100, #100 recommended for speed on i7 core 16gb ram cpu
    'tr_no':400, # number of samples for training set
    'te_no':100, # number of samples for test set
    'min_no':5, # minimum number of nodes for each graph
    "max_no":50, #maxmimum number of nodes for each graph


    #boolean values to indicate running various distributions
    "run_uniform1":True, # option to run the uniformmet1 distribution
    "run_normal1":True, #  option to run the normalmet1 distribution
    "run_normal2":True, #  option to run the normalmet2 distribution
    "run_normal3":True, #  option to run the normalmet3 distribution

    #meta-parameters
    "met_uniform1":[0,1],
    "met_normal1":[0.5,0.1],
    "met_normal2":[0.8,0.1],
    "met_normal3":[0.2,0.1],

    #save lossfigures and histograms:
    "savl_uniform1":True,
    "savl_normal1":True,
    "savl_normal2":True,
    "savl_normal3":True,
    "save_hists":True,

    #display loss plots:
    "display_plts_loss": True,

    #display hist plots:
    "display_plts_hist": True,

    #save weights:
    "save_weights":True,

    #write minimum loss to file for each dist
    "write_min":True,
    #write test error to file for each dist
    "write_test_loss":True


}


run_list=[]

hists=[]

#collect all runs to the list

if control['run_uniform1']:
    trs=ERdat(control['tr_no'],control['min_no'],control['max_no'],"Uniform",control["met_uniform1"])
    tes=ERdat(control['te_no'],control['min_no'],control['max_no'],"Uniform",control["met_uniform1"])
    run_list.append([trs,tes,control["savl_uniform1"],"Uniform",control["met_uniform1"]])

if control['run_normal1']:
    trs=ERdat(control['tr_no'],control['min_no'],control['max_no'],"Normal",control["met_normal1"])
    tes=ERdat(control['te_no'],control['min_no'],control['max_no'],"Normal",control["met_normal1"])
    run_list.append([trs,tes,control["savl_normal1"],"Normal",control['met_normal1']])

if control['run_normal2']:
    trs=ERdat(control['tr_no'],control['min_no'],control['max_no'],"Normal",control["met_normal2"])
    tes=ERdat(control['te_no'],control['min_no'],control['max_no'],"Normal",control["met_normal2"])
    run_list.append([trs,tes,control["savl_normal2"],"Normal",control['met_normal2']])

if control['run_normal3']:
    trs=ERdat(control['tr_no'],control['min_no'],control['max_no'],"Normal",control["met_normal3"])
    tes=ERdat(control['te_no'],control['min_no'],control['max_no'],"Normal",control["met_normal3"])
    run_list.append([trs,tes,control["savl_normal3"],"Normal",control['met_normal3']])

                                        
for dat in run_list:

    trainset=dat[0]
    testset=dat[1]

    # Create model
    model = Classifier(1, control['hidden_layer_size']) 
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()


    epoch_losses = []
    #check this line 
    print("Training GNN "+trainset.get_distribution_label(),end="\n")

    for epoch in range(control['epochs']):

        trainset.shuffle()

        #use collate function previously defined to batch the graphs:
        data_loader = DataLoader(trainset, batch_size=control['batch_size'], shuffle=True,
                             collate_fn=collate)
        epoch_loss = 0
        for iter, (bg, label) in enumerate(data_loader):

            prediction = model(bg)
            loss = loss_func(prediction, label.reshape((label.size()[0],1)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        #save the sqrt of the mean square error of a batch
        epoch_losses.append(np.sqrt(epoch_loss))


    model.eval()

    # Convert a list of tuples to two lists
    test_X, test_Y = map(list, zip(*testset))
    test_bg = dgl.batch(test_X)
    test_Y = torch.tensor(test_Y).float().view(-1, 1)
    pred_Y = model(test_bg)

    print("minimum loss reached = ",min(epoch_losses))

    plt.plot(epoch_losses,alpha=0.7,label=dat[3]+' '+'(%f,%f)'%(dat[4][0],dat[4][1]))
    plt.xlabel("epoch")
    plt.ylabel(r"$\sqrt{MSE}$")
    plt.legend()
    plt.title('Comparison of loss throughout training')
    plt.grid('on')
    ax = plt.gca()
    ax.set_facecolor('#D9E6E8')

    #save figures on last experiment--
    if run_list[2] and dat==run_list[-1]:
        pylab.savefig('saved_plots/Loss.png')
    #display plots on last 
    if control["display_plts_loss"] and dat==run_list[-1]:
        plt.show()
    #save weights
    if control["save_weights"]:
        torch.save(model.state_dict(),"saved_weights/"+dat[3]+'(%f,%f)'%(dat[4][0],dat[4][1])+".pt")
    #write min loss:
    if control['write_min']:
        f.write("--------------------------------"+'\n')
        f.write("minimum loss reached on trainset "+dat[3]+' '+'(%f,%f)'%(dat[4][0],dat[4][1])+ ' =  %f' %(min(epoch_losses))+'\n')
        f.write("--------------------------------"+'\n')
    if control['write_test_loss']:
        f.write("MSE for testset prediction "+dat[3]+' '+'(%f,%f)'%(dat[4][0],dat[4][1])+ ' =  %f' %(loss_func(pred_Y.detach(),test_Y).numpy())+'\n')




        #form of list and p 
    #evaluate performance on various p values:
    hist_dat=trainset.__create__histdat__(10)
    hist_preds=[]
    hist_mles=[]

    for (g_list,p) in hist_dat:
        errors_gnn=[]
        errors_mles=[]
        for g in g_list:
            pred=model(dgl.DGLGraph(g))
            #record the square root of square error
            errors_gnn.append(abs(p-pred).detach().numpy()[0])
            errors_mles.append(abs(p-max_likelihood_estimator(g)))
        hist_preds.append(np.mean(errors_gnn))
        hist_mles.append(np.mean(errors_mles))
            
    #save the error for the GNN, the MLE and save the name of the distribution the GNN was trained on
    hists.append((hist_preds,hist_mles,trainset.get_distribution_label()))



#plot each histogram:
for i in range(len(hists)):
    gnn_errors=hists[i][0]
    mle_errors=hists[i][1]
    name_of_dist=hists[i][2]
    # data to plot
    n_groups = len(gnn_errors)
         
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8
    ax.set_facecolor('#D9E6E8')
     

    rects1 = plt.bar(index, gnn_errors, bar_width,alpha=opacity,color='b',label='GNN')
  
    rects2 = plt.bar(index + bar_width, mle_errors, bar_width,alpha=opacity,color='g',label='MLE')
     
    plt.xlabel('p')
    plt.ylabel(r"$\sqrt{MSE}$")
    plt.title("Errors for varied p "+name_of_dist)
    plt.xticks(index + bar_width, tuple([float("%.2f" % (0.1*(i+1))) for i in range(n_groups)])) 
    plt.legend()
    plt.tight_layout()
    if control["save_hists"]:
        pylab.savefig('saved_plots/Hist'+name_of_dist+'.png')
    if control["display_plts_hist"]:
        plt.show()

