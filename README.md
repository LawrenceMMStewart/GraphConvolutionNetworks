# Contents 

* **erdoslib.py** - A Python library for creating datasets of Erdos-Reyni graphs, sampling p from normal or uniform distributions with specified parameters
* **train.py**  - Train the GNN on various datasets specified by the control dictionary, saving weights and plotting losses.

The architecture used is a two layer Graph Convolution Network followed by a pooling into a linear layer with Sigmoid non-linearlity.

<img src="https://github.com/LawrenceMMStewart/GraphConvolutionNetworks/blob/master/saved_plots/model.png" width="700">

Required Packages: DGL & NetworkX

## License

The code is distributed under a Creative Commons Attribution 4.0 International Public License.
