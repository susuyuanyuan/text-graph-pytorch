# Text Graph Pytorch Implementation

A re-implementation of the graph convolution network for text learning in pytorch

1. Many Cleaned up on the original implementation
2. Removed the feature computing since proven to be not beneficial
3. Added class module to build the graph
4. Added class module to train the network
5. Added module to better manage the folder structure
6. Fixed the visualization tools for better visualization
7. Re-Implementation of the graph building and speed ups

To run this code:

```
python prepare_dataset.py 20ng
python word_graph_builder_main.py 20ng
python graph_network_trainer_main.py 20ng
python visualize_result.py 20ng
```
