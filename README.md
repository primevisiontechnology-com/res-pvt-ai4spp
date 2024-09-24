# RoboNav: Robot Path Planning Using Reinforcement Learning and Attention Models

Welcome to our project! We're using the Deep Reinforcement Learning to solve the Shortest Path Problem (SPP) in package sorting warehouse Floorplans. Our AI agent is training hard, learning from its mistakes, and getting better and better at finding the shortest path. 

## What's Inside? 🎁

- **Deep Reinforcement Learning**: Our AI agent learns by interacting with its environment.
- **Grid Environment**: Our agent operates in a grid environment with and without obstacles.
- **Real Floorplan**: Our agent operates in the real floorplan environment with nodes and connections.
- **Random Start and Target Nodes**: Each episode is a new challenge, with start and target nodes randomly picked.
- **Greedy Rollouts**: During the inference step, our agent performs greedy rollouts on the available actions.
- **A-Star Search Baseline**: We compare our trained policies with an A-star search baseline to confirm validity. It's like a reality check, but for AI!
- **Masking and embeddings**: Graph Neural Networks can use masking and embeddings to learn from the graph structure. We're using them to improve our agent's performance.

This is a visualistion of the mask with the graph structure, without the mask the agent would be able to see the whole graph structure, with the mask the agent can only see the nodes that are connected to the current node.
![alt text](media/adjancyMatrix1.png)

The obstacle mask is used to prevent the agent from moving through obstacles. These edges are removed from the adjacency matrix.
![alt text](media/adjencyMatrix2.png)

## Visualizations 🎨

We've made our environment observable with beautiful visualizations. Actions are shown in red, start and target nodes in green and red, and the edges in the adjacency matrix of the graph in green.

![alt text](media/goodDRLOutputDynamic.png)

## How to Use 🚀

1. Install the required packages:
```bash
pip install rl4co==0.3.3
pip install torch==2.3.0
pip install matplotlib
pip install numpy
pip install sklearn
```

2. Import the libraries:
```python
from FPenv import FPEnv
from SPPenv import SPPEnv
from astar import AStarSearch
from SPPembeddings import SPPInitEmbedding, SPPContext, StaticEmbedding
from SPPv2env import SPPv2Env # import the dynamic environment
from Floorplan_Codes.utils import get_paths
```

3. The jupyter notebook 'exploration.ipynb' contains the code to train the Deep Reinforcement Learning algorithms and make predictions on real floorplans. It contains the code for the training and evaluation of the AM and POMO methods.

## Results 📊

The result histogram for the static environment.
![alt text](media/plotHistogram.png)

The result histogram for the dynamic environment.
![alt text](media/plotHistDyn.png)

The results of real floorplan predictions can be found in the last cell of 'exploration.ipynb'.

## Authors 🧑‍💻
Yifei Zhou (y.zhou@primevisiontechnology.com)
Timo Thans (timot@vt.edu)
