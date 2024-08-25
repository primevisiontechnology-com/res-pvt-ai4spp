from typing import Optional
import torch
import numpy as np
from typing import List, Dict

import torch.nn as nn

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.zoo import AttentionModel, AutoregressivePolicy
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.utils.trainer import RL4COTrainer
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

from utils import compute_manhattan_distance, generate_adjacency_matrix

# Import codes to read JSON floorplan
from Floorplan_Codes.floorplan import Floorplan

def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
    """Reset the environment to the initial state"""
    # If no TensorDict is provided, generate a new one
    init_locs = td["locs"] if td is not None else None
    init_edges = td["edges"] if td is not None else None
    # If no batch_size is provided, use the batch_size of the initial locations
    if batch_size is None:
        if init_locs is None:
            batch_size = self.batch_size
        else:
            batch_size = init_locs.shape[:-2]

    # If no device is provided, use the device of the initial locations
    device = init_locs.device if init_locs is not None else self.device
    self.to(device)
    # If no initial locations are provided, generate new ones
    if init_locs is None:
        grid_out = self.generate_data(batch_size=batch_size).to(device)
        init_locs = grid_out["locs"]
        init_edges = grid_out["edges"]

    # Reset step count
    step_count = torch.zeros(batch_size, dtype=torch.int64, device=device)

    # If batch_size is an integer, convert it to a list
    batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

    # Get the number of locations
    num_loc = init_locs.shape[-2]
    # print("num_loc: ", num_loc)

    # Initialize a start node
    start_nodes_tensor = torch.tensor(self.start_nodes, device=device)
    start_indices = torch.randint(0, len(self.start_nodes), (batch_size), device=device)
    first_node = start_nodes_tensor[start_indices]

    # Initialize the end node to a random node until it is unequal to the start node
    end_nodes_tensor = torch.tensor(self.end_nodes, device=device)
    while True:
        end_indices = torch.randint(0, len(self.end_nodes), (batch_size), device=device)
        end_node = end_nodes_tensor[end_indices]
        if not torch.any(torch.eq(first_node, end_node)):
            break

    batch_indices = torch.arange(len(first_node))
    available = init_edges[batch_indices, first_node]

    # Compute the distance to the target
    target = init_locs[batch_indices, end_node].unsqueeze(1)
    distance_to_target = torch.norm(init_locs - target, dim=-1, keepdim=True)


    # Initialize the index of the current node
    i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

    # Initialize the done mask
    done_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Initialize the visited nodes tensor
    visited_nodes = torch.zeros(*batch_size, num_loc, dtype=torch.bool, device=device)

    visited_nodes[batch_indices, first_node] = 1

    return TensorDict(
        {
            "locs": init_locs,
            "edges": init_edges,
            "manhattan_matrix": distance_to_target,
            "first_node": first_node,
            "current_node": first_node,
            "end_node": end_node,
            "step_count": step_count,
            "i": i,
            "action_mask": available,
            "visited": visited_nodes,
            "reward": torch.zeros((*batch_size, 1), dtype=torch.float32),
            "done_mask": done_mask,
        },
        batch_size=batch_size,
    )


def _step(self, td: TensorDict) -> TensorDict:
    # Only update the non-done elements
    done_mask = td["done_mask"]

    current_node = torch.where(~done_mask, td["action"], td["current_node"])

    step_count = td["step_count"]
    step_count = torch.where(~done_mask, step_count + 1, step_count)

    available = get_action_mask(self, td)
    
    # Add the current node to the visited nodes
    visited = td["visited"]

    batch_indices = torch.arange(len(current_node))
    visited[batch_indices, current_node] = 1

    done = current_node == td["end_node"]
    # Yifei: Change the 100 steps limits to 1000
    done |= step_count >= 1000

    done_mask = done_mask | done

    # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
    # reward = self.get_reward(td, td["action"])
    reward = torch.zeros_like(done)

    td.update(
        {
            "step_count": step_count,
            "current_node": current_node,
            "i": td["i"] + 1,
            "action_mask": available,
            "visited": visited,
            "reward": reward,
            "done": done,
            "done_mask": done_mask,
        },
    )
    return td


def get_action_mask(self, td: TensorDict) -> TensorDict:
    # Get the current node
    current_node = td["action"]

    # Create a tensor of batch indices
    batch_indices = torch.arange(len(current_node))

    # Use advanced indexing to get the neighbors for each batch
    neighbors = td["edges"][batch_indices, current_node]

    available = neighbors

    return available


def get_reward(self, td, actions) -> TensorDict:
    # Every step has a reward of -1
    step_count = td["step_count"]
    # if any of the batch element has reached maximum steps, set to infinity
    step_mask = step_count >= 1000
    # Base reward is -step_count, and penalize heavily if step count exceeds 1000
    reward = torch.where(step_mask, torch.tensor(-1000.0, dtype=torch.float32, device=step_count.device),
                         -step_count.float())

    # Get the current node
    # current_node = td["current_node"]

    # Check which nodes were previously visited
    visited = td["visited"]
    # batch_indices = torch.arange(len(current_node))

    # Check if the current node is newly visited
    # is_newly_visited = ~visited[batch_indices, current_node]

    # new_visit_reward = torch.where(is_newly_visited, torch.tensor(10.0, dtype=torch.float32, device=reward.device), 
    #                                torch.tensor(0.0, dtype=torch.float32, device=reward.device))

    # Calculate the number of uniquely visited nodes for each batch by summing the visited nodes
    num_unique_visits = visited.sum(dim=-1).float()  # Sum across the nodes dimension to get the unique count

    exploration_reward = torch.where(step_mask, num_unique_visits * 1.0,
                                     torch.tensor(0.0, dtype=torch.float32, device=step_count.device))
    
    reward += exploration_reward

    # Add the new visit reward to the total reward
    # reward += new_visit_reward
    # print("Reward: ", reward)
    return reward


def _make_spec(self, td_params):
    """Make the observation and action specs from the parameters"""
    self.observation_spec = CompositeSpec(
        locs=BoundedTensorSpec(
            low=self.min_loc,
            high=self.max_loc,
            shape=(self.num_loc, 2),
            dtype=torch.float32,
        ),
        edges=UnboundedDiscreteTensorSpec(
            shape=(self.num_loc, self.num_loc),
            dtype=torch.bool,
        ),
        manhattan_matrix=UnboundedContinuousTensorSpec(
            shape=(self.num_loc, self.num_loc),
            dtype=torch.float32,
        ),
        start_node=UnboundedDiscreteTensorSpec(
            shape=(1),
            dtype=torch.int64,
        ),
        end_node=UnboundedDiscreteTensorSpec(
            shape=(1),
            dtype=torch.int64,
        ),
        current_node=UnboundedDiscreteTensorSpec(
            shape=(1),
            dtype=torch.int64,
        ),
        i=UnboundedDiscreteTensorSpec(
            shape=(1),
            dtype=torch.int64,
        ),
        action_mask=UnboundedDiscreteTensorSpec(
            shape=(self.num_loc),
            dtype=torch.bool,
        ),
        done_mask=UnboundedDiscreteTensorSpec(
            shape=(1),
            dtype=torch.bool,
        ),
        step_count=UnboundedDiscreteTensorSpec(
            shape=(1),
            dtype=torch.int64,
        ),
        shape=(),
    )
    self.action_spec = BoundedTensorSpec(
        shape=(1,),
        dtype=torch.int64,
        low=0,
        high=self.num_loc,
    )
    self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
    self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)


def generate_data(self, batch_size) -> TensorDict:
    # Temporary set batch_size regardless, need to change
    batch_size = 4

    # Ensure batch_size is an integer
    batch_size = int(batch_size[0]) if isinstance(batch_size, list) else batch_size

    # Add batch dimension and repeat the locs tensor for each item in the batch
    if self.locs.dim() == 2:
        self.locs = self.locs.unsqueeze(0).repeat(batch_size, 1, 1)

    # Generate edges tensors
    edges = torch.zeros((batch_size, self.num_loc, self.num_loc), dtype=torch.bool)
    # Generate the adjacency matrix
    adjacency_matrix = self.generate_adjacency_matrix_fp()
    # Convert the adjacency matrix into a PyTorch tensor with a data type of torch.bool
    adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.bool)
    # Add a Batch Dimension and Repeat the Tensor
    adjacency_matrix = adjacency_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
    # Assign the Tensor to edges
    edges[:] = adjacency_matrix

    batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

    return TensorDict({"locs": self.locs, "edges": edges}, batch_size=batch_size)


def plot_graph(self, td, ax=None):
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        # Create a plot of the nodes
        _, ax = plt.subplots()

    # if batch_size greater than 0 , we need to select the first batch element
    if td.batch_size != torch.Size([]):
        td = td[0]

    locs = td["locs"]
    x, y = locs[:, 0], locs[:, 1]
    # locs = locs.detach().cpu()

    # Plot the nodes
    ax.scatter(x, y, color="tab:blue")


    # Plot the edges
    edges = td["edges"]

    x_i, y_i = locs[:, 0], locs[:, 1]

    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i, j]:
                ax.plot([x_i[i], x_i[j]], [y_i[i], y_i[j]], color='g', alpha=0.1)

    # # Setup limits and show
    # ax.set_xlim(-10, 10)
    # ax.set_ylim(-10, 10)

    ax.autoscale()

    if ax.figure:
        ax.figure.show()

    return ax

def render(self, td, actions=None, ax=None):
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        # Create a plot of the nodes
        _, ax = plt.subplots()

    td = td.detach().cpu()

    if actions is None:
        actions = td.get("action", None)
    # if batch_size greater than 0 , we need to select the first batch element
    if td.batch_size != torch.Size([]):
        td = td[0]
        actions = actions[0]

    locs = td["locs"]

    # First and end node
    end_node = td["end_node"]
    x_end, y_end = locs[end_node, 0], locs[end_node, 1]
    start_node = td["first_node"]
    x_start, y_start = locs[start_node, 0], locs[start_node, 1]

    # gather locs in order of action if available
    if actions is None:
        print("No action in TensorDict, rendering unsorted locs")
    else:
        actions = actions.detach().cpu()
        # Filter out the nodes after the end node
        end_idx = torch.where(actions == end_node)[0]
        if end_idx.numel() > 0:
            end_idx = end_idx[0]
            actions = actions[: end_idx + 1]

    print(f"Actions Sizes: {actions.shape}")
    print(f"Actions indices: {actions}")

    a_locs = gather_by_index(locs, actions, dim=0)

    # Cat the start node to the start of the action locations
    a_locs = torch.cat([torch.tensor([[x_start, y_start]]), a_locs], dim=0)
    x, y = a_locs[:, 0], a_locs[:, 1]

    # Plot the visited nodes
    ax.scatter(locs[:, 0], locs[:, 1], color="tab:blue")

    # print("end node: ", end_node)
    # Highlight the start node in green
    ax.scatter(x[0], y[0], color="tab:green")

    # Highlight the end node in red
    ax.scatter(x[-1], y[-1], color="tab:red")

    # Plot the edges
    edges = td["edges"]
    x_i, y_i = locs[:, 0], locs[:, 1]
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i, j]:
                ax.plot([x_i[i], x_i[j]], [y_i[i], y_i[j]], color='g', alpha=0.1)

    # Add arrows between visited nodes as a quiver plot
    dx, dy = np.diff(x), np.diff(y)
    ax.quiver(
        x[:-1], y[:-1], dx, dy, scale_units="xy", angles="xy", scale=1, color="r", alpha=1.0
    )

    # Highlight the last action
    ax.scatter(x_end, y_end, color="tab:red", s=100, edgecolors="black", zorder=10)
    # Highlight the first action
    ax.scatter(x_start, y_start, color="tab:green", s=100, edgecolors="black", zorder=10)

    # # Setup limits and show
    # ax.set_xlim(-10.00, 10.00)
    # ax.set_ylim(-10.00, 10.00)

    ax.autoscale()

def process_fp(self, fp_path: str):
    # Read the floorplan in the floorplan path
    self.fp = Floorplan(fp_path)
    # Get all cells as a dictionary in fp
    self.all_cells = self.fp.getCells()
    # Get the number of cells
    self.num_loc = len(self.all_cells)

    # Put cell ids (keys) in dict into a cell list
    idList = list(self.all_cells.keys())
    # Put the ids into a dictionary (key: id; value: index) for fast indexing
    self.idDic = {}
    for i in range(len(idList)):
        self.idDic[idList[i]] = i

    # Put cells (values) in dict into a cell list
    self.cellsList = list(self.all_cells.values())

    # Initialize lists to hold x and y coordinates
    x_coords = []
    y_coords = []

    # Record the possible start nodes and end nodes indices
    self.start_nodes = []
    self.end_nodes = []

    # Extract x and y coordinates separately
    for i in range(len(self.cellsList)):
        cell = self.cellsList[i]

        if cell.getType() == "input":
            self.start_nodes.append(i)
        elif cell.getType() == "target":
            self.end_nodes.append(i)

        x_coords.append(cell.pose[0])
        y_coords.append(cell.pose[1])

    # Convert lists to PyTorch tensors
    x_tensor = torch.tensor(x_coords, dtype=torch.float32)
    y_tensor = torch.tensor(y_coords, dtype=torch.float32)

    # Stack tensors together to form a 2D tensor
    self.locs = torch.stack((x_tensor, y_tensor), dim=1)


def generate_adjacency_matrix_fp(self):
    # Initialize an empty adjacency matrix
    adjacency_matrix = np.zeros((self.num_loc, self.num_loc))

    # Iterate over each node
    for i in range(self.num_loc):
        # The current cell to inspect connections
        cell = self.cellsList[i]

        # For all neighbors:
        for connection in cell.connections:
            # Get the real connected neighbor id to ensure neighbor_id in consistent format - Yifei
            if '/' in connection["connects_to"]:
                parts = connection["connects_to"].split('/')
                neighbor_id = parts[-1]
            else:
                neighbor_id = connection["connects_to"]

            # If neighbor does not exist, continue
            if neighbor_id not in self.all_cells:
                continue

            # Get neighbor's index based on id
            neighbor_index = self.idDic[neighbor_id]
            # Get neighbor cell
            neighbor = self.all_cells[neighbor_id]

            # Mark the corresponding position in the adjacency_matrix as 1
            adjacency_matrix[i, neighbor_index] = 1

    return adjacency_matrix


class FPEnv(RL4COEnvBase):
    """Floorplan Reinforcement Learning Environment for Python Training"""

    name = "tsp"

    def __init__(
            self,
            min_loc: float = 0,
            max_loc: float = 1,
            td_params: TensorDict = None,
            fp_path: str = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.process_fp(fp_path)
        self._make_spec(td_params)

    _reset = _reset
    _step = _step
    get_reward = get_reward
    # check_solution_validity = check_solution_validity
    get_action_mask = get_action_mask
    _make_spec = _make_spec
    generate_data = generate_data
    render = render
    process_fp = process_fp
    generate_adjacency_matrix_fp = generate_adjacency_matrix_fp
    plot_graph = plot_graph
