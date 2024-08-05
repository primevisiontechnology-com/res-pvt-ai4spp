# !pip install rl4co==0.3.3
# !pip install torch==2.3.0
# !pip install matplotlib

from SPPembeddings import SPPInitEmbedding, SPPContext, StaticEmbedding
from rl4co.models.zoo import AttentionModel, AutoregressivePolicy, POMO, POMOPolicy
from FPenvPlanner import FPEnvPlanner
import torch

# Load the trained model
model = torch.load('Models/TrainOnFloorplansResults8.pth')

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create the environment
infer_env = FPEnvPlanner(fp_path="floorplan.json")
td_init = infer_env.reset().to(device)

# Check for NaNs in a tensor dictionary
nan_found = False
for key, tensor in td_init.items():
    if torch.isnan(tensor).any():
        nan_found = True
        print(f"NaN values found in tensor '{key}'")
if not nan_found:
    print("No NaN values found in the tensor dictionary.")

# Get trained actions
policy = model.policy.to(device)
out = policy(td_init.clone(), infer_env, phase="test", decode_type="greedy", return_actions=True)
actions_trained = out['actions'].cpu().detach()