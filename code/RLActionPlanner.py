# !pip install rl4co==0.3.3
# !pip install torch==2.3.0

# from SPPembeddings import SPPInitEmbedding, SPPContext, StaticEmbedding
# from rl4co.models.zoo import AttentionModel, AutoregressivePolicy, POMO, POMOPolicy
from FPenvPlanner import FPEnvPlanner
from flask import Flask, jsonify, request
import torch

# Create the Flask app
app = Flask(__name__)

# Load the trained model
model = torch.load('Models/TrainOnFloorplansResults8.pth', map_location=torch.device('cpu'))

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Check for NaNs in a tensor dictionary
def check_for_nans(td):
    for key, tensor in td.items():
        if torch.isnan(tensor).any():
            return key
    return None


@app.route('/get_action_ids', methods=['POST'])
def get_action_ids():
    """
    Endpoint to retrieve action_ids.
    """

    # Extract fp_path, start_node_id, and target_node_id from the request body
    fp_path = request.json.get('fp_path', None)
    start_node_id = request.json.get('start_node_id', None)
    target_node_id = request.json.get('target_node_id', None)

    # Check if the three fields are received
    if not fp_path:
        return jsonify({"status": "error", "message": "fp_path is required"}), 400
    if not start_node_id:
        return jsonify({"status": "error", "message": "start_node_id is required"}), 400
    if not target_node_id:
        return jsonify({"status": "error", "message": "target_node_id is required"}), 400

    # Create the environment with the provided fp_path
    infer_env = FPEnvPlanner(fp_path=fp_path, start_node_id=start_node_id, target_node_id=target_node_id)
    td_init = infer_env.reset().to(device)

    # Check for NaNs
    nan_key = check_for_nans(td_init)
    if nan_key:
        return jsonify({"status": "error", "message": f"NaN values found in tensor '{nan_key}'"}), 400

    # Get trained actions
    policy = model.policy.to(device)
    out = policy(td_init.clone(), infer_env, phase="test", decode_type="greedy", return_actions=True)
    actions_trained = out['actions'].cpu().detach()

    # Process the actions_trained and get the list of absolute node ids
    action_ids = infer_env.render(td_init[0], actions_trained[0])

    return jsonify({"status": "success", "action_ids": action_ids})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)