# !pip install rl4co==0.3.3
# !pip install torch==2.3.0

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

# Global variables to store received floorplan data, action_ids, and cost
global_floorplan_data = None
global_action_ids = None
global_cost = None


# Check for NaNs in a tensor dictionary
def check_for_nans(td):
    for key, tensor in td.items():
        if torch.isnan(tensor).any():
            return key
    return None


@app.route("/")
def index():
    return "<h1>The Reinforcement Learning Server is Running!</h1>"

# This endpoint method let the client POST floorplan to the app
@app.route('/upload_floorplan', methods=['POST'])
def upload_floorplan():
    """
    Endpoint to upload the floorplan.
    """
    global global_floorplan_data

    # The entire request body is the floorplan data
    floorplan_data = request.json

    if not floorplan_data:
        return jsonify({"status": "error", "message": "Floorplan data is required"}), 400

    # Assign to global variables
    global_floorplan_data = floorplan_data
    return jsonify({"status": "success", "message": "Floorplan data received successfully"})


# This endpoint method let the client POST start_node_id and target_node_id to the app
@app.route('/upload_node_ids', methods=['POST'])
def upload_node_ids():
    """
    Endpoint to retrieve action_ids.
    """

    # Ensure that the global variables are referenced here
    global global_floorplan_data
    global global_action_ids
    global global_cost

    # Check if the floorplan data is there
    if global_floorplan_data is None:
        return jsonify({"status": "error", "message": "Floorplan data is not yet uploaded"}), 400

    # Extract start_node_id and target_node_id from the request body
    start_node_id = request.json.get('start_node_id', None)
    target_node_id = request.json.get('target_node_id', None)

    # Check if start_node_id and target_node_id are received
    if not start_node_id:
        return jsonify({"status": "error", "message": "start_node_id is required"}), 400
    if not target_node_id:
        return jsonify({"status": "error", "message": "target_node_id is required"}), 400

    # Create the environment with the provided fp_path
    infer_env = FPEnvPlanner(floorplan_data=global_floorplan_data, start_node_id=start_node_id, target_node_id=target_node_id)
    td_init = infer_env.reset().to(device)

    # Check for NaNs
    nan_key = check_for_nans(td_init)
    if nan_key:
        return jsonify({"status": "error", "message": f"NaN values found in tensor '{nan_key}'"}), 400

    # Get trained actions
    policy = model.policy.to(device)
    out = policy(td_init.clone(), infer_env, phase="test", decode_type="greedy", return_actions=True)
    actions_trained = out['actions'].cpu().detach()

    # Process the actions_trained and get the list of absolute node ids, then save to the global variable
    action_ids = infer_env.render(td_init[0], actions_trained[0])
    global_action_ids = action_ids

    # Compute the total cost, then save to the global variable
    cost = -out['reward'][0].item()
    global_cost = cost

    return jsonify({"action_ids": action_ids}), 200


# This endpoint method let the client GET action_ids from the app
@app.route('/get_action_ids', methods=['GET'])
def get_action_ids():
    global global_action_ids  # Reference the global action_ids variable
    if global_action_ids is None:
        return jsonify({"error": "Action IDs not set"}), 400

    return jsonify({"action_ids": global_action_ids}), 200


# This endpoint method let the client GET cost from the app
@app.route('/get_cost', methods=['GET'])
def get_cost():
    global global_cost  # Reference the global action_ids variable
    if global_cost is None:
        return jsonify({"error": "Cost not set"}), 400

    return jsonify({"cost": global_cost}), 200


# main method
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)