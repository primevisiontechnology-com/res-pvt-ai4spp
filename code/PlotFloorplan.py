from FPenv import FPEnv
import torch

if __name__ == "__main__":
    env = FPEnv(fp_path="floorplan_SLC.json")
    td = env.generate_data(batch_size=3)
    env.plot_graph(td["locs"], td["edges"], ax=None)