from FPenv import FPEnv
from SPPv2env import SPPv2Env
import torch

if __name__ == "__main__":
    env_SPP = SPPv2Env(num_loc=100)
    td_SPP = env_SPP.generate_data(batch_size=[3])
    ax_fp = env_SPP.plot_graph(td_SPP["locs"], td_SPP["edges"], ax=None)

    # env_fp = FPEnv(fp_path="floorplan_SLC.json")
    # td_fp = env_fp.generate_data(batch_size=3)
    # ax_fp = env_fp.plot_graph(td_fp["locs"], td_fp["edges"], ax=None)