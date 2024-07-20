import os.path

from plotter.fp import FloorplanPlotter
from utils import get_paths, gen_pdf
import uuid

if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")

    floorplans = get_paths("floorplan.json", "../", recursive=True, depth=1)
    ids = []
    for floorplan in floorplans:
        try:
            plotter = FloorplanPlotter(floorplan)
            ids.append(str(uuid.uuid1())+floorplan.split("/")[-2])
            plotter.plot_design()
            plotter.save("results/"+ids[-1])
            plotter.close()
        except Exception as e:
            print(f"An error occurred: {e}")

    # gen_pdf(ids, "results/")

    # plotter = FloorplanPlotter("../Floorplans/USPS/0026_Salt_Lake_City/floorplan.json")
    # plotter.plot_design()
    # plotter.save("results/test.png")