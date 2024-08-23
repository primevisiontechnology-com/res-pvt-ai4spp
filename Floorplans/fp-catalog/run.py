from lib.plotter.fp import FloorplanPlotter
from lib.utils import get_paths, gen_pdf
import uuid

if __name__ == "__main__":
    floorplans = get_paths("floorplan.json", "../", recursive=True, depth=1)
    ids = []
    for floorplan in floorplans:
        plotter = FloorplanPlotter(floorplan)
        ids.append(str(uuid.uuid1())+floorplan.split("/")[-2])
        plotter.plot_design()
        plotter.save("results/"+ids[-1])
        plotter.close()

    # gen_pdf(ids, "results/")
    # plotter = FloorplanPlotter("../USPS/00xx_Garside/floorplan.json")
    # plotter.plot_design()
    # plotter.save("results/test.png")