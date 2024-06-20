import json
import math
import numpy as np
from typing import List, Dict

def rotate(xy, xy0, theta):  # rotate x,y around xo,yo by theta (rad)
    x, y = xy
    x0, y0 = xy0
    xr = math.cos(theta) * (x - x0) - math.sin(theta) * (y - y0) + x0
    yr = math.sin(theta) * (x - x0) + math.cos(theta) * (y - y0) + y0
    return np.array([xr, yr])

class Floorplan:
    def __init__(self, data_path: str):
        self.zones = {}
        self.cells = {}
        self.directions = {}
        self.parse_zones_nodes(data_path)
        #self.parse_directions()

    def parse_zones_nodes(self, data_path: str):
        with open(data_path, "r") as json_file:
            self.data = json.load(json_file)

        if "zones" not in self.data:
            raise Exception("No zones in design")

        for zone in self.data["zones"]:
            zone_obj = Zone(zone["id"], zone["type"], zone["pose"])
            self.zones[zone_obj.getId()] = zone_obj
            for cell in zone["nodes"]:
                absolute_id = f'/{zone["id"]}/{cell["id"]}'
                absolute_pose = [
                    cell["pose"][i] + zone_pose
                    for i, zone_pose in enumerate(zone["pose"])
                ]
                cell_obj = Cell(
                    zone_obj,
                    absolute_id,
                    cell["type"],
                    absolute_pose,
                    cell["connections"],
                )
                self.cells[cell_obj.getId()] = cell_obj

    def parse_directions(self):
        if "directions" not in self.data:
            raise Exception("No directions in design")

        for id, direction in self.data["directions"].items():
            if "sub_directions" not in direction:
                continue
            sub_directions = []
            for direction_id, data in direction["sub_directions"].items():
                sub_direction = Direction(
                    id,
                    direction_id,
                    data["side"],
                    data["container"]["type"],
                    self.cells,
                )
                sub_directions.append(sub_direction)
            self.directions[id] = sub_directions

    def getCellFromId(self, id: str):
        return self.cells[id]

    def getZoneFromId(self, id: str):
        return self.zones[id]

    def getCells(self):
        return self.cells

    def getDirectionsList(self):
        return self.directions.values()


class Zone:
    def __init__(self, id: str, zone_type: str, zone_pose: List[float]):
        self.id = id
        self.zone_type = zone_type
        self.zone_pose = np.array(zone_pose, dtype=np.float64)

    def getId(self):
        return self.id

    def getType(self):
        return self.zone_type

    def getPose(self):
        return self.pose


class Cell:
    def __init__(
        self, zone: str, id: str, cell_type: str, pose: List[float], connections: dict
    ):
        self.zone = zone
        self.id = id
        self.cell_type = cell_type
        self.pose = np.array(pose, dtype=np.float64)
        self.connections = connections

    def getId(self):
        return self.id

    def getZoneId(self):
        return self.zone.getId()

    def getType(self):
        return self.cell_type

    def getPose(self):
        return self.pose

    def getConnections(self):
        return self.connections


class Direction:
    def __init__(
        self,
        id: str,
        direction_id: str,
        side: str,
        direction_type: str,
        cell_lookup: Dict[str, Cell],
    ):
        self.id = id
        self.direction_id = direction_id
        self.side = int(side)
        self.direction_type = direction_type
        self.cell_lookup = cell_lookup

    def calculatePose(self, cell_height: float, direction_height: float, buffer: float = 0.05):
        cell_pose = self.cell_lookup[self.id].getPose()
        cell_xy = cell_pose[0:2]
        theta = cell_pose[2]

        xy_aligned = cell_xy - self.side * 0.5 * np.array(
            [0, cell_height + direction_height + buffer]
        )
        xy = rotate(xy_aligned, cell_xy, theta)
        self.pose = np.array([xy[0], xy[1], theta])

    def getId(self):
        return self.id

    def getDirectionId(self):
        return self.direction_id

    def getSide(self):
        return self.side

    def getType(self):
        return self.direction_type

    def getPose(self):
        return self.pose
