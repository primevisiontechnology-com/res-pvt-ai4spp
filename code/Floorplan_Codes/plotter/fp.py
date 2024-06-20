import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, PathPatch
from matplotlib.path import Path

from lib.floorplan import Floorplan

base_style = {
    "node": {
        "width": 0.85,
        "height": 0.5,
        "linewidth": 0.5,
        "linestyle": "-",
        "fill": False,
    },
    "error": {"width": 0.85, "height": 0.5, "linewidth": 0.2, "alpha": 0.5},
    "dropoff-box": {
        "width": 0.9,
        "height": 0.9,
        "linewidth": 0.5,
        "linestyle": "-",
        "fill": False,
    },
    "pinpoint": {"radius": 0.3}
}

element_styles = {
    "init": {**base_style["node"], **{"color": "black"}},
    "target": {**base_style["node"], **{"color": "darkblue"}},
    "entry_and_exit": {**base_style["node"], **{"color": "purple"}},
    "dropoff-box": {**base_style["dropoff-box"], **{"color": "blue"}},
    "Lidar error": {**base_style["error"], **{"color": "yellow"}},
    "Battery empty": {**base_style["error"], **{"color": "orange"}},
    "Position error": {**base_style["error"], **{"color": "pink"}},
    "Trajectory error": {**base_style["error"], **{"color": "green"}},
    "Emergency stop": {**base_style["error"], **{"color": "red"}},
    "UNRESOLVED_ERROR": {**base_style["error"], **{"color": "black"}},
    "Stop location": {**base_style["pinpoint"], **{"color": "red"}}
}

class PlotRectanglePatch:
    def __init__(self, object_type, pose, element_styles):
        self.xy = pose[0:2]
        self.theta = pose[2]
        self.label = object_type
        self.styles = element_styles

        self.width = self.styles[self.label]["width"]
        self.height = self.styles[self.label]["height"]
        self.corner_xy = self.xy - 0.5 * np.array([self.width, self.height])

    def plot(self, ax):
        patch = Rectangle(
            xy=(self.corner_xy),
            angle=math.degrees(self.theta),
            label=self.label,
            rotation_point="center",
            **self.styles[self.label],
        )
        ax.add_patch(patch)

class PlotCirclePatch:
    def __init__(self, object_type, pose, element_styles):
        self.xy = pose
        self.label = object_type
        self.styles = element_styles
        self.radius = self.styles[self.label]["radius"]

    def plot(self, ax):
        patch = Circle(
            xy=(self.xy),
            label=self.label,
            **self.styles[self.label]
        )
        ax.add_patch(patch)

class FloorplanPlotter:
    def __init__(self, design_path: str, element_styles: dict = element_styles):
        self.fig, self.ax = plt.subplots(figsize=(8,7))
        self.ax.set_title("{} : {}".format(design_path.split("/")[-3],design_path.split("/")[-2]))
        self.design = Floorplan(design_path)
        self.all_cells = self.design.getCells()
        self.styles = element_styles

    def plot_design(self, show_connections: bool = True):
        """
        Plots the design.
        """
        self.plot_floorplan(show_connections)
        self.plot_sortplan()

    def plot_floorplan(self, show_connections: bool = True):
        """
        Plots the floorplan.

        Parameters
        ----------
        show_connections : bool, optional
            Whether to show the connections between cells, by default True
        """
        for cell in self.all_cells.values():
            cell_patch = PlotRectanglePatch(cell.getType(), cell.getPose(), self.styles)
            cell_patch.plot(self.ax)

            if show_connections:
                self._plot_connections(cell)
    
    def plot_sortplan(self):
        """
        Plots the sortplan.
        """
        for cell_directions in self.design.getDirectionsList():
            for sub_direction in cell_directions:
                # calculate the pose of the direction based on the cell and the direction height
                # TODO: where should this be defined?
                cell_height = self.styles["target"]["height"]
                direction_height = self.styles[sub_direction.getType()]["height"]
                sub_direction.calculatePose(cell_height, direction_height)

                direction_patch = PlotRectanglePatch(
                    sub_direction.getType(), sub_direction.getPose(), self.styles
                )
                direction_patch.plot(self.ax)

    def plot_errors_overlay(self, location_errors_df: pd.DataFrame):
        """
        Plots the errors on top of the design.

        Parameters
        ----------
        location_errors_df : pd.DataFrame
            A dataframe containing the errors to plot. The dataframe must have the following columns:
            - value.name: the error type (e.g. "Battery empty")
            - value.pose: the pose of the error
        """
        for i in location_errors_df.index:
            error = location_errors_df.loc[i]
            error_patch = PlotRectanglePatch(
                error["value.name"], error["value.pose"], self.styles
            )
            error_patch.plot(self.ax)

    def plot_stop_locations(self, stop_location_df: pd.DataFrame):
        """
        Plots the locations where robots stand still on top of the design.

        Parameters
        ----------
        stop_location_df : pd.DataFrame
            A dataframe containing the locations to plot. The dataframe must have the following columns:
            - robot_state-position-x: x coordinate of the stop location
            - robot_state-position-y: y coordinate of the stop location
            - freq: frequency of the x and y combination 
        """
        max_freq = stop_location_df['freq'].max()
        for i in stop_location_df.index:
            stop_location = stop_location_df.loc[i]
            styles = self.styles
            styles["Stop location"]["alpha"] = 0.9 * (stop_location['freq'] / max_freq) + 0.1
            if stop_location['freq'] > 10:
                location_patch = PlotCirclePatch(
                    "Stop location", (stop_location['robot_state-position-x'], stop_location['robot_state-position-y']), styles
                )
                location_patch.plot(self.ax)

    def show(self):
        self.ax.axis("equal")
        plt.show()

    def save(self, path: str):
        self.ax.axis("equal")
        plt.savefig(path)

    def close(self):
        plt.close()

    def _add_legend(self):
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())
        self.ax.axis("equal")

    def _plot_connections(self, cell):
        """
        Plots the connections of a cell.
        Current implementation fakes the feasible path with Bezier curves for connections with 1 control point.

        Parameters
        ----------
        cell : Cell
            The cell to plot the connections for
        """
        for connection in cell.connections:
            cell_xy = cell.pose[0:2]
            vertices = [cell_xy]
            codes = [Path.MOVETO]
            next_node_id = f'/{cell.getZoneId()}/{connection["connects_to"]}'
            if next_node_id not in self.all_cells:
                continue
            next_node = self.all_cells[next_node_id]
            next_node_xy = next_node.pose[0:2]
            if len(connection["control_points"]) > 0:
                for cp in connection["control_points"]:
                    cp_xy = np.array(cp["pose"][0:2])
                    self.ax.add_patch(Circle(xy=(cp_xy), radius=0.05))
                    vertices.append(cell_xy + (cp_xy - cell_xy) * 0.90)
                    vertices.append(next_node_xy - (next_node_xy - cp_xy) * 0.90)
                    codes.append(Path.CURVE4)
                    codes.append(Path.CURVE4)
                vertices.append(next_node_xy)
                codes.append(Path.CURVE4)
            else:
                vertices.append(next_node_xy)
                codes.append(Path.LINETO)

            path = Path(vertices, codes)
            self.ax.add_patch(
                PathPatch(path, fill=False, linestyle="--", linewidth=0.2)
            )
