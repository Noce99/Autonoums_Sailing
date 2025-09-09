import math
from abc import abstractmethod
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from getgfs import getgfs
from guppy import hpy

from Sailing_Boats_Autopilot.resources import colormaps as cmaps

import torch

from Sailing_Boats_Autopilot.geometry.grid_creation import bilinear_interpolation, greed_point_coordinate
from Sailing_Boats_Autopilot.utils import m_to_lat, m_to_lon, lat_converter, lon_converter
from Sailing_Boats_Autopilot.visualizer.sailing_visualizer import Visualizer
from Sailing_Boats_Autopilot.constants import MAX_WIND_SPEED, ABSOLUTE_PATH
from Sailing_Boats_Autopilot.resources.boats_polars.load_polar import load_polar


class Map:
    def __init__(self):
        self.data = None

    @abstractmethod
    def interpolate(self, coordinates):
        pass

    def __getitem__(self, coordinates):
        if self.data is None:
            print("Data of the map were not initialized! Returning None!")
            return None
        if len(coordinates) > 2:
            print(f'\"def __getitem__(self, indexes):\" received {len(coordinates)} indexes but need two of them'
                  f' (dropping the last {len(coordinates) - 2})!')
        if len(coordinates) < 2:
            print(f'\"def __getitem__(self, indexes):\" received {len(coordinates)} indexes but 2 are needed! '
                  f'Returning None!')
            return None
        return self.interpolate((coordinates[0], coordinates[1]))


class FileMap(Map):

    def __init__(self, path, greenwich_normalized=True):
        super().__init__()
        self.data = np.load(path).astype(float)  # min: 0, max: 1, shape: (16384, 32768), unique: [0. 1.]
        if not greenwich_normalized:
            self.data = np.concatenate(
                (self.data[:, self.data.shape[1] // 2:], self.data[:, 0:self.data.shape[1] // 2]),
                axis=1)
        # print(f"Loading a FileMap with shape: {self.data.shape}")
        self.lat_min = None
        self.lat_max = None
        self.lon_min = None
        self.lon_max = None
        self.lat_index_min = None
        self.lat_index_max = None
        self.lon_index_min = None
        self.lon_index_max = None

    def get_sub_map(self, grid):
        output_latitude_longitude = grid
        result = np.zeros((output_latitude_longitude[:, :, 0].shape[0], output_latitude_longitude[:, :, 0].shape[1]),
                          dtype=bool)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = self[output_latitude_longitude[i, j, 0], output_latitude_longitude[i, j, 1]]
        return result

    def interpolate(self, coordinates):
        latitude, longitude = coordinates
        height = self.data.shape[0]
        width = self.data.shape[1]
        float_indexes = (latitude / np.pi * (height - 1), longitude / (2 * np.pi) * (width - 1))
        # print(f"Angle: {coordinates} Float Indexes: {float_indexes}")

        # Credo di non usare quanto segue:
        if self.lat_min is None or self.lat_min > latitude:
            self.lat_min = latitude
        if self.lat_max is None or self.lat_max < latitude:
            self.lat_max = latitude
        if self.lon_min is None or self.lon_min > longitude:
            self.lon_min = longitude
        if self.lon_max is None or self.lon_max < longitude:
            self.lon_max = longitude
        if self.lat_index_min is None or self.lat_index_min > float_indexes[0]:
            self.lat_index_min = float_indexes[0]
        if self.lat_index_max is None or self.lat_index_max < float_indexes[0]:
            self.lat_index_max = float_indexes[0]
        if self.lon_index_min is None or self.lon_index_min > float_indexes[1]:
            self.lon_index_min = float_indexes[1]
        if self.lon_index_max is None or self.lon_index_max < float_indexes[1]:
            self.lon_index_max = float_indexes[1]
        # fino a qui!

        x = float_indexes[0]
        y = float_indexes[1]
        x1 = int(x)
        if x1 == width:
            x2 = 0
        else:
            x2 = x1 + 1
        y1 = int(y)
        if y1 == height:
            y2 = 0
        else:
            y2 = y1 + 1
        f11 = self.data[x1, y1]
        f12 = self.data[x1, y2]
        f22 = self.data[x2, y2]
        f21 = self.data[x2, y1]
        interpolation = bilinear_interpolation(x, y, x1, x2, y1, y2, f11, f12, f22, f21)
        return interpolation

    def print_intervals(self):
        print(f"lat = [{self.lat_min}, {self.lat_max}] [{self.lat_min / np.pi * 180}, {self.lat_max / np.pi * 180}]")
        print(f"lon = [{self.lon_min}, {self.lon_max}] [{self.lon_min / np.pi * 180}, {self.lon_max / np.pi * 180}]")
        print(f"lat_index = [{self.lat_index_min}, {self.lat_index_max}]")
        print(f"lon_index = [{self.lon_index_min}, {self.lon_index_max}]")


class WindMap(Map):

    def __init__(self, time):
        super().__init__()
        self.forecast = getgfs.Forecast("0p25")
        self.time = time
        self.get_live_data()

    def get_live_data(self):
        wind_data = self.forecast.get(["ugrd10m", "vgrd10m"],
                                      self.time.strftime("%Y%m%d %H:%M"), "[-90:90]", "[0:359.9999]")
        self.data = np.concatenate((np.flip(wind_data.variables["ugrd10m"].data, 1),
                                    np.flip(wind_data.variables["vgrd10m"].data, 1)))
        # SHAPE: (2, 721, 1425)

    def get_sub_map(self, grid):
        output_latitude_longitude = grid
        result = np.zeros((output_latitude_longitude[:, :, 0].shape[0], output_latitude_longitude[:, :, 0].shape[1], 2),
                          dtype=float)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                u, v = self[output_latitude_longitude[i, j, 0], output_latitude_longitude[i, j, 1]]
                result[i, j, 0] = u
                result[i, j, 1] = v
        return result

    def interpolate(self, coordinates):
        latitude, longitude = coordinates
        height = self.data.shape[1]
        width = self.data.shape[2]
        float_indexes = (latitude / np.pi * (height - 1), longitude / (2 * np.pi) * (width - 1))
        # print(f"Angles: {coordinates} Float Indexes: {float_indexes}")
        x = float_indexes[0]
        y = float_indexes[1]
        x1 = int(x)
        if x1 == width:
            x2 = 0
        else:
            x2 = x1 + 1
        y1 = int(y)
        if y1 == height:
            y2 = 0
        else:
            y2 = y1 + 1
        f11 = self.data[0, x1, y1]
        f12 = self.data[0, x1, y2]
        f22 = self.data[0, x2, y2]
        f21 = self.data[0, x2, y1]
        interpolation_u = bilinear_interpolation(x, y, x1, x2, y1, y2, f11, f12, f22, f21)
        f11 = self.data[1, x1, y1]
        f12 = self.data[1, x1, y2]
        f22 = self.data[1, x2, y2]
        f21 = self.data[1, x2, y1]
        interpolation_v = bilinear_interpolation(x, y, x1, x2, y1, y2, f11, f12, f22, f21)
        return interpolation_u, interpolation_v


class WindHistoricMap(Map):

    def __init__(self, random=True):
        super().__init__()
        self.random = random
        self.wind_data_u = None
        self.wind_data_v = None
        self.wind_datatime = None
        self.load_historical_data()

    def load_historical_data(self):
        if self.random:
            self.wind_data_u = torch.flip(torch.load(f"{ABSOLUTE_PATH}/wind_data/50_random_u.tensor"), [1])
            # SHAPE: (30, 721, 1425)
            self.wind_data_v = torch.flip(torch.load(f"{ABSOLUTE_PATH}/wind_data/50_random_v.tensor"), [1])
            # SHAPE: (30, 721, 1425)
            self.wind_datatime = torch.flip(torch.load(f"{ABSOLUTE_PATH}/wind_data/50_random_d.tensor"), [1])
        else:
            self.wind_data_u = torch.flip(torch.load(f"{ABSOLUTE_PATH}/wind_data/all_u.tensor"), [1])
            self.wind_data_v = torch.flip(torch.load(f"{ABSOLUTE_PATH}/wind_data/all_v.tensor"), [1])
            self.wind_datatime = torch.flip(torch.load(f"{ABSOLUTE_PATH}/wind_data/all_dates.tensor"), [1])

    def get_sub_map(self, grid, time_index):
        output_latitude_longitude = grid
        result = torch.zeros(
            (2, output_latitude_longitude[:, :, 0].shape[0], output_latitude_longitude[:, :, 0].shape[1]),
            dtype=torch.float)
        for i in range(result.shape[1]):
            for j in range(result.shape[2]):
                u, v = self.interpolate(
                    [time_index, output_latitude_longitude[i, j, 0], output_latitude_longitude[i, j, 1]])
                result[0, i, j] = u
                result[1, i, j] = v
        return result

    def get_sub_map_without_grid(self, time_index, latitude, longitude):
        height = self.wind_data_u.shape[1]
        width = self.wind_data_u.shape[2]
        if longitude > math.pi:
            longitude -= math.pi
        elif longitude < math.pi:
            longitude += math.pi
        float_indexes = (latitude / math.pi * (height - 1), longitude / (2 * math.pi) * (width - 1))
        x = int(float_indexes[0])
        y = int(float_indexes[1])
        if x < 8 or x > height - 8:
            output = torch.zeros(2, 16, 16)
        elif y >= width - 8:
            u_1 = self.wind_data_u[time_index, x - 8:x + 8, y - 8:]
            u_2 = self.wind_data_u[time_index, x - 8:x + 8, 0:16 - u_1.shape[1]]
            u = torch.concatenate([u_1, u_2], dim=1)
            v_1 = self.wind_data_v[time_index, x - 8:x + 8, y - 8:]
            v_2 = self.wind_data_v[time_index, x - 8:x + 8, 0:16 - v_1.shape[1]]
            v = torch.concatenate([v_1, v_2], dim=1)
            output = torch.concatenate([u[None, :, :], v[None, :, :]], dim=0)
        elif y < 8:
            u_2 = self.wind_data_u[time_index, x - 8:x + 8, 0:y + 8]
            u_1 = self.wind_data_u[time_index, x - 8:x + 8, -(16 - u_2.shape[1]):]
            u = torch.concatenate([u_1, u_2], dim=1)
            v_2 = self.wind_data_v[time_index, x - 8:x + 8, 0:y + 8]
            v_1 = self.wind_data_v[time_index, x - 8:x + 8, -(16 - v_2.shape[1]):]
            v = torch.concatenate([v_1, v_2], dim=1)
            output = torch.concatenate([u[None, :, :], v[None, :, :]], dim=0)
        else:
            output = torch.concatenate([self.wind_data_u[time_index, x - 8:x + 8, y - 8:y + 8][None, :, :],
                                        self.wind_data_v[time_index, x - 8:x + 8, y - 8:y + 8][None, :, :]], dim=0)
        if output.shape[2] > 20:
            print(x)
            print(y)

            exit()
        return output

    def interpolate(self, coordinates):
        i, latitude, longitude = coordinates
        height = self.wind_data_u.shape[1]
        width = self.wind_data_u.shape[2]
        float_indexes = (latitude / math.pi * (height - 1), longitude / (2 * math.pi) * (width - 1))
        # print(f"Angles: {coordinates} Float Indexes: {float_indexes}")
        x = float_indexes[0]
        y = float_indexes[1]
        x1 = int(x)
        if x1 == width:
            x2 = 0
        else:
            x2 = x1 + 1
        y1 = int(y)
        if y1 == height:
            y2 = 0
        else:
            y2 = y1 + 1
        if x1 > height or y1 > width or x2 > height or y2 > width:
            print(f"[{latitude}; {longitude}], [{x1}; {y1}; {x2}; {y2}]")
        f11 = self.wind_data_u[i, x1, y1]
        f12 = self.wind_data_u[i, x1, y2]
        f22 = self.wind_data_u[i, x2, y2]
        f21 = self.wind_data_u[i, x2, y1]
        interpolation_u = bilinear_interpolation(x, y, x1, x2, y1, y2, f11, f12, f22, f21)
        f11 = self.wind_data_v[i, x1, y1]
        f12 = self.wind_data_v[i, x1, y2]
        f22 = self.wind_data_v[i, x2, y2]
        f21 = self.wind_data_v[i, x2, y1]
        interpolation_v = bilinear_interpolation(x, y, x1, x2, y1, y2, f11, f12, f22, f21)
        return interpolation_u, interpolation_v


class SailingMap:

    def __init__(self, start_point, target_point, starting_time=None):
        """
        latitude in [-Pi/2; Pi/2]
        longitude in [-Pi; Pi]
        :param start_point: A tuple containing the latitude and longitude of the starting point (in radiant)
        :param target_point: A tuple containing the latitude and longitude of the end point
        :param starting_time: A datetime object of the starting time
        """
        self.start_point = start_point
        self.boat_position = start_point
        self.boat_orientation = None
        self.target_lat, self.target_lon = target_point
        # self.FM = FileMap(f"{ABSOLUTE_PATH}/resources/earth_and_see/earth_and_see.npy", greenwich_normalized=False)
        self.FM = None
        if starting_time is None:
            starting_time = datetime.now()
        self.WM = WindHistoricMap(starting_time)
        self.cmap = cmaps.viridis

    def get_boat_image_position(self, grid):
        difference = np.abs(grid[:, :, 0] - self.boat_position[0]) + np.abs(grid[:, :, 1] - self.boat_position[1])
        argmin_result = np.argmin(difference)
        raw = argmin_result // grid.shape[0]
        column = argmin_result - raw * grid.shape[0]
        if 3 < raw < (grid.shape[0] - 3) and 3 < column < (grid.shape[1] - 3):
            return raw / grid.shape[0], column / grid.shape[1]
        else:
            return None

    def update_boat_position(self, boat_position):
        self.boat_position = boat_position

    def get_earth_boolean(self, grid):
        """
        :param grid: result of "greed_point_coordinate((Latitude, Longitude), half_size_in_px, half_size_in_km)"
        :return: a boolean numpy array of size (2*half_size_in_px+1, 2*half_size_in_px+1) [True where there is earth and
         False where there is water]
        """
        return self.FM.get_sub_map(grid)

    def get_earth_rgb(self, grid):
        """
        :param grid: result of "greed_point_coordinate((Latitude, Longitude), half_size_in_px, half_size_in_km)"
        :return: an int numpy array of size (2*half_size_in_px+1, 2*half_size_in_px+1, 2) [(u, v) wind velocity in m/s
        with both positive and negative values]
        """
        bool_heart = self.FM.get_sub_map(grid)
        int_heart = bool_heart.astype("uint8") * 100
        rgb_heart = np.concatenate([int_heart[:, :, None], int_heart[:, :, None], int_heart[:, :, None]], axis=-1)
        return rgb_heart

    def get_wind_raw(self, grid):
        """
        :param grid: result of "greed_point_coordinate((Latitude, Longitude), half_size_in_px, half_size_in_km)"
        :return: an int numpy array of size (2*half_size_in_px+1, 2*half_size_in_px+1, 3) [(255, 255, 255) where there is
        earth and (0, 0, 0) where there is water]
        """
        return self.WM.get_sub_map(grid)

    def get_u_wind_raw(self, grid):
        """
        :param grid: result of "greed_point_coordinate((Latitude, Longitude), half_size_in_px, half_size_in_km)"
        :return: an int numpy array of size (2*half_size_in_px+1, 2*half_size_in_px+1, 3) [(255, 255, 255) where there is
        earth and (0, 0, 0) where there is water]
        """
        return self.WM.get_sub_map(grid)[:, :, 0]

    def get_v_wind_raw(self, grid):
        """
        :param grid: result of "greed_point_coordinate((Latitude, Longitude), half_size_in_px, half_size_in_km)"
        :return: an int numpy array of size (2*half_size_in_px+1, 2*half_size_in_px+1, 3) [(255, 255, 255) where there is
        earth and (0, 0, 0) where there is water]
        """
        return self.WM.get_sub_map(grid)[:, :, 1]

    def get_wind_rgb(self, grid):
        """
        :param grid: result of "greed_point_coordinate((Latitude, Longitude), half_size_in_px, half_size_in_km)"
        :return:
        """
        unnormalized_result = self.WM.get_sub_map(grid)
        normalized_result_0 = (unnormalized_result[:, :, 0] + MAX_WIND_SPEED // 2) / MAX_WIND_SPEED
        normalized_result_1 = (unnormalized_result[:, :, 1] + MAX_WIND_SPEED // 2) / MAX_WIND_SPEED
        rgb_0 = self.cmap(normalized_result_0)
        rgb_1 = self.cmap(normalized_result_1)
        return np.concatenate([rgb_0[:, :, None, :], rgb_1[:, :, None, :]], axis=-2)

    def get_u_wind_rgb(self, grid):
        """
        :param grid: result of "greed_point_coordinate((Latitude, Longitude), half_size_in_px, half_size_in_km)"
        :return:
        """
        unnormalized_result = self.WM.get_sub_map(grid)
        normalized_result_0 = (unnormalized_result[:, :, 0] + MAX_WIND_SPEED // 2) / MAX_WIND_SPEED
        rgb_0 = self.cmap(normalized_result_0)
        rgb_0[:, :, 3] = 0.8
        return (rgb_0 * 255).astype("uint8")

    def get_v_wind_rgb(self, grid):
        """
        :param grid: result of "greed_point_coordinate((Latitude, Longitude), half_size_in_px, half_size_in_km)"
        :return:
        """
        unnormalized_result = self.WM.get_sub_map(grid)
        normalized_result_1 = (unnormalized_result[:, :, 1] + MAX_WIND_SPEED // 2) / MAX_WIND_SPEED
        rgb_1 = self.cmap(normalized_result_1)
        rgb_1[:, :, 3] = 0.8
        return (rgb_1 * 255).astype("uint8")

    def get_speed_rgb(self, grid):
        unnormalized_result = self.WM.get_sub_map(grid)
        unnormalized_speed = np.sqrt(np.square(unnormalized_result[:, :, 0]) + np.square(unnormalized_result[:, :, 1]))
        normalized_speed = (unnormalized_speed + MAX_WIND_SPEED // 2) / MAX_WIND_SPEED
        rgb = self.cmap(normalized_speed)
        rgb[:, :, 3] = 0.8
        return (rgb * 255).astype("uint8")

    def get_target_angular_distance_normalized(self, grid):
        """
        http://astrophysicsformulas.com/astronomy-formulas-astrophysics-formulas/angular-distance-between-two-points-on-a-sphere/
        https://math.stackexchange.com/questions/231221/great-arc-distance-between-two-points-on-a-unit-sphere
        :param grid: result of "greed_point_coordinate((Latitude, Longitude), half_size_in_px, half_size_in_km)"
        :return:
        """
        result = torch.zeros((grid[:, :, 0].shape[0], grid[:, :, 0].shape[1]), dtype=torch.float)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = math.acos(math.sin(grid[i, j, 0]) * math.sin(self.target_lat) +
                                         math.cos(grid[i, j, 0]) * math.cos(self.target_lat) *
                                         math.cos(grid[i, j, 1] - self.target_lon))
        result = result / result.max()
        return result

    def get_wind_normalized_tensor(self, grid):
        result = torch.zeros((2, grid[:, :, 0].shape[0], grid[:, :, 0].shape[1]), dtype=torch.float)
        for i in range(result.shape[1]):
            for j in range(result.shape[2]):
                u, v = self.WM[grid[i, j, 0], grid[i, j, 1]]
                result[0, i, j] = u
                result[1, i, j] = v
        abs_max = max([torch.abs(result.max()), torch.abs(result.min())])
        result = result / abs_max
        return result

    """
    @staticmethod
    def get_distance_tensor(grid, target_position):
        result = torch.zeros((grid[:, :, 0].shape[0], grid[:, :, 0].shape[1]), dtype=torch.float)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = math.sqrt((target_position[0] - grid[i, j, 0]) ** 2 +
                                         (target_position[1] - grid[i, j, 1]) ** 2) / (2*math.pi)
        return result
    """


class SailingBoat:

    def __init__(self, position, orientation=0., boat_name="bavaria_39", delta_orientation=math.pi / 16):
        """
        latitude in [-Pi/2; Pi/2]
        longitude in [-Pi; Pi]
        orientation in [0, 2Pi]
        :param position: A tuple containing the latitude and longitude of the starting point
        :param orientation: The clockwise angle respect to NORTH in rad
        """
        self.position = position
        self.orientation = orientation
        self.polar = load_polar(boat_name)
        self.delta_orientation = delta_orientation
        self.boat_speed = 0

    def turn_clockwise(self):
        self.orientation += self.delta_orientation
        while self.orientation >= 2 * math.pi:
            self.orientation -= 2 * math.pi

    def turn_anticlockwise(self):
        self.orientation -= self.delta_orientation
        while self.orientation < 0:
            self.orientation += 2 * math.pi

    def integrate(self, wind, dt):
        """
        Integrate the sailing boat model using wind, orientation and boat polar graph
        :param dt: Time step in s
        :param wind: Tuple [(u, v)] of wind intensity in m/s
        :return: None
        """
        wind_angle = math.atan2(wind[1], wind[0]) + 3 / 2 * math.pi
        if wind_angle > 2 * math.pi:
            wind_angle = wind_angle - 2 * math.pi
        wind_angle = 2 * math.pi - wind_angle
        # print(f"wind: [{wind[0]}, {wind[1]}]")
        # print(f"wind_angle = {wind_angle/math.pi*180}")
        wind_orientation_angle = abs(wind_angle - self.orientation)
        if wind_orientation_angle > math.pi:
            wind_orientation_angle = 2 * math.pi - wind_orientation_angle
        wind_orientation_angle = math.pi - wind_orientation_angle  # in [0; math.pi]
        wind_speed = math.sqrt(wind[0] ** 2 + wind[1] ** 2)
        self.boat_speed = self.polar(wind_speed, wind_orientation_angle, m_s_and_radiant=True)
        movement = self.boat_speed * dt
        dx = movement * math.sin(self.orientation)  # lon movement in m
        dy = movement * math.cos(self.orientation)  # lat movement in m
        dlon = m_to_lon(dx, self.position)
        dlat = m_to_lat(dy)
        # print(f"wind_angle = {wind_angle / math.pi * 180} [{wind_speed} m/s]")
        # print(f"boat_angle = {self.orientation / math.pi * 180}")
        # print(f"wind_orientation_angle = {wind_orientation_angle / math.pi * 180}")
        # print(f"The boat is moving of ({dx}, {dy}) m, equivalent to ({dlat},"
        #       f" {dlon}) latitude and longitude rad.")
        # print(f"In {dt} second with an average speed of {math.sqrt(dx ** 2 + dy ** 2) / dt} m/s")
        self.position = [self.position[0] - dlat, self.position[1] + dlon]
        while self.position[0] < 0:
            self.position[0] += math.pi
        while self.position[0] > math.pi:
            self.position[0] -= math.pi
        while self.position[1] < 0:
            self.position[1] += 2 * math.pi
        while self.position[1] > 2 * math.pi:
            self.position[1] -= 2 * math.pi


if __name__ == "__main__":
    """
    grid = greed_point_coordinate((0.785398, 0.1745329), 100, 600)
    SM = SailingMap((0, 0), (0.785398, 0.1745329))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1.imshow(SM.get_u_wind_raw(grid))
    ax2.imshow(SM.get_v_wind_raw(grid))
    ax3.imshow(SM.get_u_wind_rgb(grid))
    ax4.imshow(SM.get_v_wind_rgb(grid))
    plt.show()
    """
    """
    STARTING_POSITION = [lat_converter(0, 0, 0, "N"), lon_converter(0, 0, 0, "E")]
    print(f"Starting Point: {STARTING_POSITION}")
    my_boat = SailingBoat(STARTING_POSITION, orientation=90 / 180 * math.pi)
    vi = Visualizer(SailingMap(STARTING_POSITION, (0, 0)), my_boat)
    vi.ui()
    """
    WHM = WindHistoricMap()
    grid = greed_point_coordinate(boat_position=(0.785398, 0.1745329), greed_size=100, max_greed_distance_km=1000)
    to_plot = WHM.get_sub_map(grid, time_index=0)
    plt.imshow(to_plot[1])
    plt.show()
