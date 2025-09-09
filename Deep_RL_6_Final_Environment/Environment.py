import datetime
import math
import random
import sys
import time

import cv2
import pygame
import gymnasium as gym
import numpy as np
import torch

from matplotlib import pyplot as plt
from datetime import datetime, timedelta

from Sailing_Boats_Autopilot.utils import lat_converter, lon_converter, torch_from_direction_to_ones, \
    from_start_end_to_direction, get_single_directions_embedded, random_location_in_atlantic_ocean, \
    angular_distance_degree, calculate_directions, CHECK_POINT_0, CHECK_POINT_1, CHECK_POINT_2, CHECK_POINT_3, CHECK_POINT_4,\
    CHECK_POINT_5, CHECK_POINT_6, CHECK_POINT_7, CHECK_POINT_8, CHECK_POINT_9, CHECK_POINT_10, CHECK_POINT_11
from Sailing_Boats_Autopilot.sailing_game import SailingBoat, SailingMap, WindHistoricMap
from Sailing_Boats_Autopilot.geometry.grid_creation import greed_point_coordinate
from Sailing_Boats_Autopilot.constants import ABSOLUTE_PATH


BACKGROUND_COLOR = (255, 255, 255)
TARGET_COLOR = (255, 0, 0)
AGENT_COLOR = (0, 255, 0)
LINE_COLOR = (0, 0, 0)
ARROW_COLOR = (0, 0, 255)

GRID_PIXELS_SIZE = 16  # pxs
GRID_KM_HALF_SIZE = 1000  # km

MAX_EPISODE_STEPS = 1200

BIG_WIND_MULTIPLIER = 10

DAY = datetime(year=2023, month=1, day=1, hour=8, minute=30)


class RealisticEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "no_visual"], "render_fps": 4}

    def __init__(self, start_point=None, target_point=None, start_point_target_point_function=None,
                 time_in_second_for_step=100, render_mode=None, fps=4,
                 visualizer_function=None, max_episode_steps=MAX_EPISODE_STEPS, random_wind=True):
        self.start_point_target_point_function = start_point_target_point_function
        if callable(self.start_point_target_point_function):
            self.start_point, self.target_point = self.start_point_target_point_function()
        elif start_point is None:
            self.start_point_function = None
            self.start_point = [lat_converter(64, 37, 25, "N"), lon_converter(32, 31, 10, "W")]
            # First Route : self.start_point = [lat_converter(64, 37, 25, "N"), lon_converter(32, 31, 10, "W")]
        elif callable(start_point):
            self.start_point_function = start_point
            self.start_point = self.start_point_function()
        else:
            self.start_point_function = None
            self.start_point = start_point
        if callable(self.start_point_target_point_function):
            pass
        elif target_point is None:
            self.target_point_function = None
            self.target_point = [lat_converter(44, 42, 55, "N"), lon_converter(18, 59, 3, "W")]
            # First Route : self.target_point = [lat_converter(44, 42, 55, "N"), lon_converter(18, 59, 3, "W")]
        elif callable(target_point):
            self.target_point_function = target_point
            self.target_point = self.target_point_function()
        else:
            self.target_point_function = None
            self.target_point = target_point
        self.time_in_second_for_step = time_in_second_for_step
        if render_mode is None:
            self.render_mode = "human"
        else:
            self.render_mode = render_mode
        self.metadata["render_fps"] = fps

        self.window_size = 1000
        self.window = None
        self.clock = pygame.time.Clock()
        self.episode_steps = None

        self.boat = SailingBoat(self.start_point, orientation=90 / 180 * math.pi, boat_name="vesper",
                                delta_orientation=math.pi / 16)

        self.observation = None
        self.episode_reward = None
        self.last_episode_reward = None
        self.last_reward = None

        self.random_wind = random_wind
        self.wind_map = WindHistoricMap(random=self.random_wind)
        self.last_wind_sub_map = None
        if self.random_wind:
            self.time = None
            self.time_delta_for_step = None
            self.wind_map_time_index = None
        else:
            self.time = datetime.fromisoformat("2020-11-08T18:00:00Z")
            self.time_delta_for_step = timedelta(seconds=self.time_in_second_for_step)
            self.wind_map_time_index = self.reset_wind_map_time_index()

        self.last_angular_distance = None
        self.starting_angular_distance = None
        self.last_delta_angular_distance = None

        self.visualizer_function = visualizer_function
        self.max_episode_steps = max_episode_steps

        if self.render_mode == "human" and self.visualizer_function is None:
            pygame.init()
            pygame.display.init()
            heart_map = np.load(f"{ABSOLUTE_PATH}/resources/earth_and_see/earth_and_see.npy").astype("uint8")
            heart_map = cv2.resize(heart_map, (self.window_size, self.window_size // 2), interpolation=cv2.INTER_AREA)
            # min: 0, max: 1, shape: (16384, 32768), unique: [0. 1.]
            data_rgb = np.zeros((heart_map.shape[0], heart_map.shape[1], 3))
            data_rgb[:, :, 0] = heart_map[:, :] * 255
            data_rgb[:, :, 1] = heart_map[:, :] * 255
            data_rgb[:, :, 2] = heart_map[:, :] * 255
            heart_map = data_rgb
            heart_map = np.rot90(np.flip(heart_map, 1))
            self.surf_heart_map = pygame.surfarray.make_surface(heart_map)
            del heart_map

            self.wind_arrow_image = pygame.image.load(f"{ABSOLUTE_PATH}resources/images/wind_arrow.png")
            self.boat_arrow_image = pygame.image.load(f"{ABSOLUTE_PATH}resources/images/boat_arrow.png")
            self.target_arrow_image = pygame.image.load(f"{ABSOLUTE_PATH}resources/images/target_arrow.png")
            self.boat_map_signal = pygame.image.load(f"{ABSOLUTE_PATH}resources/images/boat_map_signal.png")
            self.start_point_map_signal = pygame.image.load(f"{ABSOLUTE_PATH}resources/images/"
                                                            f"start_point_map_signal.png")
            self.target_point_map_signal = pygame.image.load(f"{ABSOLUTE_PATH}resources/images/"
                                                             f"target_point_map_signal.png")

            self.font = pygame.font.Font('freesansbold.ttf', 32)

            self.window = pygame.display.set_mode((self.window_size, self.window_size))

            self.big_wind = torch.zeros(int(GRID_PIXELS_SIZE * BIG_WIND_MULTIPLIER),
                                        int(GRID_PIXELS_SIZE * BIG_WIND_MULTIPLIER))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        distance_angle = 0
        if self.start_point_target_point_function is not None:
            self.start_point, self.target_point = self.start_point_target_point_function()
        else:
            while (distance_angle < 3 or distance_angle > 30) and self.start_point_function is not None and \
                    self.target_point_function is not None:
                # self.start_point = self.start_point_function()
                self.start_point = [lat_converter(29, 0, 0, "N"), lon_converter(45, 0, 0, "W")]
                self.target_point = self.target_point_function()
                distance_angle = angular_distance_degree(self.start_point, self.target_point)

        self.boat.position = self.start_point
        if self.random_wind is True:
            self.boat.orientation = random.randrange(0, 360) / 180 * math.pi
            self.wind_map_time_index = random.randrange(0, 30)
        else:
            self.boat.orientation = from_start_end_to_direction(self.start_point, self.target_point)
            # self.time = datetime.fromisoformat("2020-11-08T18:00:00Z")
            # self.wind_map_time_index = self.reset_wind_map_time_index()
        self.episode_steps = 0
        if self.episode_reward is not None:
            self.last_episode_reward = self.episode_reward
        else:
            self.last_episode_reward = 0
        self.episode_reward = 0

        # grid = greed_point_coordinate(self.boat.position, GRID_PIXELS_HALF_SIZE, GRID_KM_HALF_SIZE)
        # wind_map = self.map.get_wind_normalized_tensor(grid)
        # distance_map = self.map.get_target_angular_distance_normalized(grid)
        # self.observation = wind_map

        wind = self.wind_map.interpolate([self.wind_map_time_index] + self.boat.position)

        directions = calculate_directions(self.boat.orientation, self.boat.position, self.target_point, wind) \
                     / (math.pi * 2)

        speeds = torch.tensor([math.sqrt(wind[0] ** 2 + wind[1] ** 2) / 40, self.boat.boat_speed / 20])

        angular_distance = angular_distance_degree(self.boat.position, self.target_point)
        embedded_distance = torch.tensor([1.0])
        steps = torch.tensor([self.episode_steps / self.max_episode_steps])

        information = torch.concatenate([directions, speeds, embedded_distance, steps])

        # grid = greed_point_coordinate(boat_position=self.boat.position, greed_size=GRID_PIXELS_HALF_SIZE,
        #                               max_greed_distance_km=GRID_KM_HALF_SIZE)
        # self.last_wind_sub_map = self.wind_map.get_sub_map(grid, self.wind_map_time_index)
        self.last_wind_sub_map = self.wind_map.get_sub_map_without_grid(self.wind_map_time_index, self.boat.position[0],
                                                                        self.boat.position[1])

        self.observation = [information, self.last_wind_sub_map]

        self.starting_angular_distance = angular_distance
        self.last_angular_distance = angular_distance

        self.last_reward = "None"
        self.last_delta_angular_distance = "None"

        return self.observation

    def step(self, action):
        """

        :param action: [0=steering clockwise, 1=steering anticlockwise, 2=nothing]
        :return: None
        """
        self.episode_steps += 1
        # grid = greed_point_coordinate(self.boat.position, GRID_PIXELS_HALF_SIZE, GRID_KM_HALF_SIZE)
        # wind_map = self.map.get_wind_normalized_tensor(grid)
        # distance_map = self.map.get_target_angular_distance_normalized(grid)
        # self.observation = wind_map

        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(wind_map[0, :, :])
        ax2.imshow(wind_map[1, :, :])
        ax3.imshow(distance_map)
        plt.show()
        """

        if action == 0:
            self.boat.turn_clockwise()
        elif action == 1:
            self.boat.turn_anticlockwise()
        elif action == 2:
            pass
        else:
            print(f"Impossible action [{action}]")

        if self.random_wind is True:
            wind = self.wind_map.interpolate([self.wind_map_time_index] + self.boat.position)
        else:
            self.wind_map_time_index = self.get_wind_map_time_index()

            if self.wind_map_time_index is not None:
                wind = self.wind_map.interpolate([self.wind_map_time_index] + self.boat.position)
            else:
                print("Finished wind data!!")
                wind = self.wind_map.interpolate([0] + self.boat.position)

        self.boat.integrate(wind, self.time_in_second_for_step)

        if self.render_mode == "human":
            if self.visualizer_function is None:
                self.render_frame()
            else:
                self.visualizer_function()

        if self.random_wind is True:
            wind = self.wind_map.interpolate([self.wind_map_time_index] + self.boat.position)
        else:
            self.wind_map_time_index = self.get_wind_map_time_index()

            if self.wind_map_time_index is not None:
                wind = self.wind_map.interpolate([self.wind_map_time_index] + self.boat.position)
            else:
                print("Finished wind data!!")
                wind = self.wind_map.interpolate([0] + self.boat.position)

        if self.random_wind is False:
            self.time += self.time_delta_for_step

        directions = calculate_directions(self.boat.orientation, self.boat.position, self.target_point, wind) \
                     / (math.pi * 2)

        wind_speed_normalized = math.sqrt((wind[0] ** 2 + wind[1] ** 2)) / 40
        speeds = torch.tensor([wind_speed_normalized, self.boat.boat_speed / 20])

        angular_distance = angular_distance_degree(self.boat.position, self.target_point)
        embedded_distance = torch.tensor([angular_distance / self.starting_angular_distance])
        steps = torch.tensor([self.episode_steps / self.max_episode_steps])

        information = torch.concatenate([directions, speeds, embedded_distance, steps])

        # grid = greed_point_coordinate(boat_position=self.boat.position, greed_size=GRID_PIXELS_HALF_SIZE,
        #                               max_greed_distance_km=GRID_KM_HALF_SIZE)
        # self.last_wind_sub_map = self.wind_map.get_sub_map(grid, self.wind_map_time_index)
        self.last_wind_sub_map = self.wind_map.get_sub_map_without_grid(self.wind_map_time_index, self.boat.position[0],
                                                                        self.boat.position[1])

        self.observation = [information, self.last_wind_sub_map]

        if angular_distance / self.starting_angular_distance < 0.1 or angular_distance < 1:
            terminated = True
            reward = 50
        elif self.episode_steps > self.max_episode_steps:
            terminated = True
            reward = 50
        elif angular_distance > self.starting_angular_distance * 2:
            terminated = True
            reward = 50
        elif wind_speed_normalized == 0 and self.episode_steps > 10:
            terminated = True
            reward = 25
        else:
            terminated = False
            delta_distance = self.last_angular_distance - angular_distance
            reward = delta_distance / self.starting_angular_distance * 50  # - 10 / self.max_episode_steps

            self.last_delta_angular_distance = delta_distance

        self.episode_reward += reward

        self.last_angular_distance = angular_distance
        self.last_reward = reward

        return self.observation, reward, terminated, False, None

    def render(self):
        pass

    def render_frame(self):
        self.window.fill((0, 0, 0))

        self.window.blit(self.surf_heart_map, (0, 0))

        lat, lon = self.boat.position
        if lon > math.pi:
            lon -= math.pi
        elif lon < math.pi:
            lon += math.pi

        self.window.blit(self.boat_map_signal, (int(lon / (2 * math.pi) * self.window_size),
                                                int(lat / math.pi * self.window_size / 2)))

        lat, lon = self.start_point
        if lon > math.pi:
            lon -= math.pi
        elif lon < math.pi:
            lon += math.pi

        self.window.blit(self.start_point_map_signal, (int(lon / (2 * math.pi) * self.window_size),
                                                       int(lat / math.pi * self.window_size / 2)))

        lat, lon = self.target_point
        if lon > math.pi:
            lon -= math.pi
        elif lon < math.pi:
            lon += math.pi

        self.window.blit(self.target_point_map_signal, (int(lon / (2 * math.pi) * self.window_size),
                                                        int(lat / math.pi * self.window_size / 2)))

        """
        lat, lon = CHECK_POINT_0
        if lon > math.pi:
            lon -= math.pi
        elif lon < math.pi:
            lon += math.pi
        self.window.blit(self.target_point_map_signal, (int(lon / (2 * math.pi) * self.window_size),
                                                        int(lat / math.pi * self.window_size / 2)))

        lat, lon = CHECK_POINT_1
        if lon > math.pi:
            lon -= math.pi
        elif lon < math.pi:
            lon += math.pi
        self.window.blit(self.target_point_map_signal, (int(lon / (2 * math.pi) * self.window_size),
                                                        int(lat / math.pi * self.window_size / 2)))
        lat, lon = CHECK_POINT_2
        if lon > math.pi:
            lon -= math.pi
        elif lon < math.pi:
            lon += math.pi
        self.window.blit(self.target_point_map_signal, (int(lon / (2 * math.pi) * self.window_size),
                                                        int(lat / math.pi * self.window_size / 2)))

        lat, lon = CHECK_POINT_3
        if lon > math.pi:
            lon -= math.pi
        elif lon < math.pi:
            lon += math.pi
        self.window.blit(self.target_point_map_signal, (int(lon / (2 * math.pi) * self.window_size),
                                                        int(lat / math.pi * self.window_size / 2)))
        lat, lon = CHECK_POINT_4
        if lon > math.pi:
            lon -= math.pi
        elif lon < math.pi:
            lon += math.pi
        self.window.blit(self.target_point_map_signal, (int(lon / (2 * math.pi) * self.window_size),
                                                        int(lat / math.pi * self.window_size / 2)))
        lat, lon = CHECK_POINT_5
        if lon > math.pi:
            lon -= math.pi
        elif lon < math.pi:
            lon += math.pi
        self.window.blit(self.target_point_map_signal, (int(lon / (2 * math.pi) * self.window_size),
                                                        int(lat / math.pi * self.window_size / 2)))
        lat, lon = CHECK_POINT_6
        if lon > math.pi:
            lon -= math.pi
        elif lon < math.pi:
            lon += math.pi
        self.window.blit(self.target_point_map_signal, (int(lon / (2 * math.pi) * self.window_size),
                                                        int(lat / math.pi * self.window_size / 2)))
        lat, lon = CHECK_POINT_7
        if lon > math.pi:
            lon -= math.pi
        elif lon < math.pi:
            lon += math.pi
        self.window.blit(self.target_point_map_signal, (int(lon / (2 * math.pi) * self.window_size),
                                                        int(lat / math.pi * self.window_size / 2)))
        lat, lon = CHECK_POINT_8
        if lon > math.pi:
            lon -= math.pi
        elif lon < math.pi:
            lon += math.pi
        self.window.blit(self.target_point_map_signal, (int(lon / (2 * math.pi) * self.window_size),
                                                        int(lat / math.pi * self.window_size / 2)))
        lat, lon = CHECK_POINT_9
        if lon > math.pi:
            lon -= math.pi
        elif lon < math.pi:
            lon += math.pi
        self.window.blit(self.target_point_map_signal, (int(lon / (2 * math.pi) * self.window_size),
                                                        int(lat / math.pi * self.window_size / 2)))
        lat, lon = CHECK_POINT_10
        if lon > math.pi:
            lon -= math.pi
        elif lon < math.pi:
            lon += math.pi
        self.window.blit(self.target_point_map_signal, (int(lon / (2 * math.pi) * self.window_size),
                                                        int(lat / math.pi * self.window_size / 2)))

        lat, lon = CHECK_POINT_11
        if lon > math.pi:
            lon -= math.pi
        elif lon < math.pi:
            lon += math.pi
        self.window.blit(self.target_point_map_signal, (int(lon / (2 * math.pi) * self.window_size),
                                                        int(lat / math.pi * self.window_size / 2)))
        """

        if self.random_wind is True:
            wind = self.wind_map.interpolate([self.wind_map_time_index] + self.boat.position)
        else:
            self.wind_map_time_index = self.get_wind_map_time_index()

            if self.wind_map_time_index is not None:
                wind = self.wind_map.interpolate([self.wind_map_time_index] + self.boat.position)
            else:
                print("Finished wind data!!")
                wind = self.wind_map.interpolate([0] + self.boat.position)
        wind_angle = math.atan2(wind[1], wind[0]) + 3 / 2 * math.pi
        wind_speed = math.sqrt(wind[0] ** 2 + wind[1] ** 2)
        if wind_angle > 2 * math.pi:
            wind_angle = wind_angle - 2 * math.pi
        wind_angle = 2 * math.pi - wind_angle
        target_angle = from_start_end_to_direction(self.boat.position, self.target_point)

        image = pygame.transform.rotate(self.target_arrow_image, -target_angle / math.pi * 180)
        self.window.blit(image, (0.25 * self.window_size - image.get_width() // 2,
                                 0.75 * self.window_size - image.get_height() // 2))

        image = pygame.transform.rotate(self.wind_arrow_image, -wind_angle / math.pi * 180)
        self.window.blit(image, (0.25 * self.window_size - image.get_width() // 2,
                                 0.75 * self.window_size - image.get_height() // 2))

        image = pygame.transform.rotate(self.boat_arrow_image, -self.boat.orientation / math.pi * 180)
        self.window.blit(image, (0.25 * self.window_size - image.get_width() // 2,
                                 0.75 * self.window_size - image.get_height() // 2))

        text = self.font.render(f"boat speed: {self.boat.boat_speed:.3f} m/s", True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 1 / 10 * self.window_size / 2))
        self.window.blit(text, text_rect)

        text = self.font.render(f"wind speed: {wind_speed:.3f} m/s", True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 2 / 10 * self.window_size / 2))
        self.window.blit(text, text_rect)

        time_in_seconds = self.episode_steps * self.time_in_second_for_step
        if time_in_seconds < 60:
            time_to_print = f"{time_in_seconds:.3f} seconds"
        else:
            time_in_minutes = time_in_seconds / 60
            if time_in_minutes < 60:
                time_to_print = f"{time_in_minutes:.3f} minutes"
            else:
                time_in_hours = time_in_seconds / 3600
                if time_in_hours < 24:
                    time_to_print = f"{time_in_hours:.3f} hours"
                else:
                    time_in_days = time_in_hours / 24
                    time_to_print = f"{time_in_days:.3f} days"
        text = self.font.render(f"{time_to_print} elapsed [{self.episode_steps}]", True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 3 / 10 * self.window_size / 2))
        self.window.blit(text, text_rect)

        text = self.font.render(f"Distance: {self.last_angular_distance:.3f} °",
                                True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 4 / 10 * self.window_size / 2))
        self.window.blit(text, text_rect)

        text = self.font.render(f"Starting Distance: {self.starting_angular_distance:.3f} °",
                                True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 5 / 10 * self.window_size / 2))
        self.window.blit(text, text_rect)

        if type(self.last_delta_angular_distance) == str:
            text = self.font.render(f"Last d Distance: {self.last_delta_angular_distance} °",
                                    True, (0, 255, 0), (0, 0, 128))
        else:
            text = self.font.render(f"Last d Distance: {self.last_delta_angular_distance:.3f} °",
                                    True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 6 / 10 * self.window_size / 2))
        self.window.blit(text, text_rect)

        text = self.font.render(f"Cumulative Reward: {self.episode_reward + 55:.3f}",
                                True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 7 / 10 * self.window_size / 2))
        self.window.blit(text, text_rect)

        if type(self.last_reward) == str:
            text = self.font.render(f"Last Reward: {self.last_reward}",
                                    True, (0, 255, 0), (0, 0, 128))
        else:
            text = self.font.render(f"Last Reward: {self.last_reward:.3f}",
                                    True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 8 / 10 * self.window_size / 2))
        self.window.blit(text, text_rect)

        text = self.font.render(f"Last Episode Reward: {self.last_episode_reward:.3f}",
                                True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 9 / 10 * self.window_size / 2))
        self.window.blit(text, text_rect)

        size = self.last_wind_sub_map[0].shape[0]
        for i in range(size):
            for j in range(size):
                self.big_wind[int(i * BIG_WIND_MULTIPLIER):int((i + 1) * BIG_WIND_MULTIPLIER),
                              int(j * BIG_WIND_MULTIPLIER):int((j + 1) * BIG_WIND_MULTIPLIER)]\
                    = self.last_wind_sub_map[0][i, j]
        self.window.blit(pygame.surfarray.make_surface(np.rot90(np.flip(self.big_wind.numpy(), 1))), (0, 0))

        size = self.last_wind_sub_map[1].shape[0]
        for i in range(size):
            for j in range(size):
                self.big_wind[int(i * BIG_WIND_MULTIPLIER):int((i + 1) * BIG_WIND_MULTIPLIER),
                              int(j * BIG_WIND_MULTIPLIER):int((j + 1) * BIG_WIND_MULTIPLIER)] \
                    = self.last_wind_sub_map[1][i, j]
        self.window.blit(pygame.surfarray.make_surface(np.rot90(np.flip(self.big_wind.numpy(), 1))),
                         (0, int(size * BIG_WIND_MULTIPLIER)))

        pygame.display.update()

        """
        plt.imshow(self.last_wind_sub_map[0])
        plt.draw()
        plt.pause(0.001)
        """

        self.clock.tick(self.metadata["render_fps"])

    def reset_wind_map_time_index(self):
        y = self.time.year
        m = self.time.month
        d = self.time.day
        h = self.time.hour
        # print(f"Searched: [{h}; {d}; {m}; {y}]")
        for i in range(self.wind_map.wind_datatime.shape[0]):
            # print(self.wind_map.wind_datatime[i])
            cy = self.wind_map.wind_datatime[i, 3]
            cm = self.wind_map.wind_datatime[i, 2]
            cd = self.wind_map.wind_datatime[i, 1]
            ch = self.wind_map.wind_datatime[i, 0]
            if (h <= ch and d == cd and m == cm and y == cy) or (d < cd and m == cm and y == cy) or \
                    (m < cm and y == cy) or y < cy:
                """
                print(f"Find Out: [{self.wind_map.wind_datatime[i][0]}; "
                      f"{self.wind_map.wind_datatime[i][1]}; "
                      f"{self.wind_map.wind_datatime[i][2]}; "
                      f"{self.wind_map.wind_datatime[i][3]}]")
                """
                return i
        return None

    def get_wind_map_time_index(self):
        y = self.time.year
        m = self.time.month
        d = self.time.day
        h = self.time.hour
        i = self.wind_map_time_index
        if i is None:
            return None
        cy = self.wind_map.wind_datatime[i, 3]
        cm = self.wind_map.wind_datatime[i, 2]
        cd = self.wind_map.wind_datatime[i, 1]
        ch = self.wind_map.wind_datatime[i, 0]
        if (h <= ch and d == cd and m == cm and y == cy) or (d < cd and m == cm and y == cy) or (m < cm and y == cy) \
                or y < cy:
            """
            print("*" * 30)
            print(f"Searched: [{h}; {d}; {m}; {y}]")
            print(f"Find Out: [{self.wind_map.wind_datatime[i][0]}; "
                  f"{self.wind_map.wind_datatime[i][1]}; "
                  f"{self.wind_map.wind_datatime[i][2]}; "
                  f"{self.wind_map.wind_datatime[i][3]}]")
            print("@"*30)
            """
            return i
        i += 1
        cy = self.wind_map.wind_datatime[i, 3]
        cm = self.wind_map.wind_datatime[i, 2]
        cd = self.wind_map.wind_datatime[i, 1]
        ch = self.wind_map.wind_datatime[i, 0]
        if (h <= ch and d == cd and m == cm and y == cy) or (d < cd and m == cm and y == cy) or (m < cm and y == cy) \
                or y < cy:
            """
            print("*" * 30)
            print(f"Searched: [{h}; {d}; {m}; {y}]")
            print(f"Find Out: [{self.wind_map.wind_datatime[i][0]}; "
                  f"{self.wind_map.wind_datatime[i][1]}; "
                  f"{self.wind_map.wind_datatime[i][2]}; "
                  f"{self.wind_map.wind_datatime[i][3]}]")
            print("@" * 30)
            """
            return i
        if (h > 18 and d == 1 and m == 3 and y == 2021) or (d > 1 and m == 3 and y == 2021) or (m > 3 and y == 2021) or \
                (y > 2021):
            return None
        print("I should not be there but I will figure it out anyway!")
        print("*" * 30)
        print(f"Searched: [{h}; {d}; {m}; {y}]")
        print(f"Find Out: [{self.wind_map.wind_datatime[i - 1][0]}; "
              f"{self.wind_map.wind_datatime[i - 1][1]}; "
              f"{self.wind_map.wind_datatime[i - 1][2]}; "
              f"{self.wind_map.wind_datatime[i - 1][3]}]")
        print(f"Find Out: [{self.wind_map.wind_datatime[i][0]}; "
              f"{self.wind_map.wind_datatime[i][1]}; "
              f"{self.wind_map.wind_datatime[i][2]}; "
              f"{self.wind_map.wind_datatime[i][3]}]")
        print("@" * 30)
        return self.reset_wind_map_time_index()

    def close(self):
        pygame.display.quit()
        pygame.quit()


if __name__ == "__main__":
    env = RealisticEnvironment(start_point=random_location_in_atlantic_ocean,
                               target_point=random_location_in_atlantic_ocean,
                               render_mode="human", fps=1, time_in_second_for_step=10000)
    env.reset()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit()
        obs, rew, term, _, _ = env.step(random.randint(0, 2))
        if term:
            env.reset()
