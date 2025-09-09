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

from Sailing_Boats_Autopilot.utils import lat_converter, lon_converter
from Sailing_Boats_Autopilot.sailing_game import SailingBoat, SailingMap
from Sailing_Boats_Autopilot.geometry.grid_creation import greed_point_coordinate
from Sailing_Boats_Autopilot.constants import ABSOLUTE_PATH

BACKGROUND_COLOR = (255, 255, 255)
TARGET_COLOR = (255, 0, 0)
AGENT_COLOR = (0, 255, 0)
LINE_COLOR = (0, 0, 0)
ARROW_COLOR = (0, 0, 255)

GRID_PIXELS_HALF_SIZE = 5  # pxs
GRID_KM_HALF_SIZE = 400  # km

MAX_EPISODE_STEPS = 70

DAY = datetime.datetime(year=2023, month=1, day=1, hour=8, minute=30)


class RealisticEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "no_visual"], "render_fps": 4}

    def __init__(self, start_point=None, target_point=None, time_in_second_for_step=100, render_mode=None, fps=4):
        if start_point is None:
            self.start_point_function = None
            self.start_point = [lat_converter(64, 37, 25, "N"), lon_converter(32, 31, 10, "W")]
            # First Route : self.start_point = [lat_converter(64, 37, 25, "N"), lon_converter(32, 31, 10, "W")]
        elif callable(start_point):
            self.start_point_function = start_point
            self.start_point = self.start_point_function()
        else:
            self.start_point_function = None
            self.start_point = start_point
        if target_point is None:
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

        self.boat = SailingBoat(self.start_point, orientation=90 / 180 * math.pi)
        self.map = SailingMap(self.start_point, self.target_point, starting_time=None)

        self.observation = None
        self.episode_reward = None
        self.last_episode_reward = None

        if self.render_mode == "human":
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
            self.boat_map_signal = pygame.image.load(f"{ABSOLUTE_PATH}resources/images/boat_map_signal.png")
            self.start_point_map_signal = pygame.image.load(f"{ABSOLUTE_PATH}resources/images/"
                                                            f"start_point_map_signal.png")
            self.target_point_map_signal = pygame.image.load(f"{ABSOLUTE_PATH}resources/images/"
                                                             f"target_point_map_signal.png")

            self.font = pygame.font.Font('freesansbold.ttf', 32)

            self.window = pygame.display.set_mode((self.window_size, self.window_size))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        distance_angle = 0
        while (distance_angle < 5 or distance_angle > 9) and self.start_point_function is not None and\
                self.target_point_function is not None:
            # self.start_point = self.start_point_function()
            self.start_point = [lat_converter(29, 0, 0, "N"), lon_converter(45, 0, 0, "W")]
            self.target_point = self.target_point_function()
            distance_angle = math.acos(math.sin(self.start_point[0]) * math.sin(self.target_point[0]) +
                                       math.cos(self.start_point[0]) * math.cos(self.target_point[0]) *
                                       math.cos(self.start_point[1] - self.target_point[1]))
            distance_angle = distance_angle / math.pi * 180
        self.boat.position = self.start_point
        self.boat.orientation = random.randrange(0, 360) / 180 * math.pi
        self.episode_steps = 0
        if self.episode_reward is not None:
            self.last_episode_reward = self.episode_reward
        else:
            self.last_episode_reward = 0
        self.episode_reward = 0

        grid = greed_point_coordinate(self.boat.position, GRID_PIXELS_HALF_SIZE, GRID_KM_HALF_SIZE)
        wind_map = self.map.get_wind_normalized_tensor(grid)
        # distance_map = self.map.get_target_angular_distance_normalized(grid)
        self.observation = wind_map

        return self.observation

    def step(self, action):
        """

        :param action: [0=steering clockwise, 1=steering anticlockwise, 2=nothing]
        :return: None
        """
        self.episode_steps += 1
        reward = 0
        grid = greed_point_coordinate(self.boat.position, GRID_PIXELS_HALF_SIZE, GRID_KM_HALF_SIZE)
        wind_map = self.map.get_wind_normalized_tensor(grid)
        # distance_map = self.map.get_target_angular_distance_normalized(grid)
        self.observation = wind_map

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

        self.boat.integrate(self.map.WM.interpolate(self.boat.position), self.time_in_second_for_step)

        if self.render_mode == "human":
            self.render_frame()

        distance_angle = math.acos(math.sin(self.boat.position[0]) * math.sin(self.target_point[0]) +
                                   math.cos(self.boat.position[0]) * math.cos(self.target_point[0]) *
                                   math.cos(self.boat.position[1] - self.target_point[1]))
        self.boat.angular_distance = distance_angle

        if self.boat.boat_speed < 0.1:
            reward -= 20

        if distance_angle / math.pi * 180 < 1:
            terminated = True
            reward += 1000
        elif self.episode_steps > MAX_EPISODE_STEPS:
            terminated = True
            reward -= distance_angle * 100
        elif distance_angle / math.pi * 180 > 12:
            terminated = True
            reward -= max(distance_angle * 100, 500)
        else:
            terminated = False
            reward -= 0.5

        self.episode_reward += reward

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

        wind = self.map.WM.interpolate(self.boat.position)
        wind_angle = math.atan2(wind[1], wind[0]) + 3 / 2 * math.pi
        if wind_angle > 2 * math.pi:
            wind_angle = wind_angle - 2 * math.pi
        wind_angle = 2 * math.pi - wind_angle
        image = pygame.transform.rotate(self.wind_arrow_image, -wind_angle / math.pi * 180)
        self.window.blit(image, (0.25 * self.window_size - image.get_width() // 2,
                                 0.75 * self.window_size - image.get_height() // 2))

        image = pygame.transform.rotate(self.boat_arrow_image, -self.boat.orientation / math.pi * 180)
        self.window.blit(image, (0.25 * self.window_size - image.get_width() // 2,
                                 0.75 * self.window_size - image.get_height() // 2))

        text = self.font.render(f"boat speed: {self.boat.boat_speed:.3f} m/s", True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 1 / 6 * self.window_size / 2))
        self.window.blit(text, text_rect)

        wind = self.map.WM.interpolate(self.boat.position)
        wind_speed = math.sqrt(wind[0] ** 2 + wind[1] ** 2)
        text = self.font.render(f"wind speed: {wind_speed:.3f} m/s", True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 2 / 6 * self.window_size / 2))
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
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 3 / 6 * self.window_size / 2))
        self.window.blit(text, text_rect)

        angular_distance = self.boat.angular_distance
        text = self.font.render(f"Angular Distance: {angular_distance / math.pi * 180:.3f} deg",
                                True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 4 / 6 * self.window_size / 2))
        self.window.blit(text, text_rect)

        text = self.font.render(f"Last Episode Reward: {self.last_episode_reward:.3f}",
                                True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 5 / 6 * self.window_size / 2))
        self.window.blit(text, text_rect)

        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        pygame.display.quit()
        pygame.quit()


if __name__ == "__main__":
    env = RealisticEnvironment(render_mode="human", fps=30000, time_in_second_for_step=10000)
    env.reset()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit()
        obs, rew, term, _, _ = env.step(random.randint(0, 2))
        if term:
            env.reset()
