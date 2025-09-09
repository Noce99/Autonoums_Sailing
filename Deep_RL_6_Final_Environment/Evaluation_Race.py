import math
import sys
from datetime import datetime, timedelta

import cv2
import numpy as np
import pygame
import torch
from screeninfo import get_monitors

from Sailing_Boats_Autopilot.constants import ABSOLUTE_PATH
from Sailing_Boats_Autopilot.sailing_game import WindHistoricMap, SailingBoat
from Sailing_Boats_Autopilot.utils import from_start_end_to_direction, lat_converter, lon_converter, \
    random_start_end_race
from Sailing_Boats_Autopilot.Deep_RL_6_Final_Environment.Environment import RealisticEnvironment, \
    GRID_PIXELS_SIZE, BIG_WIND_MULTIPLIER


class CompetitionBoat:
    def __init__(self, name, start_date_time: datetime):
        self.name = name

        self.date_time = torch.load(f"{ABSOLUTE_PATH}/resources/GPS/{self.name}_date_time.tensor")
        self.location = torch.load(f"{ABSOLUTE_PATH}/resources/GPS/{self.name}_location.tensor")

        self.time_index = 0

        start_year = start_date_time.year
        start_month = start_date_time.month
        start_day = start_date_time.day
        start_hour = start_date_time.hour

        while True:
            year = self.date_time[self.time_index][0]
            month = self.date_time[self.time_index][1]
            day = self.date_time[self.time_index][2]
            hour = self.date_time[self.time_index][3]
            if year == start_year and month == start_month and day == start_day and hour == start_hour:
                break
            self.time_index += 1

    def get_position(self, wanted_date_time: datetime):
        wanted_year = wanted_date_time.year
        wanted_month = wanted_date_time.month
        wanted_day = wanted_date_time.day
        wanted_hour = wanted_date_time.hour
        while True:
            try:
                next_year = self.date_time[self.time_index + 1][0]
                next_month = self.date_time[self.time_index + 1][1]
                next_day = self.date_time[self.time_index + 1][2]
                next_hour = self.date_time[self.time_index + 1][3]
            except IndexError:
                return 0., 0.
            if wanted_year <= next_year and wanted_month <= next_month and \
                    wanted_day <= next_day and wanted_hour <= next_hour:
                break
            self.time_index += 1
        location = self.location[self.time_index]
        return float(location[0]), float(location[1])


class EvaluationRace:
    def __init__(self, render_mode="human", time_in_second_for_step=100, fps=30, max_episode_steps=None):
        self.env = RealisticEnvironment(start_point_target_point_function=random_start_end_race,
                                        render_mode=render_mode, time_in_second_for_step=time_in_second_for_step,
                                        fps=fps, visualizer_function=self.draw_frame,
                                        max_episode_steps=max_episode_steps,
                                        random_wind=False)

        if render_mode == "human":
            self.MCOQ = CompetitionBoat("MCOQ", self.env.time)
            self.PRYS = CompetitionBoat("PRYS", self.env.time)
            self.STAR = CompetitionBoat("STAR", self.env.time)
            self.MCOQ_old_positions = []
            self.PRYS_old_positions = []
            self.STAR_old_positions = []
            self.AGENT_old_positions = []

            pygame.init()
            self.window_size = 1000
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.FPS = fps
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Evaluation Race")

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
            self.MCOQ_map_signal = pygame.image.load(f"{ABSOLUTE_PATH}resources/images/MCOQ_map_signal.png")
            self.PRYS_map_signal = pygame.image.load(f"{ABSOLUTE_PATH}resources/images/PRYS_map_signal.png")
            self.STAR_map_signal = pygame.image.load(f"{ABSOLUTE_PATH}resources/images/STAR_map_signal.png")

            self.font = pygame.font.Font('freesansbold.ttf', 32)

            self.big_wind = torch.zeros(GRID_PIXELS_SIZE * BIG_WIND_MULTIPLIER,
                                        GRID_PIXELS_SIZE * BIG_WIND_MULTIPLIER)

    def draw_frame(self):

        if self.env.episode_steps == 1:
            pass
            """
            self.MCOQ = CompetitionBoat("MCOQ", self.env.time)
            self.PRYS = CompetitionBoat("PRYS", self.env.time)
            self.STAR = CompetitionBoat("STAR", self.env.time)
            self.MCOQ_old_positions = []
            self.PRYS_old_positions = []
            self.STAR_old_positions = []
            self.AGENT_old_positions = []
            """

        self.window.fill((0, 0, 0))

        self.window.blit(self.surf_heart_map, (0, 0))

        lat, lon = self.env.boat.position
        if lon > math.pi:
            lon -= math.pi
        elif lon < math.pi:
            lon += math.pi

        for l_1, l_2 in self.AGENT_old_positions:
            pygame.draw.circle(self.window, (234, 0, 0), (int(l_2 / (2 * math.pi) * self.window_size),
                                                          int(l_1 / math.pi * self.window_size / 2)), 1)
        self.AGENT_old_positions.append((lat, lon))
        pygame.draw.circle(self.window, (234, 0, 0), (int(lon / (2 * math.pi) * self.window_size),
                                                      int(lat / math.pi * self.window_size / 2)), 5)

        lat, lon = self.MCOQ.get_position(self.env.time)
        if lon > math.pi:
            lon -= math.pi
        elif lon < math.pi:
            lon += math.pi

        for l_1, l_2 in self.MCOQ_old_positions:
            pygame.draw.circle(self.window, (255, 195, 43), (int(l_2 / (2 * math.pi) * self.window_size),
                                                             int(l_1 / math.pi * self.window_size / 2)), 1)
        self.MCOQ_old_positions.append((lat, lon))
        pygame.draw.circle(self.window, (255, 195, 43), (int(lon / (2 * math.pi) * self.window_size),
                                                         int(lat / math.pi * self.window_size / 2)), 5)

        lat, lon = self.PRYS.get_position(self.env.time)
        if lon > math.pi:
            lon -= math.pi
        elif lon < math.pi:
            lon += math.pi

        for l_1, l_2 in self.PRYS_old_positions:
            pygame.draw.circle(self.window, (0, 188, 253), (int(l_2 / (2 * math.pi) * self.window_size),
                                                            int(l_1 / math.pi * self.window_size / 2)), 1)
        self.PRYS_old_positions.append((lat, lon))
        pygame.draw.circle(self.window, (0, 188, 253), (int(lon / (2 * math.pi) * self.window_size),
                                                        int(lat / math.pi * self.window_size / 2)), 5)

        lat, lon = self.STAR.get_position(self.env.time)
        if lon > math.pi:
            lon -= math.pi
        elif lon < math.pi:
            lon += math.pi

        for l_1, l_2 in self.STAR_old_positions:
            pygame.draw.circle(self.window, (255, 125, 16), (int(l_2 / (2 * math.pi) * self.window_size),
                                                             int(l_1 / math.pi * self.window_size / 2)), 1)
        self.STAR_old_positions.append((lat, lon))
        pygame.draw.circle(self.window, (255, 125, 16), (int(lon / (2 * math.pi) * self.window_size),
                                                         int(lat / math.pi * self.window_size / 2)), 5)

        lat, lon = self.env.start_point
        if lon > math.pi:
            lon -= math.pi
        elif lon < math.pi:
            lon += math.pi

        pygame.draw.circle(self.window, (0, 205, 56), (int(lon / (2 * math.pi) * self.window_size),
                                                       int(lat / math.pi * self.window_size / 2)), 5)

        lat, lon = self.env.target_point
        if lon > math.pi:
            lon -= math.pi
        elif lon < math.pi:
            lon += math.pi

        pygame.draw.circle(self.window, (237, 50, 174), (int(lon / (2 * math.pi) * self.window_size),
                                                         int(lat / math.pi * self.window_size / 2)), 5)

        self.env.wind_map_time_index = self.env.get_wind_map_time_index()

        if self.env.wind_map_time_index is not None:
            wind = self.env.wind_map.interpolate([self.env.wind_map_time_index] + self.env.boat.position)
        else:
            print("Finished wind data!!")
            wind = self.env.wind_map.interpolate([0] + self.env.boat.position)
        wind_angle = math.atan2(wind[1], wind[0]) + 3 / 2 * math.pi
        wind_speed = math.sqrt(wind[0] ** 2 + wind[1] ** 2)
        if wind_angle > 2 * math.pi:
            wind_angle = wind_angle - 2 * math.pi
        wind_angle = 2 * math.pi - wind_angle
        target_angle = from_start_end_to_direction(self.env.boat.position, self.env.target_point)

        image = pygame.transform.rotate(self.target_arrow_image, -target_angle / math.pi * 180)
        self.window.blit(image, (0.25 * self.window_size - image.get_width() // 2,
                                 0.75 * self.window_size - image.get_height() // 2))

        image = pygame.transform.rotate(self.wind_arrow_image, -wind_angle / math.pi * 180)
        self.window.blit(image, (0.25 * self.window_size - image.get_width() // 2,
                                 0.75 * self.window_size - image.get_height() // 2))

        image = pygame.transform.rotate(self.boat_arrow_image, -self.env.boat.orientation / math.pi * 180)
        self.window.blit(image, (0.25 * self.window_size - image.get_width() // 2,
                                 0.75 * self.window_size - image.get_height() // 2))

        text = self.font.render(f"boat speed: {self.env.boat.boat_speed:.3f} m/s", True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 1 / 10 * self.window_size / 2))
        self.window.blit(text, text_rect)

        text = self.font.render(f"wind speed: {wind_speed:.3f} m/s", True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 2 / 10 * self.window_size / 2))
        self.window.blit(text, text_rect)

        time_in_seconds = self.env.episode_steps * self.env.time_in_second_for_step
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
        text = self.font.render(f"{time_to_print} elapsed [{self.env.episode_steps}]", True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 3 / 10 * self.window_size / 2))
        self.window.blit(text, text_rect)

        text = self.font.render(f"Distance: {self.env.last_angular_distance:.3f} °",
                                True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 4 / 10 * self.window_size / 2))
        self.window.blit(text, text_rect)

        text = self.font.render(f"Starting Distance: {self.env.starting_angular_distance:.3f} °",
                                True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 5 / 10 * self.window_size / 2))
        self.window.blit(text, text_rect)

        if type(self.env.last_delta_angular_distance) == str:
            text = self.font.render(f"Last d Distance: {self.env.last_delta_angular_distance} °",
                                    True, (0, 255, 0), (0, 0, 128))
        else:
            text = self.font.render(f"Last d Distance: {self.env.last_delta_angular_distance:.3f} °",
                                    True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 6 / 10 * self.window_size / 2))
        self.window.blit(text, text_rect)

        text = self.font.render(f"Cumulative Reward: {self.env.episode_reward + 55:.3f}",
                                True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 7 / 10 * self.window_size / 2))
        self.window.blit(text, text_rect)

        if type(self.env.last_reward) == str:
            text = self.font.render(f"Last Reward: {self.env.last_reward}",
                                    True, (0, 255, 0), (0, 0, 128))
        else:
            text = self.font.render(f"Last Reward: {self.env.last_reward:.3f}",
                                    True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 8 / 10 * self.window_size / 2))
        self.window.blit(text, text_rect)

        text = self.font.render(f"Last Episode Reward: {self.env.last_episode_reward:.3f}",
                                True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 9 / 10 * self.window_size / 2))
        self.window.blit(text, text_rect)

        size = self.env.last_wind_sub_map[0].shape[0]
        for i in range(size):
            for j in range(size):
                self.big_wind[int(i * BIG_WIND_MULTIPLIER):int((i + 1) * BIG_WIND_MULTIPLIER),
                int(j * BIG_WIND_MULTIPLIER):int((j + 1) * BIG_WIND_MULTIPLIER)] \
                    = self.env.last_wind_sub_map[0][i, j]
        self.window.blit(pygame.surfarray.make_surface(np.rot90(np.flip(self.big_wind.numpy(), 1))), (0, 0))

        size = self.env.last_wind_sub_map[1].shape[0]
        for i in range(size):
            for j in range(size):
                self.big_wind[int(i * BIG_WIND_MULTIPLIER):int((i + 1) * BIG_WIND_MULTIPLIER),
                int(j * BIG_WIND_MULTIPLIER):int((j + 1) * BIG_WIND_MULTIPLIER)] \
                    = self.env.last_wind_sub_map[1][i, j]
        self.window.blit(pygame.surfarray.make_surface(np.rot90(np.flip(self.big_wind.numpy(), 1))),
                         (0, int(size * BIG_WIND_MULTIPLIER)))

        """
        text = self.font.render(f"Last Episode Reward: {self.last_episode_reward:.3f}",
                                True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 5 / 9 * self.window_size / 2))
        self.window.blit(text, text_rect)

        text = self.font.render(f"{self.observation[0][:8].int().tolist()}",
                                True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 6 / 9 * self.window_size / 2))
        self.window.blit(text, text_rect)

        text = self.font.render(f"{self.observation[0][8:16].int().tolist()}",
                                True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 7 / 9 * self.window_size / 2))
        self.window.blit(text, text_rect)

        text = self.font.render(f"{self.observation[0][16:24].int().tolist()}",
                                True, (0, 255, 0), (0, 0, 128))
        text_rect = text.get_rect()
        text_rect.center = (int(self.window_size - (0.5 * self.window_size - text_rect.width) // 2) -
                            (0.5 * text_rect.width), int(0.5 * self.window_size + 8 / 9 * self.window_size / 2))
        self.window.blit(text, text_rect)
        """
        print(self.env.time)
        pygame.display.update()
        self.clock.tick(self.FPS)


def human_player():
    race = EvaluationRace(render_mode="human", time_in_second_for_step=3000, fps=4, max_episode_steps=1000)

    race.env.reset()

    need_to_start_new_episode = False
    action_to_be_processed = None
    while True:
        if need_to_start_new_episode:
            race.env.reset()
            need_to_start_new_episode = False

        if action_to_be_processed is None:
            action = 2
        else:
            action = action_to_be_processed
            action_to_be_processed = None

        _, _, terminated, _, _ = race.env.step(action)
        if terminated:
            need_to_start_new_episode = True
        event_list = pygame.event.get()
        for ev in event_list:
            if ev.type == pygame.QUIT:
                exit()
            elif ev.type == pygame.KEYUP:
                if ev.key == 100:  # D
                    action_to_be_processed = 0
                elif ev.key == 97:  # A
                    action_to_be_processed = 1
                elif ev.key == 32:  # SPACE BAR
                    need_to_start_new_episode = True
                else:
                    print(ev.key)


def bot(weight_path):
    raise NotImplementedError

    race = EvaluationRace(time_in_second_for_step=3000)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        race.draw_frame()
        race.integrate()


if __name__ == "__main__":
    human_player()
