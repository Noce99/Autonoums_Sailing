import math

import numpy as np
import pygame
from screeninfo import get_monitors
from math import pi
import cv2

from Sailing_Boats_Autopilot.geometry.grid_creation import greed_point_coordinate
from Sailing_Boats_Autopilot.constants import ABSOLUTE_PATH


class Visualizer:
    def __init__(self, sailing_map, boat):
        pygame.init()
        monitor_height = get_monitors()[0].height
        self.window_size = monitor_height - int(0.2 * monitor_height)
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("World Visualizer")

        self.sailing_map = sailing_map

        self.boat_image = pygame.image.load(f"{ABSOLUTE_PATH}resources/images/boat_from_above.png") \
            .convert_alpha()
        self.boat_image = pygame.transform.scale(self.boat_image,
                                                 (int(0.02 * self.window_size), int(0.02 * self.window_size) * 4))

        self.wind_arrow_image = pygame.image.load(f"{ABSOLUTE_PATH}resources/images/wind_arrow.png") \
            .convert_alpha()
        self.wind_arrow_image = pygame.transform.scale(self.wind_arrow_image,
                                                       (int(0.25 * self.window_size), int(0.25 * self.window_size)))

        self.font = pygame.font.Font('freesansbold.ttf', 32)
        self.text_speed = self.font.render('0 m/s 0 days', True, (0, 255, 0), (0, 0, 128))
        self.text_speed_rect = self.text_speed.get_rect()
        self.text_speed_rect.center = (int(0.2*self.window_size), int(0.95*self.window_size))

        self.boat = boat
        self.sailing_map.boat_position = self.boat.position
        self.sailing_map.boat_orientation = self.boat.orientation

        # self.boat_image = cv2.imread("resources/sailing_boat/boat_from_above.png")
        # self.boat_image = cv2.resize(self.boat_image, (int(0.05*self.window_size), int(0.05*self.window_size)),
        #                             interpolation=cv2.INTER_AREA)
        self.center = self.sailing_map.start_point
        self.km = 100  # 5000
        self.time_in_seconds = 0
        self.time_steps = 1000

    def display(self):
        grid = greed_point_coordinate(self.center, 100, self.km)

        wind_map = self.sailing_map.get_speed_rgb(grid)
        wind_map = np.rot90(np.flip(wind_map, 1))
        resized_wind_map = cv2.resize(wind_map, (self.window_size, self.window_size), interpolation=cv2.INTER_AREA)

        surf_wind_map = pygame.Surface(resized_wind_map.shape[0:2], pygame.SRCALPHA, 32)
        pygame.pixelcopy.array_to_surface(surf_wind_map, resized_wind_map[:, :, 0:3])
        surface_alpha = np.array(surf_wind_map.get_view('A'), copy=False)
        surface_alpha[:, :] = resized_wind_map[:, :, 3]

        heart_map = self.sailing_map.get_earth_rgb(grid)
        heart_map = np.rot90(np.flip(heart_map, 1))
        resized_heart_map = cv2.resize(heart_map, (self.window_size, self.window_size), interpolation=cv2.INTER_AREA)
        surf_heart_map = pygame.surfarray.make_surface(resized_heart_map)

        self.screen.blit(surf_heart_map, (0, 0))
        del surface_alpha
        self.screen.blit(surf_wind_map, (0, 0))

        position = self.sailing_map.get_boat_image_position(grid)
        if position is not None:
            boat_relative_raw, boat_relative_column = position
            image = pygame.transform.rotate(self.boat_image, -self.sailing_map.boat_orientation / math.pi * 180)
            self.screen.blit(image, (boat_relative_column * self.window_size - image.get_width() // 2,
                                     boat_relative_raw * self.window_size - image.get_height() // 2))
            print(f"Boat Position: {self.boat.position}")
            wind = self.sailing_map.WM.interpolate(self.boat.position)
            wind_angle = math.atan2(wind[1], wind[0]) + 3 / 2 * math.pi
            if wind_angle > 2 * math.pi:
                wind_angle = wind_angle - 2 * math.pi
            wind_angle = 2 * math.pi - wind_angle
            image = pygame.transform.rotate(self.wind_arrow_image, -wind_angle / math.pi * 180)
            self.screen.blit(image, (0.87 * self.window_size - image.get_width() // 2,
                                     0.87 * self.window_size - image.get_height() // 2))

        self.screen.blit(self.text_speed, self.text_speed_rect)
        pygame.display.update()

    def ui(self):
        self.display()
        clock = pygame.time.Clock()
        while True:
            event_list = pygame.event.get()
            for ev in event_list:
                if ev.type == pygame.QUIT:
                    exit()
                elif ev.type == pygame.KEYUP:
                    if ev.key == 43:  # +
                        if self.km > 200:
                            self.km -= 100
                            self.display()
                        print(f"Center: {self.center}, Km: {self.km}")
                    elif ev.key == 45:  # -
                        if self.km < 6314:
                            self.km += 100
                            self.display()
                        print(f"Center: {self.center}, Km: {self.km}")
                    elif ev.key == 119:  # W
                        if self.center[0] > 0.01 * self.km / 200:
                            self.center[0] -= 0.01 * self.km / 200
                        print(f"Center: {self.center}, Km: {self.km}")
                        self.display()
                    elif ev.key == 100:  # D
                        self.center[1] += 0.01 * self.km / 200
                        if self.center[1] > 2 * pi:
                            self.center[1] -= 2 * pi
                        print(f"Center: {self.center}, Km: {self.km}")
                        self.display()
                    elif ev.key == 115:  # S
                        if self.center[0] < pi - 0.01 * self.km / 200:
                            self.center[0] += 0.01 * self.km / 200
                        print(f"Center: {self.center}, Km: {self.km}")
                        self.display()
                    elif ev.key == 97:  # A
                        self.center[1] -= 0.01 * self.km / 200
                        if self.center[1] < 0:
                            self.center[1] += 2 * pi
                        print(f"Center: {self.center}, Km: {self.km}")
                        self.display()
                    elif ev.key == 32:  # Space Bar
                        pass
                        """
                        self.boat.integrate(self.sailing_map.WM.interpolate(self.boat.position), 1000)
                        self.sailing_map.boat_position = self.boat.position
                        self.sailing_map.boat_orientation = self.boat.orientation
                        self.display()
                        """
                    elif ev.key == 122:  # Z
                        self.boat.turn_clockwise()
                        self.sailing_map.boat_orientation = self.boat.orientation
                        self.display()
                    elif ev.key == 120:  # X
                        self.boat.turn_anticlockwise()
                        self.sailing_map.boat_orientation = self.boat.orientation
                        self.display()
                    else:
                        print(ev.key)

            clock.tick(1)
            self.boat.integrate(self.sailing_map.WM.interpolate(self.boat.position), self.time_steps)
            self.time_in_seconds += self.time_steps
            self.sailing_map.boat_position = self.boat.position
            self.sailing_map.boat_orientation = self.boat.orientation
            self.update_speed()
            self.display()

    def update_speed(self):
        self.text_speed = self.font.render(f"{self.boat.boat_speed:.2f} m/s {self.time_in_seconds/3600:.2f} days",
                                           True, (0, 255, 0), (0, 0, 128))


def blit_alpha(target, source, location, opacity):
    x = location[0]
    y = location[1]
    temp = pygame.Surface((source.get_width(), source.get_height())).convert()
    temp.blit(target, (-x, -y))
    temp.blit(source, (0, 0))
    temp.set_alpha(opacity)
    target.blit(temp, location)
