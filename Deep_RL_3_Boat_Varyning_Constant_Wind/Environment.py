import math
import sys
import time

import pygame
import gymnasium as gym
import torch
from gymnasium import spaces
import random

BACKGROUND_COLOR = (255, 255, 255)
TARGET_COLOR = (255, 0, 0)
AGENT_COLOR = (0, 255, 0)
LINE_COLOR = (0, 0, 0)
ARROW_COLOR = (0, 0, 255)


class MyFirstSimpleEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "no_visual"], "render_fps": 4}

    directions = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5, 11: 6, 12: 6, 13: 7, 14: 7, 15: 0}
    inv_directions = {0: 0, 1: 45, 2: 90, 3: 135, 4: 180, 5: 225, 6: 270, 7: 315}

    def __init__(self, render_mode=None, size=5, fps=4, wind_from_theta=0):
        """
        :param wind_from_theta: theta counter clock wise from x-axis (in degrees)
        """
        self.size = size
        self.render_mode = render_mode
        self.metadata["render_fps"] = fps
        self.wind_from_theta = wind_from_theta

        self.window_size = 500
        self.window = None
        self.clock = None
        self.agent_location = None
        self.target_location = None
        self.distances_from_target = None
        self.max_distance = math.sqrt(2) * size
        self.episode_steps = None

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int)
            })

        self.action_space = spaces.Discrete(7)
        possible_directions = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
        self.action_to_direction = {ii: possible_directions[ii] for ii in range(8)}
        # self.wind_map = torch.rand((self.size, self.size))
        self.wind_map = torch.ones((self.size, self.size), dtype=torch.float32) * (self.wind_from_theta / 360)

        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

    def get_observation_as_table(self):
        a_l_y, a_l_x = self.agent_location

        grid_x, grid_y = torch.meshgrid(torch.arange(0, self.size), torch.arange(0, self.size), indexing='ij')
        a_x = torch.ones_like(grid_x) * a_l_x
        a_y = torch.ones_like(grid_x) * a_l_y
        agent_position = torch.sqrt(torch.square(torch.subtract(a_x, grid_x)) + torch.square(
            torch.subtract(a_y, grid_y))) / self.max_distance
        return torch.cat([agent_position[None, :], self.distances_from_target[None, :], self.wind_map[None, :]])

    def load_state_from_tensor(self, tensor):
        min_flatten = torch.argmin(tensor[0])
        min_raw = min_flatten // tensor[0].shape[0]
        min_column = min_flatten % tensor[0].shape[1]
        self.agent_location = (int(min_column), int(min_raw))
        self.distances_from_target = tensor[1]
        self.wind_map = tensor[2]
        min_flatten = torch.argmin(tensor[1])
        min_raw = min_flatten // tensor[1].shape[0]
        min_column = min_flatten % tensor[1].shape[1]
        self.target_location = (int(min_column), int(min_raw))

    def get_agent_target_distance(self):
        a_l_x, a_l_y = self.agent_location
        t_l_x, t_l_y = self.target_location
        diff_x = abs(a_l_x - t_l_x)
        diff_y = abs(a_l_y - t_l_y)
        return diff_x + diff_y

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_location = (random.randrange(self.size), random.randrange(self.size))
        self.target_location = self.agent_location
        self.episode_steps = 0
        while self.target_location == self.agent_location:
            self.target_location = (random.randrange(self.size), random.randrange(self.size))

        grid_x, grid_y = torch.meshgrid(torch.arange(0, self.size), torch.arange(0, self.size), indexing='ij')
        t_l_y, t_l_x = self.target_location
        a_x = torch.ones_like(grid_x) * t_l_x
        a_y = torch.ones_like(grid_x) * t_l_y
        self.distances_from_target = torch.sqrt(torch.square(torch.subtract(a_x, grid_x)) + torch.square(
            torch.subtract(a_y, grid_y))) / self.max_distance

        self.wind_map = torch.ones((self.size, self.size), dtype=torch.float32) * random.random()

        observation = self.get_observation_as_table()
        distance = self.get_observation_as_table()

        if self.render_mode == "human":
            self.render_frame()

        return observation, distance

    def clip_locations(self, old_x, old_y, new_x, new_y):
        if new_x < 0 or new_y < 0 or new_x >= self.size or new_y >= self.size:
            return old_x, old_y
        else:
            return new_x, new_y

    def step(self, action):
        self.episode_steps += 1
        reward = 0
        start_a_l_x, start_a_l_y = self.agent_location
        wind_direction = self.from_norm_theta_to_direction(self.wind_map[start_a_l_x, start_a_l_y])
        if action != wind_direction:
            d_x, d_y = self.action_to_direction[action]
        else:
            d_x, d_y = (0, 0)
        a_l_x, a_l_y = self.clip_locations(start_a_l_x, start_a_l_y, start_a_l_x + d_x, start_a_l_y + d_y)
        if start_a_l_x == a_l_x and start_a_l_y == a_l_y:
            reward -= 1000
        self.agent_location = (a_l_x, a_l_y)

        terminated = self.agent_location == self.target_location
        if terminated:
            reward += 1000

        reward -= self.distances_from_target[a_l_y, a_l_x]

        observation = self.get_observation_as_table()
        distance = self.get_agent_target_distance()

        if self.render_mode == "human":
            self.render_frame()

        return observation, reward, terminated, False, distance

    def render(self):
        pass

    def from_norm_theta_to_direction(self, normalize_theta):
        return self.directions[int((normalize_theta * 360) // 22.5)]

    @staticmethod
    def draw_arrow(background, size_of_a_square_in_pixels, color, dx, dy, direction):
        half_arrow_size = 0.3*size_of_a_square_in_pixels
        small_half_arrow_size = 0.2*size_of_a_square_in_pixels
        dtheta = 0.15
        pygame.draw.line(background, color,
                         (dx+(size_of_a_square_in_pixels // 2 + int(math.cos(direction)*half_arrow_size)),
                          dy+(size_of_a_square_in_pixels // 2 + int(-math.sin(direction)*half_arrow_size))),
                         (dx + (size_of_a_square_in_pixels // 2 - int(math.cos(direction) * half_arrow_size)),
                          dy + (size_of_a_square_in_pixels // 2 - int(-math.sin(direction) * half_arrow_size))))
        pygame.draw.polygon(background, color, [
            (dx + (size_of_a_square_in_pixels // 2 - int(math.cos(direction + dtheta) * small_half_arrow_size)),
             dy + (size_of_a_square_in_pixels // 2 - int(-math.sin(direction + dtheta) * small_half_arrow_size))),
            (dx + (size_of_a_square_in_pixels // 2 - int(math.cos(direction) * half_arrow_size)),
             dy + (size_of_a_square_in_pixels // 2 - int(-math.sin(direction) * half_arrow_size))),
            (dx + (size_of_a_square_in_pixels // 2 - int(math.cos(direction - dtheta) * small_half_arrow_size)),
             dy + (size_of_a_square_in_pixels // 2 - int(-math.sin(direction - dtheta) * small_half_arrow_size))),
        ])

    def render_frame(self):
        self.clock = pygame.time.Clock()

        background = pygame.Surface((self.window_size, self.window_size))
        background.fill(BACKGROUND_COLOR)
        size_of_a_square_in_pixels = self.window_size // self.size

        t_l_x, t_l_y = self.target_location
        pygame.draw.rect(background, TARGET_COLOR,
                         pygame.Rect(t_l_x * size_of_a_square_in_pixels,
                                     t_l_y * size_of_a_square_in_pixels,
                                     size_of_a_square_in_pixels,
                                     size_of_a_square_in_pixels))
        a_l_x, a_l_y = self.agent_location
        pygame.draw.circle(background, AGENT_COLOR, ((a_l_x + .5) * size_of_a_square_in_pixels,
                                                     (a_l_y + .5) * size_of_a_square_in_pixels),
                           size_of_a_square_in_pixels // 2.2)

        for i in range(self.size):
            pygame.draw.line(background, LINE_COLOR, (i * size_of_a_square_in_pixels, 0),
                             (i * size_of_a_square_in_pixels, self.window_size), width=3)
            pygame.draw.line(background, LINE_COLOR, (0, i * size_of_a_square_in_pixels),
                             (self.window_size, i * size_of_a_square_in_pixels), width=3)
            for j in range(self.size):
                self.draw_arrow(background, size_of_a_square_in_pixels, ARROW_COLOR,
                                i*size_of_a_square_in_pixels, j*size_of_a_square_in_pixels,
                                self.wind_map[i, j]*2*math.pi)
                self.draw_arrow(background, size_of_a_square_in_pixels, TARGET_COLOR,
                                i * size_of_a_square_in_pixels, j * size_of_a_square_in_pixels,
                                self.inv_directions[
                                    self.from_norm_theta_to_direction(self.wind_map[i, j])
                                ]*2*math.pi/360)

        self.window.blit(background, background.get_rect())
        # pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        pygame.display.quit()
        pygame.quit()


if __name__ == "__main__":
    env = MyFirstSimpleEnvironment("human", fps=30000, wind_from_theta=220)
    env.reset()
    i = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit()
        env.reset()
        env.agent_location = (2, 2)
        env.render_frame()
        time.sleep(2)
        env.step(i)
        time.sleep(1)
        i += 1
        if i == 8:
            i = 0
        print(env.get_observation_as_table())
