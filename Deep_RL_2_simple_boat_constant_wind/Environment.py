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


class MyFirstSimpleEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "no_visual"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, fps=4, wind_from=0):
        """
        :param wind_from: 0 est, 1 north_est, 2 north, 3 north_west, 4 west, 5 south_west, 6 south, 7 south_est
        """
        self.size = size
        self.render_mode = render_mode
        self.metadata["render_fps"] = fps
        self.wind_from = wind_from

        self.window_size = 500
        self.window = None
        self.clock = None
        self.agent_location = None
        self.target_location = None
        self.distances_from_target = None

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int)
            })

        self.action_space = spaces.Discrete(7)
        possible_directions = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
        self.action_to_direction = {i: possible_directions[i] for i in range(8)}
        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

    def get_observation_as_table(self):
        table = torch.zeros((self.size, self.size), dtype=torch.float32)
        a_l_y, a_l_x = self.agent_location
        t_l_y, t_l_x = self.target_location
        table[a_l_x, a_l_y] = 1.
        table[t_l_x, t_l_y] = -1.
        return table

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
        while self.target_location == self.agent_location:
            self.target_location = (random.randrange(self.size), random.randrange(self.size))
        observation = self.get_observation_as_table()
        distance = self.get_observation_as_table()

        if self.render_mode == "human":
            self.render_frame()

        grid_x, grid_y = torch.meshgrid(torch.arange(0, self.size), torch.arange(0, self.size), indexing='ij')
        t_l_y, t_l_x = self.target_location
        a_x = torch.ones_like(grid_x)*t_l_x
        a_y = torch.ones_like(grid_x)*t_l_y
        self.distances_from_target = torch.square(torch.subtract(a_x, grid_x)) + torch.square(torch.subtract(a_y, grid_y))

        return observation, distance

    def clip_locations(self, old_x, old_y, new_x, new_y):
        if new_x < 0 or new_y < 0 or new_x >= self.size or new_y >= self.size:
            return old_x, old_y
        else:
            return new_x, new_y

    def step(self, action):
        if action != self.wind_from:
            d_x, d_y = self.action_to_direction[action]
        else:
            d_x, d_y = (0, 0)
        a_l_x, a_l_y = self.agent_location
        a_l_x, a_l_y = self.clip_locations(a_l_x, a_l_y, a_l_x + d_x, a_l_y + d_y)
        self.agent_location = (a_l_x, a_l_y)

        terminated = self.agent_location == self.target_location
        reward = 0
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
        self.window.blit(background, background.get_rect())
        # pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        pygame.display.quit()
        pygame.quit()


if __name__ == "__main__":
    env = MyFirstSimpleEnvironment("human", fps=30000, wind_from=1)
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
