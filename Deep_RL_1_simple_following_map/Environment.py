import sys
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
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, fps=4):
        self.size = size
        self.render_mode = render_mode
        self.metadata["render_fps"] = fps

        self.window_size = 500
        self.window = None
        self.clock = None
        self.agent_location = None
        self.target_location = None

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int)
            })

        self.action_space = spaces.Discrete(4)
        self.action_to_direction = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

    def get_observation_as_locations(self):
        return {"agent": self.agent_location, "target": self.target_location}

    def get_observation_as_table(self):
        table = torch.zeros((self.size, self.size), dtype=torch.float32)
        a_l_x, a_l_y = self.agent_location
        t_l_x, t_l_y = self.target_location
        table[a_l_y, a_l_x] = 1.
        table[t_l_y, t_l_x] = -1.
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
        observation = self.get_observation_as_locations()
        distance = self.get_agent_target_distance()

        if self.render_mode == "human":
            self.render_frame()

        return observation, distance

    def clip_locations(self, x, y):
        if x < 0:
            x = 0
        if x >= self.size:
            x = self.size - 1
        if y < 0:
            y = 0
        if y >= self.size:
            y = self.size - 1
        return x, y

    def step(self, action):
        d_x, d_y = self.action_to_direction[action]
        a_l_x, a_l_y = self.agent_location
        a_l_x, a_l_y = self.clip_locations(a_l_x + d_x, a_l_y + d_y)
        self.agent_location = (a_l_x, a_l_y)

        terminated = self.agent_location == self.target_location
        if terminated:
            reward = 1
        else:
            reward = 0

        observation = self.get_observation_as_locations()
        distance = self.get_agent_target_distance()

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
        pygame.draw.circle(background, AGENT_COLOR, ((a_l_x + .5)*size_of_a_square_in_pixels,
                                                     (a_l_y + .5)*size_of_a_square_in_pixels),
                           size_of_a_square_in_pixels // 2.2)

        for i in range(self.size):
            pygame.draw.line(background, LINE_COLOR, (i * size_of_a_square_in_pixels, 0),
                             (i * size_of_a_square_in_pixels, self.window_size), width=3)
            pygame.draw.line(background, LINE_COLOR, (0, i * size_of_a_square_in_pixels),
                             (self.window_size, i * size_of_a_square_in_pixels), width=3)
        self.window.blit(background, background.get_rect())
        #pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        pygame.display.quit()
        pygame.quit()


if __name__ == "__main__":
    env = MyFirstSimpleEnvironment("human")
    env.reset()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit()
        env.step(random.randint(0, 3))
        print(env.get_observation_as_table())

