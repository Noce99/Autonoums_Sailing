import math
import random
import time

import matplotlib.pyplot as plt
import pygame
import torch
from Sailing_Boats_Autopilot.Deep_RL_5_Simpler_Environment.NNs import SimpleCnn
from collections import namedtuple
from Sailing_Boats_Autopilot.Deep_RL_5_Simpler_Environment.Environment import RealisticEnvironment, \
    GRID_PIXELS_HALF_SIZE
from Sailing_Boats_Autopilot.constants import ABSOLUTE_PATH
from Sailing_Boats_Autopilot.utils import random_location_in_atlantic_ocean, from_start_end_to_direction
from tqdm import tqdm
from datetime import datetime
from guppy import hpy

STARTING_EPSILON = 1.0
START_EPSILON_FINE_TUNING = 0.4
FINISHING_EPSILON = 0

START_BUFFER_SIZE = 10000
MAX_BUFFER_SIZE = 10000
STEP_BEFORE_ALIGN_NETWORKS = 5000
STEP_BEFORE_SAVE_PLOTS = 100000
MINI_BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE = 0.001

HISTORY_TMP_LIST_SIZE = 1000

BufferElement = namedtuple("BufferElement", ["state", "action", "reward", "next_state", "terminated"])


class SimpleDQN:
    def __init__(self, training_steps, weight_path=None, fps=4, render_mode="human"):
        self.size = GRID_PIXELS_HALF_SIZE * 2 + 1
        self.fps = fps
        self.render_mode = render_mode
        self.training_nn = SimpleCnn(self.size).cuda()
        self.historical_nn = SimpleCnn(self.size).cuda()
        if weight_path is not None:
            self.training_nn.load_state_dict(torch.load(weight_path))
            self.historical_nn.load_state_dict(torch.load(weight_path))
        self.buffer = []
        self.episode_reward = 0
        self.need_to_start_new_episode = True
        self.epsilon = STARTING_EPSILON
        self.training_steps = training_steps
        self.optimizer = torch.optim.Adam(self.training_nn.parameters(), lr=LEARNING_RATE)
        self.loss_history = []
        self.tmp_loss_history = []
        self.episode_steps_history = []
        self.tmp_reward_history = []
        self.reward_history = []
        self.epsilon_angular_coefficient = None

        if training_steps != 0:
            self.epsilon_angular_coefficient = (STARTING_EPSILON - FINISHING_EPSILON) / (0 - training_steps)

    def align_nns(self):
        self.historical_nn.load_state_dict(self.training_nn.state_dict())

    def add_element_to_buffer(self, x):
        self.buffer.append(x)
        if len(self.buffer) > MAX_BUFFER_SIZE:
            self.buffer.pop(0)

    def train_on_mini_batch_from_buffer(self, mini_batch_size):
        mini_batch_size = min(mini_batch_size, len(self.buffer))
        selected_elements = random.choices(self.buffer, k=mini_batch_size)
        mini_batch_state = torch.concatenate([element.state[None, :] for element in selected_elements], axis=0)
        mini_batch_next_state = torch.concatenate([element.next_state[None, :] for element in selected_elements],
                                                  axis=0)
        mini_batch_reward = torch.Tensor([element.reward for element in selected_elements])
        mini_batch_actions = torch.tensor([element.action for element in selected_elements]) + \
                             torch.arange(start=0, end=mini_batch_size * 3, step=3, dtype=torch.int)

        self.optimizer.zero_grad()

        training_nn_result = self.training_nn(mini_batch_state.cuda())

        action_values = torch.take(training_nn_result.cpu(), mini_batch_actions)
        with torch.no_grad():
            next_states_value = torch.max(self.historical_nn(mini_batch_next_state.cuda()), dim=1).values.detach()

        expected_action_values = next_states_value.cpu() * GAMMA + mini_batch_reward

        loss_t = torch.nn.MSELoss()(action_values, expected_action_values)
        loss_t.backward()
        self.tmp_loss_history.append(loss_t.detach())
        if len(self.tmp_loss_history) > HISTORY_TMP_LIST_SIZE:
            self.loss_history.append(sum(self.tmp_loss_history) / len(self.tmp_loss_history))
            self.tmp_loss_history = []
        self.optimizer.step()

    def update_epsilon(self, step_counter, starting_epsilon=STARTING_EPSILON):
        self.epsilon = starting_epsilon + self.epsilon_angular_coefficient * step_counter

    @torch.no_grad()
    def game_step(self, env, start_tensor=None):
        if self.need_to_start_new_episode:
            # self.episode_steps_history.append(env.episode_steps)
            """
            if self.epsilon < 0.3 and self.last_starting_state is not None and env.episode_steps > 10:
                self.hard_states.append(self.last_starting_state)
            """
            env.reset()
            if start_tensor is not None:
                # env.load_state_from_tensor(start_tensor)
                raise NotImplementedError
            self.need_to_start_new_episode = False
            # self.last_starting_state = env.get_observation_as_table()

        state_s = env.observation

        if random.random() < self.epsilon:
            action = random.randint(0, 2)
        else:
            action = int(torch.argmax(self.training_nn(state_s.cuda())))
        observation, reward, terminated, _, distance = env.step(action)
        state_s_prime = env.observation

        self.tmp_reward_history.append(reward)
        if len(self.tmp_reward_history) > HISTORY_TMP_LIST_SIZE:
            self.reward_history.append(sum(self.tmp_reward_history) / len(self.tmp_reward_history))
            self.tmp_reward_history = []

        if terminated:
            self.need_to_start_new_episode = True
        self.add_element_to_buffer(BufferElement(state=state_s,
                                                 action=action,
                                                 reward=reward,
                                                 next_state=state_s_prime,
                                                 terminated=terminated))

    def start_training_session(self):
        env = RealisticEnvironment(start_point=random_location_in_atlantic_ocean,
                                   target_point=random_location_in_atlantic_ocean, fps=self.fps,
                                   render_mode=self.render_mode, time_in_second_for_step=10000)
        step_counter = 0

        for _ in tqdm(range(START_BUFFER_SIZE), colour="red", desc="Warm-up"):
            self.game_step(env)

        my_bar = tqdm(range(self.training_steps), colour="green", desc="Training")
        for _ in my_bar:
            self.game_step(env)
            self.train_on_mini_batch_from_buffer(MINI_BATCH_SIZE)
            if step_counter % STEP_BEFORE_ALIGN_NETWORKS == 0:
                self.align_nns()

            if step_counter % (self.training_steps // 100) == 0 and step_counter != 0:
                plt.plot(self.loss_history)
                plt.title("Loss")
                plt.savefig(f"{ABSOLUTE_PATH}/Deep_RL_5_Simpler_Environment/plots/loss.png")
                plt.clf()
                plt.plot(self.reward_history)
                plt.title("Reward")
                plt.savefig(f"{ABSOLUTE_PATH}/Deep_RL_5_Simpler_Environment/plots/reward.png")
                plt.clf()

            if step_counter % (self.training_steps // 100) == 0 and step_counter != 0:
                now = datetime.now()
                torch.save(self.training_nn.state_dict(), f"{ABSOLUTE_PATH}/Deep_RL_5_Simpler_Environment/models/"
                                                          f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_"
                                                          f"{now.second}.pt")

            self.update_epsilon(step_counter)
            step_counter += 1
            my_bar.set_description(f"({step_counter} steps, epsilon={self.epsilon:.2f}, "
                                   f"buffer_size={len(self.buffer)})")

            if self.render_mode == "human":
                event_list = pygame.event.get()
                for ev in event_list:
                    if ev.type == pygame.QUIT:
                        exit()

    def visual_evaluation(self, epsilon=0.0):
        env = RealisticEnvironment(start_point=random_location_in_atlantic_ocean,
                                   target_point=random_location_in_atlantic_ocean, fps=self.fps,
                                   render_mode="human", time_in_second_for_step=10000)
        while True:
            if self.need_to_start_new_episode:
                env.reset()
                self.need_to_start_new_episode = False
            state_s = env.observation

            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                actions_values = self.training_nn(state_s.cuda())
                # print(actions_values)
                action = int(torch.argmax(actions_values))
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                self.need_to_start_new_episode = True
            event_list = pygame.event.get()
            for ev in event_list:
                if ev.type == pygame.QUIT:
                    exit()


def new_train(weight_name=None, render_mode="human"):
    if weight_name is not None:
        weight_path = f"{ABSOLUTE_PATH}/Deep_RL_5_Simpler_Environment/models/{weight_name}"
    else:
        weight_path = None
    dqn = SimpleDQN(training_steps=5000000, fps=20000000, weight_path=weight_path, render_mode=render_mode)
    dqn.start_training_session()
    now = datetime.now()
    torch.save(dqn.training_nn.state_dict(), f"{ABSOLUTE_PATH}/Deep_RL_5_Simpler_Environment/models/"
                                             f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_"
                                             f"{now.second}.pt")


def visual_evaluation(name, epsilon=0.0):
    dqn = SimpleDQN(training_steps=0, weight_path=f"{ABSOLUTE_PATH}/Deep_RL_5_Simpler_Environment/"
                                                  f"models/{name}", fps=30)
    dqn.visual_evaluation(epsilon)


def fine_tuning(size, name, steps=600000):
    raise NotImplementedError


if __name__ == "__main__":
    visual_evaluation("buoni_pacifico.pt", epsilon=0.05)
    # new_train(render_mode="no-human", weight_name=None)
    # fine_tuning(5, "2023_7_25_9_39_0.pt")
