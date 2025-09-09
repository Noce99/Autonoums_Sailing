import random
import time

import matplotlib.pyplot as plt
import torch
from NNs import SimpleCnn
from collections import namedtuple
from Environment import MyFirstSimpleEnvironment
from tqdm import tqdm
from datetime import datetime

STARTING_EPSILON = 1.0
START_EPSILON_FINE_TUNING = 0.4
FINISHING_EPSILON = 0

START_BUFFER_SIZE = 100
MAX_BUFFER_SIZE = 10000
STEP_BEFORE_ALIGN_NETWORKS = 1000
STEP_BEFORE_SAVE_PLOTS = 10000
MINI_BATCH_SIZE = 10
GAMMA = 0.99
LEARNING_RATE = 0.001

BufferElement = namedtuple("BufferElement", ["state", "action", "reward", "next_state", "terminated"])


class SimpleDQN:
    def __init__(self, size, training_steps, weight_path=None, fps=4):
        self.size = size
        self.fps = fps
        self.training_nn = SimpleCnn(self.size)
        self.historical_nn = SimpleCnn(self.size)
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
        self.episode_steps_history = []
        self.epsilon_angular_coefficient = None
        self.last_starting_state = None
        self.hard_states = []
        if training_steps != 0:
            self.epsilon_angular_coefficient = (STARTING_EPSILON-FINISHING_EPSILON)/(0 - training_steps)

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
                             torch.arange(start=0, end=mini_batch_size * 8, step=8, dtype=torch.int)

        self.optimizer.zero_grad()

        training_nn_result = self.training_nn(mini_batch_state)
        action_values = torch.take(training_nn_result, mini_batch_actions)
        with torch.no_grad():
            next_states_value = torch.max(self.historical_nn(mini_batch_next_state), dim=1).values.detach()

        expected_action_values = next_states_value * GAMMA + mini_batch_reward

        loss_t = torch.nn.MSELoss()(action_values, expected_action_values)
        loss_t.backward()
        self.loss_history.append(loss_t.detach())
        self.optimizer.step()

    def update_epsilon(self, step_counter, starting_epsilon=STARTING_EPSILON):
        self.epsilon = starting_epsilon + self.epsilon_angular_coefficient * step_counter

    @torch.no_grad()
    def game_step(self, env, start_tensor=None):
        if self.need_to_start_new_episode:
            self.episode_steps_history.append(env.episode_steps)
            if self.epsilon < 0.3 and self.last_starting_state is not None and env.episode_steps > 10:
                self.hard_states.append(self.last_starting_state)
            if start_tensor is None:
                env.reset()
            else:
                env.reset()
                env.load_state_from_tensor(start_tensor)
            self.need_to_start_new_episode = False
            self.last_starting_state = env.get_observation_as_table()

        state_s = env.get_observation_as_table()
        if random.random() < self.epsilon:
            action = random.randint(0, 7)
        else:
            action = int(torch.argmax(self.training_nn(state_s)))
        observation, reward, terminated, _, distance = env.step(action)
        state_s_prime = env.get_observation_as_table()
        if terminated:
            self.need_to_start_new_episode = True
        self.add_element_to_buffer(BufferElement(state=state_s,
                                                 action=action,
                                                 reward=reward,
                                                 next_state=state_s_prime,
                                                 terminated=terminated))

    def start_training_session(self):
        env = MyFirstSimpleEnvironment("human", size=self.size, fps=self.fps)
        step_counter = 0

        for _ in tqdm(range(START_BUFFER_SIZE), colour="red", desc="Warm-up"):
            self.game_step(env)

        my_bar = tqdm(range(self.training_steps), colour="green", desc="Training")
        for _ in my_bar:
            self.game_step(env)
            if len(self.buffer) < START_BUFFER_SIZE:
                continue

            self.train_on_mini_batch_from_buffer(MINI_BATCH_SIZE)
            if step_counter % STEP_BEFORE_ALIGN_NETWORKS == 0:
                self.align_nns()

            if step_counter % STEP_BEFORE_SAVE_PLOTS == 0 and step_counter != 0:
                plt.plot(self.loss_history)
                plt.title("Loss")
                plt.savefig("./plots/loss.png")
                plt.clf()
                plt.plot(self.episode_steps_history)
                plt.title("Episode Steps")
                plt.savefig("./plots/steps.png")

            self.update_epsilon(step_counter)
            step_counter += 1
            my_bar.set_description(f"({step_counter} steps, epsilon={self.epsilon:.2f}, "
                                   f"buffer_size={len(self.buffer)}, hard_states={len(self.hard_states)})")

        if len(self.hard_states) != 0:
            print(f"shape of an element = {self.hard_states[0].shape}")
            hard_states_tensor = torch.cat([tensor[None, :] for tensor in self.hard_states], dim=0)
            print(f"Shape of Hard States Tensor = {hard_states_tensor.shape}")
            torch.save(hard_states_tensor, "hard_states.tensor")
        else:
            print("No Hard State found!")

    def start_hard_states_training(self, steps):
        self.epsilon = START_EPSILON_FINE_TUNING
        self.epsilon_angular_coefficient = (START_EPSILON_FINE_TUNING - FINISHING_EPSILON) / (0 - steps)

        hard_states_tensor = torch.load("hard_states.tensor")
        env = MyFirstSimpleEnvironment("human", size=self.size, fps=self.fps)

        my_bar = tqdm(range(steps), colour="yellow", desc="Final Training")

        step_counter = 0

        for _ in my_bar:
            self.game_step(env, hard_states_tensor[random.randrange(0, hard_states_tensor.shape[0]), :])
            if len(self.buffer) < START_BUFFER_SIZE:
                continue

            self.train_on_mini_batch_from_buffer(MINI_BATCH_SIZE)
            if step_counter % STEP_BEFORE_ALIGN_NETWORKS == 0:
                self.align_nns()

            if step_counter % STEP_BEFORE_SAVE_PLOTS == 0 and step_counter != 0:
                plt.plot(self.loss_history)
                plt.title("Loss")
                plt.savefig("./plots/loss.png")
                plt.clf()
                plt.plot(self.episode_steps_history)
                plt.title("Episode Steps")
                plt.savefig("./plots/steps.png")

            self.update_epsilon(step_counter, starting_epsilon=START_EPSILON_FINE_TUNING)
            step_counter += 1
            my_bar.set_description(f"({step_counter} steps, epsilon={self.epsilon:.2f}, "
                                   f"buffer_size={len(self.buffer)}")

    def visual_evaluation(self, epsilon=0.0):
        env = MyFirstSimpleEnvironment("human", size=self.size, fps=self.fps)
        while True:
            if self.need_to_start_new_episode:
                env.reset()
                self.need_to_start_new_episode = False
            state_s = env.get_observation_as_table()
            print(state_s)
            exit()
            if random.random() < epsilon:
                action = random.randint(0, 7)
            else:
                action = int(torch.argmax(self.training_nn(state_s)))
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                self.need_to_start_new_episode = True


def new_train(size):
    dqn = SimpleDQN(size=size, training_steps=6000000, fps=1000000)
    dqn.start_training_session()
    now = datetime.now()
    torch.save(dqn.training_nn.state_dict(), f"./models/{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_"
                                             f"{now.second}.pt")


def visual_evaluation(size, name, epsilon=0.0):
    dqn = SimpleDQN(size=size, training_steps=0, weight_path=f"./models/{name}", fps=4)
    dqn.visual_evaluation(epsilon)


def fine_tuning(size, name, steps=600000):
    dqn = SimpleDQN(size=size, training_steps=600000, weight_path=f"./models/{name}", fps=1000000)
    dqn.start_hard_states_training(steps)
    now = datetime.now()
    torch.save(dqn.training_nn.state_dict(), f"./models/{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_"
                                             f"{now.second}.pt")


if __name__ == "__main__":
    visual_evaluation(5, "2023_8_4_1_27_58.pt", epsilon=0.05)
    # new_train(10)
    #fine_tuning(5, "2023_7_25_9_39_0.pt")

