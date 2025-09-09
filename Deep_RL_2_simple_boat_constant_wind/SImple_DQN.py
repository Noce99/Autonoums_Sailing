import random
import matplotlib.pyplot as plt
import torch
from NNs import SimpleCnn
from collections import namedtuple
from Environment import MyFirstSimpleEnvironment
from tqdm import tqdm
from datetime import datetime

STARTING_EPSILON = 1
FINISHING_EPSILON = 0.02
EPSILON_ANGULAR_COEFFICIENT = -0.000001

START_BUFFER_SIZE = 100
MAX_BUFFER_SIZE = 10000
STEP_BEFORE_ALIGN_NETWORKS = 1000
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

    def update_epsilon(self, step_counter):
        self.epsilon = STARTING_EPSILON + EPSILON_ANGULAR_COEFFICIENT * step_counter

    @torch.no_grad()
    def game_step(self, env):
        if self.need_to_start_new_episode:
            env.reset()
            self.need_to_start_new_episode = False

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
        env = MyFirstSimpleEnvironment("no_visual", size=self.size, fps=self.fps)
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
            self.update_epsilon(step_counter)
            step_counter += 1
            my_bar.set_description(f"({step_counter} steps, epsilon={self.epsilon}, "
                                   f"buffer_size={len(self.buffer)})")

        plt.plot(self.loss_history)
        plt.title("Loss")
        plt.show()

    def visual_evaluation(self, random_actions=0.1):
        env = MyFirstSimpleEnvironment("human", size=self.size, fps=self.fps)
        while True:
            if self.need_to_start_new_episode:
                env.reset()
                self.need_to_start_new_episode = False
            state_s = env.get_observation_as_table()
            if random.random() < random_actions:
                action = random.randint(0, 7)
            else:
                action = int(torch.argmax(self.training_nn(state_s)))
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                self.need_to_start_new_episode = True


def new_train(size):
    dqn = SimpleDQN(size=size, training_steps=1000000, fps=1000000)
    dqn.start_training_session()
    now = datetime.now()
    torch.save(dqn.training_nn.state_dict(), f"./models/{now.second}_{now.minute}_{now.hour}_"
                                             f"{now.day}_{now.month}_{now.year}.pt")


def visual_evaluation(size, name):
    dqn = SimpleDQN(size=size, training_steps=0, weight_path=f"./models/{name}", fps=4)
    dqn.visual_evaluation()


if __name__ == "__main__":
    visual_evaluation(10, "21_35_11_7_7_2023.pt")
    #new_train(10)
