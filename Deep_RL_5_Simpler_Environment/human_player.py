import pygame

from Sailing_Boats_Autopilot.Deep_RL_5_Simpler_Environment.Environment import RealisticEnvironment
from Sailing_Boats_Autopilot.utils import random_location_in_atlantic_ocean

if __name__ == "__main__":
    env = RealisticEnvironment(start_point=random_location_in_atlantic_ocean,
                               target_point=random_location_in_atlantic_ocean, fps=1,
                               render_mode="human", time_in_second_for_step=10000)
    env.reset()
    need_to_start_new_episode = False
    action_to_be_processed = None
    while True:
        if need_to_start_new_episode:
            env.reset()
            need_to_start_new_episode = False
        state_s = env.observation

        if action_to_be_processed is None:
            action = 2
        else:
            action = action_to_be_processed
            action_to_be_processed = None

        _, _, terminated, _, _ = env.step(action)
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
