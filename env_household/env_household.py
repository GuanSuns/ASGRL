import cv2
import gym
from addict import addict
from gym import spaces
import os
import numpy as np
import plotly.colors as colors
from skimage.transform import resize

from utils.grid_renderer import Grid_Renderer


class Env_Household:
    # noinspection PySetFunctionToLiteral
    def __init__(self, success_reward=0):
        """
        Success_reward is 0 by default since in ASGRL & other
                Planning+RL baselines, the reward is given by the symbolic model (not from the env)
        But to run Vanilla Q-Learning, success_reward should be set to 1.
        """
        # actions: up, down, left, right
        self.action_space = spaces.Discrete(4)
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(226,),
                                            dtype=np.int16)
        self.obs_type = np.int16
        # game config
        self.success_reward = success_reward
        self.height = 15
        self.width = 15
        self.objects = addict.Dict({
            'agent': {
                'id': 1,
                'location': (9, 3),
                'color': colors.hex_to_rgb(colors.qualitative.D3[7])  # gray
            },
            'wall': {
                'id': 2,
                'locations': [(i, 0) for i in range(self.height)] +
                             [(i, self.width-1) for i in range(self.height)] +
                             [(self.height - 1, i) for i in range(self.width)] +
                             [(0, i) for i in range(self.width)] +
                             [(i, 4) for i in range(self.height)] +
                             [(i, 8) for i in range(self.height)] +
                             [(i, 11) for i in range(self.height)] +
                             [(7, i) for i in range(4, 9)] +
                             [(11, i) for i in range(4, 9)] +
                             [(9, i) for i in range(8, 12)] +
                             [(4, i) for i in range(8, 12)],
                'color': colors.hex_to_rgb(colors.qualitative.D3[5])  # dark brown
            },
            'charging_dock': {
                'id': 3,
                'location': (1, 5),
                'color': colors.hex_to_rgb(colors.qualitative.Plotly[5])  # blue
            },
            'no_lock_doors': {
                'id': 4,
                'locations': [(11, 6), (7, 6), (5, 8), (2, 8), (10, 8)],
                'color': colors.hex_to_rgb(colors.qualitative.Plotly[9])  # yellow
            },
            'lock_door_0': {
                'id': 5,
                'location': (9, 4),
                'color': colors.hex_to_rgb(colors.qualitative.Plotly[3])  # purple
            },
            'unlock_door_0': {
                'id': 6,
                'location': (9, 4),
                'color': colors.hex_to_rgb(colors.qualitative.Plotly[9])  # yellow
            },
            'lock_door_1': {
                'id': 7,
                'location': (10, 8),
                'color': colors.hex_to_rgb(colors.qualitative.Plotly[3])  # purple
            },
            'unlock_door_1': {
                'id': 8,
                'location': (10, 8),
                'color': colors.hex_to_rgb(colors.qualitative.Plotly[9])  # yellow
            },
            'final_door': {
                'id': 9,
                'location': (12, 11),
                'color': colors.hex_to_rgb(colors.qualitative.Plotly[9])  # yellow
            },
            'destination': {
                'id': 10,
                'location': (1, 13),
                'color': colors.hex_to_rgb(colors.qualitative.Alphabet[17])  # red
            },
            'target_key_0': {
                'id': 11,
                'location': (1, 2),
                'color': colors.hex_to_rgb(colors.qualitative.Plotly[2])  # green
            },
            'target_key_1': {
                'id': 12,
                'location': (4, 1),
                'color': colors.hex_to_rgb(colors.qualitative.Plotly[2])  # green
            },
            'no_use_key_0': {
                'id': 13,
                'location': (6, 1),
                'color': colors.hex_to_rgb(colors.qualitative.D3[1])  # orange
            },
        })
        # finalize the coordinates of wall
        wall_positions = set(self.objects.wall['locations'])
        wall_positions = wall_positions - set(self.objects.no_lock_doors['locations'])
        wall_positions = wall_positions - set([self.objects.lock_door_0['location']])
        wall_positions = wall_positions - set([self.objects.lock_door_1['location']])
        wall_positions = wall_positions - set([self.objects.final_door['location']])
        self.wall_positions = set(wall_positions)
        self.objects.wall['locations'] = list(wall_positions)
        # save the coordinates of no-lock doors
        self.no_lock_door_positions = set(self.objects.no_lock_doors.locations)
        # map
        self.grid = np.zeros(shape=(self.height, self.width))
        # flags
        self.curr_room = 0
        self.in_room_0 = True
        self.visited_room_0 = True
        self.in_room_1 = False
        self.visited_room_1 = False
        self.in_room_2 = False
        self.visited_room_2 = False
        self.in_room_3 = False
        self.visited_room_3 = False
        self.in_room_4 = False
        self.visited_room_4 = False
        self.in_room_5 = False
        self.visited_room_5 = False
        self.in_room_6 = False
        self.visited_room_6 = False
        self.in_room_7 = False
        self.visited_room_7 = False
        self.is_charged = False
        self.carrying_key = False
        self.carried_key = set()
        self.carrying_target_key_0 = False
        self.carrying_target_key_1 = False
        self.carrying_no_use_key = False

        self.door_0_unlocked = False
        self.door_1_unlocked = False
        self.at_destination = False
        self.failed = False
        # state
        self.info = dict()
        self.agent_pos = self.objects.agent.location
        # init
        self._init()
        # init renderer
        color_map = {self.objects[obj].id: self.objects[obj].color for obj in self.objects}
        self.renderer = Grid_Renderer(grid_size=20, color_map=color_map)

    def _reset_in_room_info(self):
        self.in_room_0 = True
        self.in_room_1 = False
        self.in_room_2 = False
        self.in_room_3 = False
        self.in_room_4 = False
        self.in_room_5 = False
        self.in_room_6 = False
        self.in_room_7 = False

    def _init(self):
        # update flags and state info
        self.curr_room = 0
        self.in_room_0 = True
        self.visited_room_0 = True
        self.in_room_1 = False
        self.visited_room_1 = False
        self.in_room_2 = False
        self.visited_room_2 = False
        self.in_room_3 = False
        self.visited_room_3 = False
        self.in_room_4 = False
        self.visited_room_4 = False
        self.in_room_5 = False
        self.visited_room_5 = False
        self.in_room_6 = False
        self.visited_room_6 = False
        self.in_room_7 = False
        self.visited_room_7 = False
        self.is_charged = False
        self.carrying_key = False
        self.carried_key = set()
        self.carrying_target_key_0 = False
        self.carrying_target_key_1 = False
        self.carrying_no_use_key = False
        self.door_0_unlocked = False
        self.door_1_unlocked = False
        self.at_destination = False
        self.failed = False

        self.info = dict()
        self.agent_pos = self.objects.agent.location
        # init grid
        self.grid = np.zeros(shape=(self.height, self.width))
        # init wall
        for wall_pos in self.wall_positions:
            self.grid[wall_pos[0], wall_pos[1]] = self.objects.wall.id
        # place charging dock
        charging_pos = self.objects.charging_dock.location
        self.grid[charging_pos[0], charging_pos[1]] = self.objects.charging_dock.id
        # place doors
        for door_pos in self.objects.no_lock_doors.locations:
            self.grid[door_pos[0], door_pos[1]] = self.objects.no_lock_doors.id
        # lock door 0
        lock_door_pos = self.objects.lock_door_0.location
        self.grid[lock_door_pos[0], lock_door_pos[1]] = self.objects.lock_door_0.id
        # lock door 1
        lock_door_pos = self.objects.lock_door_1.location
        self.grid[lock_door_pos[0], lock_door_pos[1]] = self.objects.lock_door_1.id
        # final door
        final_door_pos = self.objects.final_door.location
        self.grid[final_door_pos[0], final_door_pos[1]] = self.objects.final_door.id
        # place the agent
        self.grid[self.agent_pos[0], self.agent_pos[1]] = self.objects.agent.id
        # place the keys
        self.grid[self.objects.target_key_0.location[0], self.objects.target_key_0.location[1]] = self.objects.target_key_0.id
        self.grid[self.objects.target_key_1.location[0], self.objects.target_key_1.location[1]] = self.objects.target_key_1.id
        self.grid[self.objects.no_use_key_0.location[0], self.objects.no_use_key_0.location[1]] = self.objects.no_use_key_0.id
        # place destination
        dest_pos = self.objects.destination.location
        self.grid[dest_pos[0], dest_pos[1]] = self.objects.destination.id

    def _get_curr_room(self):
        """ Return the current room id """
        agent_pos = self.agent_pos
        if 0 < agent_pos[0] < self.height - 1 and 0 < agent_pos[1] < 4:
            return 0
        elif 11 < agent_pos[0] < self.height - 1 and 4 < agent_pos[1] < 8:
            return 1
        elif 7 < agent_pos[0] < 11 and 4 < agent_pos[1] < 8:
            return 2
        elif 0 < agent_pos[0] < 7 and 4 < agent_pos[1] < 8:
            return 3
        elif 9 < agent_pos[0] < self.height - 1 and 8 < agent_pos[1] < 11:
            return 4
        elif 4 < agent_pos[0] < 9 and 8 < agent_pos[1] < 11:
            return 5
        elif 0 < agent_pos[0] < 4 and 8 < agent_pos[1] < 11:
            return 6
        elif 0 < agent_pos[0] < self.height - 1 and 11 < agent_pos[1] < self.width - 1:
            return 7
        else:
            # not in any room (e.g. when passing through certain door)
            return -1

    def _get_grid_obs(self):
        grid_obs = np.copy(self.grid).flatten()
        obs = np.append(grid_obs, [int(self.is_charged)])
        return obs.astype(self.obs_type)

    def render(self, mode='rgb_array'):
        return self.renderer.render_2d_grid(self.grid)

    def step(self, action):
        def _go_to(x, y):
            _old_pos = tuple(self.agent_pos)
            self.agent_pos = (x, y)
            self.grid[x, y] = self.objects.agent.id
            if self.agent_pos != _old_pos:
                if _old_pos == self.objects.charging_dock.location:
                    self.grid[_old_pos[0], _old_pos[1]] = self.objects.charging_dock.id
                elif _old_pos in self.no_lock_door_positions:
                    self.grid[_old_pos[0], _old_pos[1]] = self.objects.no_lock_doors.id
                elif _old_pos == self.objects.unlock_door_0.location:
                    self.grid[_old_pos[0], _old_pos[1]] = self.objects.unlock_door_0.id
                elif _old_pos == self.objects.unlock_door_1.location:
                    self.grid[_old_pos[0], _old_pos[1]] = self.objects.unlock_door_1.id
                else:
                    self.grid[_old_pos[0], _old_pos[1]] = 0
        assert action <= 3
        old_pos = tuple(self.agent_pos)
        next_x, next_y = self.agent_pos[0] + self.actions[action][0], self.agent_pos[1] + self.actions[action][1]
        # update grid
        # check if the agent will hit into walls, then make no change
        if self.grid[next_x, next_y] == self.objects.wall.id:
            pass
        elif self.grid[next_x, next_y] == self.objects.charging_dock.id:
            # should not recharge until all the doors are unlocked
            if self.door_0_unlocked and self.door_1_unlocked:
                self.is_charged = True
                _go_to(next_x, next_y)
            else:
                # otherwise won't have the power to complete remaining steps
                self.is_charged = True
                self.failed = True
                _go_to(next_x, next_y)
        # key pickup
        elif self.grid[next_x, next_y] == self.objects.target_key_0.id:
            if not self.carrying_key:
                self.carried_key.add('target_key_0')
                self.carrying_key = True
                self.carrying_target_key_0 = True
                _go_to(next_x, next_y)
        elif self.grid[next_x, next_y] == self.objects.target_key_1.id:
            if not self.carrying_key:
                self.carried_key.add('target_key_1')
                self.carrying_key = True
                self.carrying_target_key_1 = True
                _go_to(next_x, next_y)
        elif self.grid[next_x, next_y] == self.objects.no_use_key_0.id:
            if not self.carrying_key:
                self.carried_key.add('no_use_key_0')
                self.carrying_key = True
                self.carrying_no_use_key = True
                _go_to(next_x, next_y)
        # at the final door
        elif (next_x, next_y) == self.objects.final_door.location:
            final_door_pos = self.objects.final_door.location
            # only allow entering room 7 not existing room 7 due to limited power capacity
            if old_pos == (final_door_pos[0], final_door_pos[1] - 1):
                next_x, next_y = final_door_pos[0], final_door_pos[1] + 1
                _go_to(next_x, next_y)
        # at door 0
        elif (next_x, next_y) == self.objects.lock_door_0.location:
            # if the door is lock now
            if self.grid[next_x, next_y] == self.objects.lock_door_0.id:
                lock_door_pos = self.objects.lock_door_0.location
                # if facing room 2 (not existing room 2)
                if old_pos == (lock_door_pos[0], lock_door_pos[1] - 1):
                    if self.carrying_target_key_0:
                        self.carrying_target_key_0 = False
                        self.carrying_key = False
                        # unlock the door
                        self.door_0_unlocked = True
                        self.grid[next_x, next_y] = self.objects.unlock_door_0.id
            else:
                _go_to(next_x, next_y)
        # at door 1
        elif (next_x, next_y) == self.objects.lock_door_1.location:
            # if the door is lock now
            if self.grid[next_x, next_y] == self.objects.lock_door_1.id:
                lock_door_pos = self.objects.lock_door_1.location
                # if facing room 4 (not existing room 4)
                if old_pos == (lock_door_pos[0], lock_door_pos[1] - 1):
                    if self.carrying_target_key_1:
                        self.carrying_target_key_1 = False
                        self.carrying_key = False
                        # unlock the door
                        self.door_1_unlocked = True
                        self.grid[next_x, next_y] = self.objects.unlock_door_1.id
            else:
                _go_to(next_x, next_y)
        # at destination
        elif (next_x, next_y) == self.objects.destination.location:
            if self.is_charged:
                self.at_destination = True
                _go_to(next_x, next_y)
            else:
                self.failed = True
        elif self.grid[next_x, next_y] == 0 or self.grid[next_x, next_y] == self.objects.no_lock_doors.id:
            _go_to(next_x, next_y)
        # update flags
        curr_room = self._get_curr_room()
        self.curr_room = curr_room
        self._reset_in_room_info()
        if curr_room == 0:
            self.in_room_0 = True
            self.visited_room_0 = True
        elif curr_room == 1:
            self.in_room_1 = True
            self.visited_room_1 = True
        elif curr_room == 2:
            self.in_room_2 = True
            self.visited_room_2 = True
        elif curr_room == 3:
            self.in_room_3 = True
            self.visited_room_3 = True
        elif curr_room == 4:
            self.in_room_4 = True
            self.visited_room_4 = True
        elif curr_room == 5:
            self.in_room_5 = True
            self.visited_room_5 = True
        elif curr_room == 6:
            self.in_room_6 = True
            self.visited_room_6 = True
        elif curr_room == 7:
            self.in_room_7 = True
            self.visited_room_7 = True
        # update info
        self.info['visited_room_0'] = self.visited_room_0
        self.info['visited_room_1'] = self.visited_room_1
        self.info['visited_room_2'] = self.visited_room_2
        self.info['visited_room_3'] = self.visited_room_3
        self.info['visited_room_4'] = self.visited_room_4
        self.info['visited_room_5'] = self.visited_room_5
        self.info['visited_room_6'] = self.visited_room_6
        self.info['visited_room_7'] = self.visited_room_7
        self.info['door_1_unlocked'] = self.door_1_unlocked
        self.info['door_0_unlocked'] = self.door_0_unlocked
        self.info['key_picked_first_door'] = (len(self.carried_key) >= 1)
        self.info['key_picked_second_door'] = (len(self.carried_key) >= 2)
        self.info['is_charged'] = self.is_charged
        self.info['at_destination'] = self.at_destination
        # return info
        done = self.at_destination or self.failed
        reward = self.success_reward if self.at_destination else 0
        return self._get_grid_obs(), float(reward), done, addict.Dict(self.info)

    def reset(self, **kwargs):
        self._init()
        return self._get_grid_obs()


def main():
    print('starting household')
    import cv2
    env = Env_Household()
    obs = env.reset()
    print(f'[INFO] observation shape: {obs.shape}')
    done = False
    info = None

    while not done:
        grid_img = env.render()
        cv2.imshow('grid render', cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
        a = cv2.waitKey(0)
        # actions: up, down, left, right
        if a == ord('q'):
            break
        elif a == ord('w'):
            a = 0
        elif a == ord('a'):
            a = 2
        elif a == ord('s'):
            a = 1
        elif a == ord('d'):
            a = 3
        elif a == ord('e'):
            a = 4
        obs, reward, done, info = env.step(int(a))
        print(reward, done, info)


if __name__ == "__main__":
    main()
