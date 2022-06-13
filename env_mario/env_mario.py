import cv2
import gym
from addict import addict
from gym import spaces
import os
import numpy as np
import plotly.colors as colors
from skimage.transform import resize

from utils.grid_renderer import Grid_Renderer


class Env_Mario:
    def __init__(self, use_state=True, info_img=False):
        self.pytorch_img_state = False
        self.info_img = info_img
        # actions: up, down, left, right
        self.action_space = spaces.Discrete(4)
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        self.img_h = 84
        self.img_w = 84
        self.use_state = use_state
        if self.use_state:
            self.observation_space = spaces.Box(low=0, high=1, shape=(88, ), dtype=np.int16)
            self.obs_type = np.int16
        else:
            self.observation_space = gym.spaces.Box(low=0, high=255,
                                                    shape=(1, self.img_h, self.img_w),
                                                    dtype=np.uint8)
            self.obs_type = np.uint8
        # game config
        self.hard_exploration = True    # whether to make the exploration harder
        self.success_reward = 0
        self.height = 8
        self.width = 11
        self.objects = addict.Dict({
            'agent': {
                'id': 1,
                'location': (1, 1),
                'color': colors.hex_to_rgb(colors.qualitative.Plotly[1])  # red
            },
            'walkable_area': {
                'id': 0,
                'locations': [(1, i) for i in range(1, self.width - 2)] + [(6, i) for i in range(1, self.width - 1)],
                'color': (0, 0, 0),
            },
            'wall': {
                'id': 2,
                'locations': [],  # will be filled in automatically
                'color': colors.hex_to_rgb(colors.qualitative.D3[5])  # dark brown
            },
            'tube': {
                'id': 3,
                'locations': [(i, self.width-4) for i in range(2, 6)],
                'color': colors.hex_to_rgb(colors.qualitative.Plotly[7])  # light green
            },
            'ladder': {
                'id': 4,
                'locations': [(i, 5) for i in range(2, 6)],
                'color': colors.hex_to_rgb(colors.qualitative.Light24[6])  # yellow
            },
            'worn_ladder': {
                'id': 5,
                'color': colors.hex_to_rgb(colors.qualitative.D3[8])  # orange
            },
            'hidden_key': {
                'id': 6,
                'location': (5, 3),
                'color': colors.hex_to_rgb(colors.qualitative.Plotly[5])  # blue
            },
            'key': {
                'id': 7,
                'location': (5, 1),
                'color': colors.hex_to_rgb(colors.qualitative.Plotly[3])  # purple
            },
            'door': {
                'id': 8,
                'location': (1, self.width-2),
                'color': colors.hex_to_rgb(colors.qualitative.D3[2])  # green
            },
            'dead_agent': {
                'id': 9,
                'color': colors.hex_to_rgb(colors.qualitative.D3[7])  # gray
            },
        })
        # automatically compute the coordinates of wall
        self.wall_locations = set()
        occupied_pos = set()
        for obj in self.objects:
            if 'location' in self.objects[obj]:
                occupied_pos.add(self.objects[obj].location)
            elif 'locations' in self.objects[obj]:
                [occupied_pos.add(pos) for pos in self.objects[obj].locations]
        for x in range(self.height):
            for y in range(self.width):
                if (x, y) not in occupied_pos:
                    self.wall_locations.add((x, y))
        self.objects.wall.locations = list(self.wall_locations)
        # other obj info
        self.n_objects = len(self.objects)
        self.grid = np.zeros(shape=(self.height, self.width))
        # flags
        self.at_ladder = False
        self.visited_ladder = False
        self.at_tube = False
        self.visited_tube = False
        self.visited_bottom = False
        self.back_to_upper = False
        self.picked_hidden_key = False
        self.picked_key = False
        self.door_opened = False
        self.agent_dead = False
        # state
        self.info = dict()
        self.agent_pos = self.objects.agent.location
        self.wall_locations = self.objects.wall.locations
        self.ladder_locations = set(self.objects.ladder.locations)
        self.tube_locations = set(self.objects.tube.locations)
        # init
        self._init()
        # init renderer
        color_map = {self.objects[obj].id:self.objects[obj].color for obj in self.objects}
        self.renderer = Grid_Renderer(grid_size=20, color_map=color_map)

    def _init(self):
        # update flags and state info
        self.at_ladder = False
        self.visited_ladder = False
        self.at_tube = False
        self.visited_tube = False
        self.picked_hidden_key = False
        self.picked_key = False
        self.door_opened = False
        self.agent_dead = False
        self.visited_bottom = False
        self.back_to_upper = False

        self.info = dict()
        self.agent_pos = self.objects.agent.location
        # init grid
        self.grid = np.zeros(shape=(self.height, self.width))
        # init wall
        for wall_pos in self.wall_locations:
            self.grid[wall_pos[0], wall_pos[1]] = self.objects.wall.id
        # place the agent
        self.grid[self.agent_pos[0], self.agent_pos[1]] = self.objects.agent.id
        # place the tube
        for tube_pos in self.objects.tube.locations:
            self.grid[tube_pos[0], tube_pos[1]] = self.objects.tube.id
        # place the tube
        for ladder in self.objects.ladder.locations:
            self.grid[ladder[0], ladder[1]] = self.objects.ladder.id
        # place the key
        self.grid[self.objects.key.location[0], self.objects.key.location[1]] = self.objects.key.id
        # place the hidden key
        self.grid[self.objects.hidden_key.location[0], self.objects.hidden_key.location[1]] = self.objects.hidden_key.id
        # place the door
        self.grid[self.objects.door.location[0], self.objects.door.location[1]] = self.objects.door.id

    def render(self, mode='rgb_array'):
        return self.renderer.render_2d_grid(self.grid)

    def _get_img_obs(self):
        rgb_img = self.render()
        # obs = np.expand_dims(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY), axis=2)
        obs = resize(rgb_img, (84, 84), anti_aliasing=False)
        if self.pytorch_img_state:
            # to Pytorch channel format
            obs = np.moveaxis(obs, -1, 0) / 255.0
            return obs
        else:
            flatten_obs = obs.flatten().astype(np.float16).tolist()
            return flatten_obs

    def _get_grid_obs(self):
        grid_obs = np.copy(self.grid).flatten()
        return grid_obs.astype(self.obs_type)

    def _obs(self):
        if not self.use_state:
            return self._get_img_obs()
        else:
            return self._get_grid_obs()

    def step(self, action):
        def _go_to(x, y):
            old_pos = tuple(self.agent_pos)
            self.agent_pos = (x, y)
            self.grid[x, y] = self.objects.agent.id
            if self.agent_pos != old_pos:
                if old_pos in self.ladder_locations:
                    self.grid[old_pos[0], old_pos[1]] = self.objects.worn_ladder.id
                elif old_pos in self.tube_locations:
                    self.grid[old_pos[0], old_pos[1]] = self.objects.tube.id
                else:
                    self.grid[old_pos[0], old_pos[1]] = 0

        assert action <= 3
        old_pos = tuple(self.agent_pos)
        next_x, next_y = self.agent_pos[0] + self.actions[action][0], self.agent_pos[1] + self.actions[action][1]
        # update grid
        early_stop_done = False
        # check if the agent will hit into walls, then make no change
        if self.grid[next_x, next_y] == self.objects.wall.id:
            pass
        elif self.grid[next_x, next_y] == self.objects.ladder.id:
            self.visited_ladder = True
            _go_to(next_x, next_y)
        elif self.grid[next_x, next_y] == self.objects.tube.id:
            # we can only go down using the tube
            if next_x > self.agent_pos[0]:
                self.visited_tube = True
                _go_to(next_x, next_y)
        elif self.grid[next_x, next_y] == self.objects.hidden_key.id:
            self.picked_hidden_key = True
            _go_to(next_x, next_y)
        elif self.grid[next_x, next_y] == self.objects.key.id:
            self.picked_key = True
            _go_to(next_x, next_y)
        elif self.grid[next_x, next_y] == self.objects.door.id:
            if self.picked_key and self.picked_hidden_key:
                self.door_opened = True
                _go_to(next_x, next_y)
            elif self.hard_exploration:
                early_stop_done = True
                _go_to(next_x, next_y)
        elif self.grid[next_x, next_y] == self.objects.worn_ladder.id:
            # hard exploration: not early stop, stay at the same place; otherwise terminate directly
            if not self.hard_exploration:
                _go_to(next_x, next_y)
                self.grid[next_x, next_y] = self.objects.dead_agent.id
                early_stop_done = True
                self.agent_dead = True
        elif self.grid[next_x, next_y] == 0:
            if not (old_pos in self.tube_locations and action == 0):
                # edge case: the agent went one step down on the ladder and try going back to the top
                if old_pos in self.ladder_locations and action == 0 and self.grid[old_pos[0]+1, old_pos[1]] == self.objects.ladder.id:
                    early_stop_done = True
                elif old_pos in self.ladder_locations and action == 1 and self.grid[old_pos[0]-1, old_pos[1]] == self.objects.ladder.id:
                    early_stop_done = True
                _go_to(next_x, next_y)
        # update flag
        self.visited_bottom = self.visited_bottom or self.agent_pos[0] == self.height - 2
        self.back_to_upper = self.back_to_upper or (self.agent_pos[0] == 1 and old_pos[0] != 1)
        self.at_ladder = not self.agent_dead and tuple(self.agent_pos) in self.ladder_locations
        self.at_tube = tuple(self.agent_pos) in self.tube_locations
        # update info
        self.info['at_ladder'] = self.at_ladder
        self.info['visited_ladder'] = self.visited_ladder
        self.info['at_tube'] = self.at_tube
        self.info['visited_tube'] = self.visited_tube
        self.info['picked_hidden_key'] = self.picked_hidden_key
        self.info['picked_key'] = self.picked_key
        self.info['visited_bottom'] = self.visited_bottom
        self.info['back_to_upper'] = self.back_to_upper
        self.info['door_opened'] = self.door_opened
        # noinspection PyTypeChecker
        self.info['next_tuple_state'] = tuple(self._get_grid_obs().tolist())
        if self.info_img:
            self.info['next_img_state'] = np.copy(self._get_img_obs())
        done = early_stop_done or self.door_opened
        reward = self.success_reward if self.door_opened else 0
        return self._obs(), float(reward), done, addict.Dict(self.info)

    def reset(self, **kwargs):
        self._init()
        return self._obs()


def main():
    print('starting mario')
    import cv2
    env = Env_Mario(use_state=True, info_img=True)
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
        print(reward, done)


if __name__ == "__main__":
    main()
