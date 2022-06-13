from datetime import datetime
import os
import sys
import shutil
import pickle

import imageio
import torch
import wandb
import json


class Wandb_Logger:
    def __init__(self, proj_name, run_name):
        wandb.init(project=proj_name, name=run_name)

    def log(self, log_dict, prefix='', step=None):
        log_info = log_dict
        if prefix != '':
            log_info = dict()
            for k in log_dict:
                log_info[prefix + '/' + k] = log_dict[k]
        if step is not None:
            wandb.log(log_info, step=step)
        else:
            wandb.log(log_info)


DEFAULT_PROJECT_NAME = 'default-project'        # default project name
DEFAULT_IS_USE_WANDB = False        # whether to use wandb for logging
DEFAULT_SAVE_TRAJ = False  # whether to save transitions/trajectories


class ExperimentManager:
    def __init__(self, saved_dir, expr_name, is_add_time_to_name=True):
        # add timestamp to expr name
        if is_add_time_to_name:
            now = datetime.now()
            d = now.strftime('%Y%m%d_%H%M%S')
            expr_name = expr_name + '_' + d
        self.expr_name = expr_name
        self.project_name = DEFAULT_PROJECT_NAME
        self.is_use_wandb = DEFAULT_IS_USE_WANDB

        # create expr dir
        if not os.path.exists(saved_dir):
            os.mkdir(saved_dir)
        self.saved_dir = os.path.join(saved_dir, expr_name)
        os.mkdir(self.saved_dir)

        # create image saved dir
        self.img_dir = os.path.join(self.saved_dir, 'images')
        os.mkdir(self.img_dir)

        # create snapshot saved dir
        self.snapshot_dir = os.path.join(self.saved_dir, 'snapshots')
        os.mkdir(self.snapshot_dir)

        # trajectories holder
        self.trajectories = list()    # traj with transition information
        self.current_trajectory = list()

        # create log file
        self.logfile = open(os.path.join(self.saved_dir, 'log.txt'), 'a')

    def add_trajectories(self, trajs):
        """ Save the basic trajectories """
        for traj in trajs:
            for t in range(len(traj)):
                curr_state = traj[t][0]
                action = traj[t][1]
                reward = traj[t][2]
                next_state = traj[t][3]
                done = traj[t][4]
                info = traj[t][5]
                self.add_transition(curr_state, action, reward, next_state, done, info)

    def add_transition(self, curr_state, action, reward, next_state, done, info):
        """
        Save the transition
        """
        if not DEFAULT_SAVE_TRAJ:
            return
        self.current_trajectory.append((curr_state, action, reward, next_state, done, info))
        if done:
            self.trajectories.append(self.current_trajectory)
            self.current_trajectory = list()

    def save_models(self, checkpoint, postfix=None, is_snapshot=True, prefix='model'):
        """
        Save current model
        :param checkpoint: the parameters of the models, see example in pytorch's documentation: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        :param is_snapshot: whether saved in the snapshot directory
        :param prefix: the prefix of the file name
        :param postfix: the postfix of the file name (can be episode number, frame number and so on)
        """
        saved_dir = self.snapshot_dir if is_snapshot else self.saved_dir

        if postfix is not None:
            fname = get_unique_fname(saved_dir, prefix + '_' + postfix + '.tar')
        else:
            fname = get_unique_fname(saved_dir, prefix + '_' + self.expr_name + '.tar')
        torch.save(checkpoint, os.path.join(saved_dir, fname))

    def save_trajectories_snapshot(self):
        """
        save trajectories to file
        is_separated_file: whether to save each trajectory to its own file
        """
        fname = 'traj_' + self.expr_name + '.pickle'
        fname = get_unique_fname(self.snapshot_dir, fname)
        fname = os.path.join(self.snapshot_dir, fname)
        with open(fname, 'wb') as f_traj:
            pickle.dump(self.trajectories, f_traj, protocol=pickle.HIGHEST_PROTOCOL)

    def dump_data(self, data, file_name):
        fname = os.path.join(self.saved_dir, file_name)
        with open(fname, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    def dump_dict_json(self, data, file_name, indent=4):
        """
        save dict data as json
        """
        fname = os.path.join(self.saved_dir, file_name + '.json')
        with open(fname, 'w') as json_file:
            json.dump(data, json_file, indent=indent)

    def dump_rgb_image(self, fname, img, img_format='png', step=None, episode=None):
        if step is None:
            ts = str(datetime.now().timestamp()).replace('.', '_')
            step = ts
            episode = 'nan'
        fname = fname + '_' + str(episode) + '_' + str(step) + '.' + img_format
        fname = os.path.join(self.img_dir, fname)
        img.save(fname)

    def dump_video(self, fname, images, fps=6):
        fname = fname + '.mp4'
        fname = os.path.join(self.saved_dir, fname)
        imageio.mimwrite(fname, images, fps=fps)

    def make_copy_file(self, source):
        # make a copy of the source file
        config_fname = os.path.basename(source)
        saved_fname = os.path.join(self.saved_dir, config_fname)
        shutil.copyfile(source, saved_fname)

    def redirect_output_to_logfile_as_well(self):
        class Logger(object):
            def __init__(self, logfile):
                self.stdout = sys.stdout
                self.logfile = logfile

            def write(self, message):
                self.stdout.write(message)
                self.logfile.write(message)

            def flush(self):
                # this flush method is needed for python 3 compatibility.
                # this handles the flush command by doing nothing.
                # you might want to specify some extra behavior here.
                pass
        sys.stdout = Logger(self.logfile)
        sys.stderr = sys.stdout

    def use_wandb(self, is_use):
        self.is_use_wandb = is_use

    def set_wandb_project_name(self, name):
        self.project_name = name

    def init_wandb(self):
        if self.is_use_wandb:
            wandb.init(project=self.project_name, name=self.expr_name)

    def save_config_wandb(self, config):
        """
        The config is assumed to be a nested dict, e.g. {"hyper param": {"learning rate": 0.1}}
        """
        wandb_config = {}
        for key0 in config:
            for key1 in config[key0]:
                param_name = str(key0) + '_' + str(key1)
                param_value = config[key0][key1]
                wandb_config[param_name] = str(param_value)
        if self.is_use_wandb:
            wandb.config.update(wandb_config)

    def log_wandb(self, log_dict, step=None):
        if self.is_use_wandb:
            if step is not None:
                wandb.log(log_dict, step=step)
            else:
                wandb.log(log_dict)

    def watch_wandb(self, model_list):
        if self.is_use_wandb:
            wandb.watch(model_list, log='parameters')

    def open_virtual_display(self):
        from pyvirtualdisplay import Display
        display = Display(visible=0, size=(1080, 608))
        display.start()


def get_unique_fname(file_dir, fname_base):
    name, extension = os.path.splitext(fname_base)
    post_fix = 0
    while True:
        fname = name + '_' + str(post_fix) + extension
        if not os.path.exists(os.path.join(file_dir, fname)):
            return fname
        post_fix += 1
