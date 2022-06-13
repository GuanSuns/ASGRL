import wandb


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

