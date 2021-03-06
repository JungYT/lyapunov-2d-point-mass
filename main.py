import numpy as np
import numpy.random as random
import ray
from ray import tune
from ray.rllib.agents import ddpg
from ray.rllib.agents import ppo
import gym
from pathlib import Path

import fym
from fym.core import BaseEnv, BaseSystem
from postProcessing import plot_TwoDimTwoPointMass
from dynamics import Env, compute_init


def train():
    cfg = fym.config.load(as_dict=True)
    analysis = tune.run(ppo.PPOTrainer, **cfg)
    parent_path = Path(analysis.get_last_checkpoint(
        metric="episode_reward_mean",
        mode="max"
    )).parent.parent
    checkpoint_paths = analysis.get_trial_checkpoints_paths(
        trial=str(parent_path)
    )
    return checkpoint_paths


@ray.remote(num_cpus=5)
def sim(initial, checkpoint_path, env_config, num=0):
    env = Env(env_config)
    agent = ppo.PPOTrainer(env=Env, config={"explore": False})

    agent.restore(checkpoint_path)
    parent_path = Path(checkpoint_path).parent
    data_path = Path(parent_path, f"test_{num+1}", "env_data.h5")
    plot_path = Path(parent_path, f"test_{num+1}")
    env.logger = fym.Logger(data_path)

    obs = env.reset(initial)
    while True:
        action = agent.compute_single_action(obs)
        obs, _, done, _ = env.step(action)
        if done:
            break
    env.close()
    plot_TwoDimTwoPointMass(plot_path, data_path)


def validate(parent_path):
    _, info = fym.logging.load(
        Path(parent_path, 'checkpoint_paths.h5'),
        with_info=True
    )
    checkpoint_paths = info['checkpoint_paths']
    initials = compute_init()
    fym.config.update({"config.env_config.max_t": 20})
    env_config = ray.put(fym.config.load("config.env_config", as_dict=True))
    print("Validating...")
    futures = [sim.remote(initial, path[0], env_config, num=i)
               for i, initial in enumerate(initials)
               for path in checkpoint_paths]
    ray.get(futures)

def plot_data(parent_path_list):
    for parent_path in parent_path_list:
        _, info = fym.logging.load(
            Path(parent_path, 'checkpoint_paths.h5'),
            with_info=True
        )
        checkpoint_paths = info['checkpoint_paths']
        # for checkpoint_data in checkpoint_paths:
        checkpoint_path = Path(checkpoint_paths[-1][0]).parent
        test_path_list = [x for x in checkpoint_path.iterdir() if x.is_dir()]
        # for i in range(len(test_path_list)):
        data_path = Path(test_path_list[-1], "env_data.h5")
        plot_path = Path(test_path_list[-1])
        print("Ploting", str(plot_path))
        plot_rllib_test(plot_path, data_path)


def config():
    fym.config.reset()
    fym.config.update({
        "config": {
            "env": Env,
            "env_config": {
                "dt": 0.01,
                "max_t": 10.,
                "solver": "rk4"
            },
            "num_gpus": 0,
            "num_workers": 4,
            # "num_envs_per_worker": 50,
            "lr": 0.001,
            "gamma": 0.9,
            # "lr": tune.grid_search([0.001, 0.0005, 0.0001]),
            # "gamma": tune.grid_search([0.9, 0.99, 0.999])
            # "actor_lr": tune.grid_search([0.001, 0.003, 0.0001]),
            # "critic_lr": tune.grid_search([0.001, 0.003, 0.0001]),
            # "actor_lr": 0.001,
            # "critic_lr": 0.0001,
            # "gamma": tune.grid_search([0.9, 0.99, 0.999, 0.9999]),
            # "exploration_config": {
            #     "random_timesteps": 10000,
            #     "scale_timesteps": 100000,
            # },
        },
        "stop": {
            "training_iteration": 2,
        },
        "local_dir": "./ray_results",
        "checkpoint_freq": 1,
        "checkpoint_at_end": True,
    })


def main():
    checkpoint_paths = train()
    parent_path = "/".join(checkpoint_paths[0][0].split('/')[0:-3])
    checkpoint_logger = fym.logging.Logger(
        Path(parent_path, 'checkpoint_paths.h5')
    )
    checkpoint_logger.set_info(checkpoint_paths=checkpoint_paths)
    checkpoint_logger.set_info(config=fym.config.load(as_dict=True))
    checkpoint_logger.close()
    breakpoint()
    return parent_path

def debug():
    config()
    cfg = fym.config.load(as_dict=True)
    env = Env(cfg['config']['env_config'])
    
    obs = env.reset()
    while True:
        action = np.array([0.3, 0.1])
        obs, reward, done, info = env.step(action)
        if done:
            break
    env.close()

# def plot_debug(parent_path):
#     _, info = fym.logging.load(
#         Path(parent_path, 'checkpoint_paths.h5'),
#         with_info=True
#     )
#     checkpoint_path = info['checkpoint_paths'][0][0]
#     parent_path = Path(checkpoint_path).parent
#     data_path = Path(parent_path, f"test_{1}", "env_data.h5")
#     plot_path = Path(parent_path, f"test_{1}")
#     plot_validation(plot_path, data_path)

if __name__ == "__main__":
    config()
    # debug()

    ray.shutdown()
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    ## To train, validate, and make figure
    parent_path = main()
    ## To validate and make figure
    # parent_path = './ray_results/PPO_2022-02-14_13-57-59'
    ## plot debugging
    # plot_debug(parent_path)

    validate(parent_path)
    ray.shutdown()

