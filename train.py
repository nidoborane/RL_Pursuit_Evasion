import os
import gymnasium as gym
import envs
import yaml
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

curriculum = {
    "ShortDistance" : ["SteepTurn", "ShortDistance"],
    "SmallAzimuth" : ["SmallAzimuth"],
    "LargeAzimuth" : ["LargeAzimuth"]
    }

curriculum_steps = {
   "SteepTurn" : 30000000,
   "ShortDistance" : 40000000,
   "SmallAzimuth" : 50000000,
   "LargeAzimuth" : 42000000
}

def load_config(maneuver):
    config_path = Path(f"config/{maneuver}_train_config.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config

def make_env(maneuver, n_envs=8, seed=None):
    config = load_config(maneuver)

    return make_vec_env(
        f"AirMissileEnv-{maneuver}-v0", 
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs = {"config" : config}
    )

def train_model(maneuver):
    stages = curriculum[maneuver]
    n_envs = 8
    save_freq = 1000000 // n_envs

    env = make_env(stages[0], n_envs)

    model = PPO(
        "MultiInputPolicy",
        env,
        gamma=0.99,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=1024,
        verbose=1,
        tensorboard_log=f"./tensorboard_logs/{maneuver}/",
        device="auto"
    )

    for stage in stages:
        timesteps = curriculum_steps[stage]

        new_env = make_env(stage, n_envs, seed=None)

        path = f"./ppo_{stage.lower()}_logs/"

        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=path,
            name_prefix="timestep",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )

        os.makedirs(path, exist_ok=True)
        checkpoints = [
                f for f in os.listdir(path)
                if f.startswith("timestep") and f.endswith(".zip")
        ]

        if os.path.exists(path + "final_model.zip"):
            print(f"\n===========Stage {stage} already finished===========\n")
        
        else:
            if checkpoints:
                checkpoints.sort(
                    key=lambda x: int(x.split("_")[1])
                )
                latest_checkpoint = checkpoints[-1]

                model = PPO.load(path + latest_checkpoint, env=new_env)

                print(f"\n===========Resuming Training on stage {stage}===========\n")

                env.close()
                env = new_env
                remaining = max(timesteps - model.num_timesteps, 0)

                model.learn(
                    total_timesteps=remaining, 
                    callback=checkpoint_callback,
                    tb_log_name=stage,
                    reset_num_timesteps=False
                )

            else:
                print(f"\n===========Training on stage {stage}===========\n")

                model.set_env(new_env)

                env.close()
                env = new_env

                model.learn(
                    total_timesteps=timesteps, 
                    callback=checkpoint_callback,
                    tb_log_name=stage,
                    reset_num_timesteps=True
                )

            model.save(path + "final_model")

    print(f"\n===========Finished training for {maneuver}===========n")
    env.close()

if __name__ == "__main__":
    for maneuver in curriculum:
        train_model(maneuver)