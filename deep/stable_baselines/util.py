from typing import TypedDict
import numpy as np

class TrainSettings(TypedDict):
    name: str
    expname: str
    total_timesteps: int
    log_interval: int  # log avg reward in the interval (in num timesteps)
    checkpoint_freq: int
    eval_freq: int
    evaluation_n: int  # n of episodes to simulate in evaluation phase
    n_envs: int


def get_basic_train_settings(name: str) -> TrainSettings:
    basic_train_setting: TrainSettings = {
        "name": name,
        "expname": "",
        "total_timesteps": int(5e5),
        "log_interval": int(1e3),
        "checkpoint_freq": int(1e5),
        "eval_freq": int(1e5),
        "evaluation_n": int(250),
        "n_envs": 1,
    }
    return basic_train_setting


class ModelSettings(TypedDict):
    policy: str
    learning_rate: float
    seed: float
    kwargs: dict  # network-specific hyperparams


class LearningRateDecay:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __repr__(self):
        return f"LearningRateDecay:<{self.start}>-><{self.end}>"

    def __call__(self, progress_remainig: float) -> float:
        '''
        progress_remaning starts 1 to 0
        '''
        
        progress = 1-progress_remainig
        rate = self.end / self.start

        # if progress < 0.2:
        #     return self.start * (5 * progress)

        # progress = (progress - 0.2) * 1.25
        return self.start * (rate**progress)

class CosineAnnealingDecay:
    def __init__(self, start:float, eta_min:float = 0.0, T_max:float = 1.0):
        self.start = start
        self.T_max = T_max
        self.eta_min = eta_min

    def __repr__(self):
        return f"CosineAnnealingLRDecay:<{self.start}>-><{self.eta_min}>"

    def __call__(self, progress_remainig: float) -> float:
        '''
        progress_remaning starts 1 to 0
        '''
        return self.eta_min + (self.start - self.eta_min) * (1+np.cos((1-progress_remainig)*np.pi)) / 2