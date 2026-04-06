import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_torcs import TorcsEnv

run_name = "corkscrew_steer_v2"

def make_env(target_speed=30.0):
    def _init():
        env = TorcsEnv(
            vision=False,
            throttle=False,
            gear_change=False,
            target_speed=target_speed,
            max_steps=1500,
        )
        env = Monitor(
            env,
            filename=f"./logs/{run_name}/monitor.csv",
            info_keywords=(
                "terminal_reason",
                "lap_completed",
                "lap_time",
                "max_dist_from_start",
                "ep_progress_reward",
                "ep_align_reward",
                "ep_center_penalty",
                "ep_lateral_penalty",
                "ep_steer_change_penalty",
                "ep_steer_abs_penalty",
                "ep_terminal_penalty",
            ),
        )
        return env
    return _init

class LiveInfoCallback(BaseCallback):
    STEP_KEYS = (
        "progress",
        "forward",
        "track_pos",
        "cos_angle",
        "safe_progress",
        "align_factor",
        "center_factor",
        "clearance_factor",
        "max_dist_from_start",
    )

    EP_KEYS = (
        "lap_completed",
        "lap_time",
        "max_dist_from_start",
        "ep_progress_reward",
        "ep_align_reward",
        "ep_center_penalty",
        "ep_lateral_penalty",
        "ep_steer_change_penalty",
        "ep_steer_abs_penalty",
        "ep_terminal_penalty",
    )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        if not infos:
            return True

        info = infos[0]  # masz jedno środowisko
        for key in self.STEP_KEYS:
            if key in info:
                self.logger.record(f"env_step/{key}", float(info[key]))

        if len(dones) > 0 and bool(dones[0]):
            for key in self.EP_KEYS:
                if key in info:
                    self.logger.record(f"env_ep/{key}", float(info[key]))

        if self.n_calls % 100 == 0:
            self.logger.dump(self.num_timesteps)

        return True


class TerminationStatsCallback(BaseCallback):
    def __init__(self, print_freq=10, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.counts = {
            "off_track": 0,
            "backward": 0,
            "low_progress": 0,
            "time_limit": 0,
            "lap_completed": 0,
            "other": 0,
            "server_shutdown": 0,
        }
        self.episodes = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is None or infos is None:
            return True

        for done, info in zip(dones, infos):
            if done:
                reason = info.get("terminal_reason", "other")
                if reason not in self.counts:
                    reason = "other"

                self.counts[reason] += 1
                self.episodes += 1
                total = max(self.episodes, 1)

                for key in self.counts:
                    self.logger.record(f"termination/{key}_rate", self.counts[key] / total)

                if self.episodes % self.print_freq == 0:
                    print(
                        f"[termination] ep={self.episodes} | "
                        + " | ".join(f"{k}={v}" for k, v in self.counts.items())
                    )
        return True   

def main():
    os.makedirs(f"./models/{run_name}", exist_ok=True)
    os.makedirs(f"./logs/{run_name}", exist_ok=True)
    os.makedirs(f"./tensorboard_logs/{run_name}", exist_ok=True)

    env = DummyVecEnv([make_env(target_speed=30.0)])

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f"./tensorboard_logs/{run_name}/",
        learning_rate=1e-4,
        buffer_size=300_000,
        batch_size=256,
        ent_coef="auto",
        learning_starts=10_000,
        gamma=0.99,
        tau=0.005,
        train_freq=(1, "step"),
        gradient_steps=1,
        stats_window_size=20,
        policy_kwargs=dict(net_arch=[256, 256]),
        device="auto",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=f"./models/{run_name}/",
        name_prefix=run_name,
    )

    callback = CallbackList([
        checkpoint_callback,
        TerminationStatsCallback(print_freq=10),
        LiveInfoCallback(),
    ])

    model.learn(
        total_timesteps=500_000,
        callback=callback,
        reset_num_timesteps=True,
        tb_log_name=run_name,
    )

    model.save(f"./models/{run_name}/{run_name}_final")
    print("Trening zakończony pomyślnie")


if __name__ == "__main__":
    main()