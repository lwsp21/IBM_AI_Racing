import gymnasium as gym
from gymnasium import spaces
import numpy as np
import snakeoil3_gym as snakeoil3
import collections as col
import os
import time
import pyautogui
import pathlib

class TorcsEnv(gym.Env):
    def __init__(self, vision=False, throttle=False, gear_change=False,
                 target_speed=30.0, max_steps=1500):
        super().__init__()

        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.target_speed = target_speed
        self.max_steps = max_steps
  
        self._force_relaunch_next_reset = False
        self.client = None
        self.last_steer = 0.0
        self.prev_dist_raced = 0.0
        self.max_dist_from_start = 0.0
        self.prev_dist_from_start = 0.0
        self.progress_window = col.deque(maxlen=50)
        self.milestones_hit = set()

        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        if vision is False:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28 + 64 * 64 * 3,), dtype=np.float32)

        self.observation = None
        self.reset_torcs()
        
    def step(self, u):
        client = getattr(self, "client", None)
        if client is None or getattr(client, "so", None) is None:
            self._force_relaunch_next_reset = True
            return self.get_obs(), 0.0, True, False, {"terminal_reason": "server_shutdown"}

        client = self.client
        this_action = self.agent_to_torcs(u)
        action_torcs = client.R.d

        # Steering
        action_torcs["steer"] = this_action["steer"]  # in [-1, 1]

        # Simple Automatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.target_speed
            speed = client.S.d["speedX"]

            if speed < target_speed - 3:
                action_torcs["accel"] = 0.25
                action_torcs["brake"] = 0.0
            elif speed > target_speed + 3:
                action_torcs["accel"] = 0.0
                action_torcs["brake"] = 0.15
            else:
                action_torcs["accel"] = 0.05
                action_torcs["brake"] = 0.0

        # Automatic Gear Change
        if self.gear_change is True:
            action_torcs["gear"] = this_action["gear"]
        else:
            action_torcs["gear"] = 1
            if client.S.d["speedX"] > 50: action_torcs["gear"] = 2
            if client.S.d["speedX"] > 80: action_torcs["gear"] = 3
            if client.S.d["speedX"] > 110: action_torcs["gear"] = 4
            if client.S.d["speedX"] > 140: action_torcs["gear"] = 5
            if client.S.d["speedX"] > 170: action_torcs["gear"] = 6


        client.respond_to_server()
        client.get_servers_input()

        if getattr(client, "so", None) is None:
            self._force_relaunch_next_reset = True
            return self.get_obs(), 0.0, True, False, {"terminal_reason": "server_shutdown"}

        obs = client.S.d
        self.observation = self.make_observaton(obs)

        angle = float(obs["angle"])
        speed_x = float(obs["speedX"])
        speed_y = float(obs["speedY"])
        track_pos = float(obs["trackPos"])
        track = np.asarray(obs["track"], dtype=np.float32)
        dist_raced = float(obs["distRaced"])
        current_steer = float(this_action["steer"])
        cos_a = np.cos(angle)

        dist_from_start = float(obs["distFromStart"])
        self.max_dist_from_start = max(self.max_dist_from_start, dist_from_start)

        delta_dist = dist_raced - self.prev_dist_raced
        progress = np.clip(delta_dist, 0.0, 5.0)
        self.prev_dist_raced = dist_raced
        self.progress_window.append(progress)

        prev_dfs = self.prev_dist_from_start
        curr_dfs = dist_from_start
        delta_dfs = curr_dfs - prev_dfs

        if delta_dfs < -1000:   # zabezpieczenie na wrap przy mecie, jeśli kiedyś wystąpi
            delta_dfs = 0.0

        self.prev_dist_from_start = curr_dfs

        '''
        milestone_bonus = 0.0
        for m in (50, 100, 150, 200, 300, 400, 500, 700, 900):
            if dist_from_start >= m and m not in self.milestones_hit:
                self.milestones_hit.add(m)
                milestone_bonus += 2.0
        '''
        # pomocniczo zostawiamy też forward
        forward = speed_x * cos_a

        # płynność sterowania
        steer_change = abs(current_steer - self.last_steer)
        self.last_steer = current_steer

        # reward główny
        progress_reward = 2.0 * progress
        dfs_bonus = 0.1 * max(0.0, delta_dfs)
        align_reward = 0.3 * cos_a
        center_penalty = 0.15 * abs(track_pos)
        lateral_penalty = 0.01 * abs(speed_y)
        steer_change_penalty = 0.01 * steer_change
        steer_abs_penalty = 0.0
        safe_progress = progress
        align_factor = max(0.0, cos_a)
        center_factor = max(0.0, 1.0 - abs(track_pos))
        clearance_factor = 1.0

        reward = (
            progress_reward
            + dfs_bonus
            + align_reward
            - center_penalty
            - lateral_penalty
            - steer_change_penalty
            - steer_abs_penalty
        )

        # Termination judgement
        episode_terminate = False
        terminal_reason = None
        terminal_penalty = 0.0
        truncated = False
        lap_completed = 0.0
        lap_time = 0.0

        min_track = float(track.min())

        if abs(track_pos) > 1.0 or min_track < 0.0:
            terminal_penalty = 50.0 if self.time_step < 100 else 30.0
            reward -= terminal_penalty
            episode_terminate = True
            terminal_reason = "off_track"
            client.R.d["meta"] = True

        if not episode_terminate and cos_a < 0.0:
            terminal_penalty += 5.0
            reward -= terminal_penalty
            episode_terminate = True
            terminal_reason = "backward"
            client.R.d["meta"] = True

        if not episode_terminate and self.time_step > 150:
            if sum(self.progress_window) < 3.0:
                terminal_penalty += 5.0
                reward -= terminal_penalty
                episode_terminate = True
                terminal_reason = "low_progress"
                client.R.d["meta"] = True

        if not episode_terminate and self.time_step >= self.max_steps:
            truncated = True
            episode_terminate = True
            terminal_reason = "time_limit"
            client.R.d["meta"] = True

        if not episode_terminate:
            lap_time = float(obs.get("lastLapTime", 0.0))
            if lap_time > 0.0:
                lap_completed = 1.0
                reward += 100.0
                episode_terminate = True
                terminal_reason = "lap_completed"
                client.R.d["meta"] = True
        
        if client.R.d["meta"]:
            client.respond_to_server()

        self.time_step += 1

        self.ep_progress_reward += progress_reward
        self.ep_align_reward += align_reward
        self.ep_center_penalty += center_penalty
        self.ep_lateral_penalty += lateral_penalty
        self.ep_steer_change_penalty += steer_change_penalty
        self.ep_steer_abs_penalty += steer_abs_penalty
        self.ep_terminal_penalty += terminal_penalty

        info = {
            "terminal_reason": terminal_reason,
            "lap_completed": float(lap_completed),
            "lap_time": float(lap_time),
            "max_dist_from_start": float(self.max_dist_from_start),
            "progress": float(progress),
            "forward": float(forward),
            "track_pos": float(track_pos),
            "cos_angle": float(cos_a),
            "progress_reward": float(progress_reward),
            "align_reward": float(align_reward),
            "center_penalty": float(center_penalty),
            "lateral_penalty": float(lateral_penalty),
            "steer_change_penalty": float(steer_change_penalty),
            "steer_abs_penalty": float(steer_abs_penalty),
            "terminal_penalty": float(terminal_penalty),
            "safe_progress": float(safe_progress),
            "align_factor": float(align_factor),
            "center_factor": float(center_factor),
            "clearance_factor": float(clearance_factor),
        }
        
        if episode_terminate:
            info.update({
                "ep_progress_reward": float(self.ep_progress_reward),
                "ep_align_reward": float(self.ep_align_reward),
                "ep_center_penalty": float(self.ep_center_penalty),
                "ep_lateral_penalty": float(self.ep_lateral_penalty),
                "ep_steer_change_penalty": float(self.ep_steer_change_penalty),
                "ep_steer_abs_penalty": float(self.ep_steer_abs_penalty),
                "ep_terminal_penalty": float(self.ep_terminal_penalty),
            })

        terminated = bool(episode_terminate and not truncated)
        return self.get_obs(), float(reward), terminated, truncated, info

    def close(self):
        if self.client is not None:
            try:
                self.client.shutdown()
            except Exception:
                pass
            self.client = None
        os.system('taskkill /f /im wtorcs.exe >nul 2>&1')

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.time_step = 0
        self.last_steer = 0.0
        self.progress_window.clear()
        self.milestones_hit = set()

        if self.client is not None and getattr(self.client, "so", None) is not None:
            try:
                self.client.R.d["meta"] = True
                self.client.respond_to_server()
            except Exception:
                pass
            self.client.shutdown()
            self.client = None

        if getattr(self, "_force_relaunch_next_reset", False):
            self.reset_torcs()
            self._force_relaunch_next_reset = False

        self.client = snakeoil3.Client(p=3001, vision=self.vision)
        self.client.MAX_STEPS = np.inf
        self.client.get_servers_input()

        obs = self.client.S.d
        self.observation = self.make_observaton(obs)
        self.prev_dist_raced = float(obs["distRaced"])
        self.prev_dist_from_start = float(obs["distFromStart"])
        self.max_dist_from_start = float(obs["distFromStart"])

        self.ep_progress_reward = 0.0
        self.ep_align_reward = 0.0
        self.ep_center_penalty = 0.0
        self.ep_lateral_penalty = 0.0
        self.ep_steer_change_penalty = 0.0
        self.ep_steer_abs_penalty = 0.0
        self.ep_terminal_penalty = 0.0

        return self.get_obs(), {}

    def reset_torcs(self):
        cwd = os.getcwd()
        torcs_dir = pathlib.Path(r"C:\torcs")
        if torcs_dir.exists():
            os.chdir(torcs_dir)

        os.system('taskkill /f /im wtorcs.exe >nul 2>&1')
        time.sleep(1.0)
        
        if self.vision is True:
            os.system('start "" wtorcs.exe -nofuel -nodamage -nolaptime -vision')
        else:
            os.system('start "" wtorcs.exe -nofuel -nodamage -nolaptime')
        
        time.sleep(3.0) 
        for key in ['enter', 'enter', 'up', 'up', 'enter', 'enter']:
            pyautogui.press(key)
            time.sleep(0.2)
        time.sleep(5.0)
        os.chdir(cwd)

    def get_obs(self):
        if self.observation is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    
        obs = self.observation
        parts = []
        for field in obs._fields:
            val = getattr(obs, field)
            if isinstance(val, np.ndarray):
                parts.append(val.ravel())
            else:
                parts.append(np.array([val], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def agent_to_torcs(self, u):
        a = np.asarray(u, dtype=np.float32).ravel()
        idx = 0
        torcs_action = {'steer': float(a[idx])}
        idx += 1

        if self.throttle is True:
            action_raw = float(a[idx])
            if action_raw > 0:
                torcs_action.update({'accel': action_raw, 'brake': 0.0})
            else:
                torcs_action.update({'accel': 0.0, 'brake': abs(action_raw)})
            idx += 1

        if self.gear_change is True:
            gear_raw = float(a[idx])  
            gear = int(np.clip(np.round(((gear_raw + 1.0) / 2.0) * 5.0) + 1, 1, 6))
            torcs_action.update({'gear': gear})

        return torcs_action

    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec = obs_image_vec
        rgb = []
        temp = []
        for i in range(0,12286,3):
            temp.append(image_vec[i])
            temp.append(image_vec[i+1])
            temp.append(image_vec[i+2])
            rgb.append(temp)
            temp = []
        return np.array(rgb, dtype=np.uint8)

    def make_observaton(self, raw_obs):
        if self.vision is False:
            names = ['speedX', 'speedY', 'angle', 'trackPos', 'distFromStart', 'track', 'wheelSpinVel']
            Observation = col.namedtuple('Observation', names)
            return Observation(
                speedX=np.array([raw_obs['speedX']], dtype=np.float32) / 100.0,
                speedY=np.array([raw_obs['speedY']], dtype=np.float32) / 100.0,
                angle=np.array([raw_obs['angle']], dtype=np.float32) / np.pi,  # normalizacja kąta do zakresu [-1, 1]
                trackPos=np.array([raw_obs['trackPos']], dtype=np.float32),
                distFromStart=np.array([raw_obs['distFromStart']], dtype=np.float32) / 1000.0,
                track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32) / 100.0 #normalizacja prędkości obrotowej koła do zakresu [0, 1]
            )
        else:
            names = ['speedX', 'speedY', 'angle', 'trackPos', 'distFromStart', 'track', 'wheelSpinVel', 'img']
            Observation = col.namedtuple('Observation', names)
            image_rgb = self.obs_vision_to_image_rgb(raw_obs['img'])

            return Observation(
                speedX=np.array([raw_obs['speedX']], dtype=np.float32) / 100.0,
                speedY=np.array([raw_obs['speedY']], dtype=np.float32) / 100.0,
                angle=np.array([raw_obs['angle']], dtype=np.float32) / np.pi,  # normalizacja kąta do zakresu [-1, 1]
                trackPos=np.array([raw_obs['trackPos']], dtype=np.float32),
                distFromStart=np.array([raw_obs['distFromStart']], dtype=np.float32) / 1000.0,
                track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32) / 100.0, #normalizacja prędkości obrotowej koła do zakresu [0, 1]
                img=image_rgb
            )

'''
tensorboard --logdir ./tensorboard_logs <- urochamianie tensorboarda 

To są sensory które możemy użyć
sensors= [
        'curLapTime',
        'lastLapTime',
        'stucktimer',
        'damage',
        'focus',
        'fuel',
        'gear',
        'distRaced',
        'distFromStart',
        'racePos',
        'opponents',
        'wheelSpinVel',
        'z',
        'speedZ', - to jest prędkość w osi pionowej na razie raczej niepotrzbne bo auto w góre i dół nie lata
        'speedY',
        'speedX',
        'targetSpeed',
        'rpm', - to na razie usunąłem ale może się przydać jeżeli będziemy chcieli żeby model zmieniał biegi
        'skid',
        'slip',
        'track',
        'trackPos',
        'angle',
        ]
'''