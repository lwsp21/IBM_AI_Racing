import os
from stable_baselines3 import SAC
from gym_torcs import TorcsEnv

def main():
    env = TorcsEnv(vision=False, throttle=False, gear_change=False)

    model_filename = "torcs_sac_480000_steps.zip"
    model_path = os.path.join(".", "models", model_filename)
    
    print(f"Wczytuje model z: {model_path}")
    try:
        model = SAC.load(model_path)
        print("Model wczytany")
    except Exception as e:
        print(f"Błąd wczytywania")
        print(e)
        return

    obs, info = env.reset()
    
    print("Start")
    
    while True:
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Koniec epizodu Powód: {info.get('terminal_reason')}")
            obs, info = env.reset()

if __name__ == "__main__":
    main()