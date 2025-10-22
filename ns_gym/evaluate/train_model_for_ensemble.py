import gymnasium as gym
import stable_baselines3 as sb3
import os
from pathlib import Path

# --- Configuration ---
ENV_NAME = "FrozenLake-v1"
BASE_DIR = Path(__file__).parent
SAVE_DIR = BASE_DIR / "evaluation_model_weights" / "FrozenLakeV1"


MODELS_TO_TRAIN = {
    "PPO": sb3.PPO,
    "A2C": sb3.A2C,
    "DQN": sb3.DQN,
}
TOTAL_TIMESTEPS = 100000


if __name__ == "__main__":
    # Create the base Gymnasium environment
    env = gym.make(ENV_NAME)

    print(f"--- Starting training for {ENV_NAME} ---")

    for model_name, model_class in MODELS_TO_TRAIN.items():
        print(f"\nTraining {model_name}...")

        # Create the specific directory for the agent
        agent_save_dir = SAVE_DIR / model_name
        os.makedirs(agent_save_dir, exist_ok=True)
        save_path = agent_save_dir / "model.zip"

        # Instantiate the model
        # For discrete environments like FrozenLake, "MlpPolicy" is a good default
        model = model_class("MlpPolicy", env, verbose=0)

        # Train the model
        model.learn(total_timesteps=TOTAL_TIMESTEPS)

        # Save the trained model
        model.save(save_path)

        print(f"✅ Saved trained {model_name} model to {save_path}")

    env.close()
    print("\n--- All training complete. ---")
