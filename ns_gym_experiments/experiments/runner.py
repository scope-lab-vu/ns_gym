import subprocess
import os

runner_dir = os.path.dirname(os.path.abspath(__file__))


experiments =["NS_FrozenLake/PAMCTS_single_discrete_transition_without_change_notification.py",
        "NS_FrozenLake/PAMCTS_single_discrete_transition_with_change_notification.py"] 


for exp in experiments:
    try:
        print(f"Starting exp. {exp}")
        res = subprocess.run(["python", os.path.join(runner_dir, exp)])
        print(res)
    except subprocess.CalledProcessError as e:
        print(f"Error in running {exp}: {e}")
