from ns_gym import base
import mujoco
import gymnasium as gym



def generate_params_table(cls_name: str):

    header = "| Parameter | Description | Default Value |\n"
    header += "|-----------|-------------|---------------|"

    rows = [header]

    if cls_name in base.TUNABLE_PARAMS:
        params_dict = base.TUNABLE_PARAMS[cls_name]

        for param_name,default_value in params_dict.items():
            
            row = f"| `{param_name}` |  | {default_value} |"
            rows.append(row)


    return "\n".join(rows)

def generate_params_table_mujoco():

    for env_in in base.SUPPORTED_MUJOCO_ENV_IDS:
        print("Generating table for:", env_in)
        env = gym.make(env_in)
        cls_name = env.unwrapped.__class__.__name__


        if cls_name in base.MUJOCO_GETTERS.keys():
            header = "| Parameter | Description | Default Value |\n"
            header += "|-----------|-------------|---------------|"

            rows = [header]

            getters_dict = base.MUJOCO_GETTERS[cls_name]

            for param_name,fns in getters_dict.items():
                getter_fn = fns[0]
                default_value = getter_fn(env.unwrapped)

                row = f"| `{param_name}` |  | {default_value} |"
                rows.append(row)

            markdown_table = "\n".join(rows)
            print(f"## {cls_name}\n")
            print(markdown_table)
            print("\n\n")



def generate_all_tables():
    for cls_name in base.TUNABLE_PARAMS.keys():
        markdown_table = generate_params_table(cls_name)
        print(f"## {cls_name}\n")
        print(markdown_table)
        print("\n\n")


    for cls_name in base.MUJOCO_GETTERS.keys():

        markdown_table = generate_params_table(cls_name)
        print(f"## {cls_name}\n")
        print(markdown_table)
        print("\n\n")


if __name__ == "__main__":
    markdown_table = generate_params_table( "Continuous_MountainCarEnv")
    print(markdown_table)

    generate_params_table_mujoco()

