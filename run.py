import subprocess
import os
from rich.console import Console
CONSOLE = Console(width=120)

nerfstudio_python = "/home/farsee/miniconda3/envs/nerfstudio/bin/python"

# 定义多个数据目录
base_dir = "/home/farsee/dev/3dMass/upload/1743413530732.816/"

# name_id = [dir_id for dir_id in os.listdir(base_dir)]

# name_id = ['1743413530732.823']
# name_id = ['1743413530732.818', '1743413530732.823', '1743413530732.821', '1743413530732.822', '1743413530732.820', '1743413530732.819']
# name_id = ['1743413530732.820']
name_id = ['7751','144']
# name_id = ['1743413530732.822', '1743413530732.820', '1743413530732.817', '1743413530732.819']

# 遍历数据目录并执行训练
for id in name_id:
    CONSOLE.print("[green]Running model id {}".format(id))
    # 设置参数
    dataws_dir = os.path.join(base_dir,id)
    model_path = os.path.join(dataws_dir,"output_models_sfm")
    save_iteration_output_dir = os.path.join(dataws_dir, "fastGS","output_models")
    # 构造命令
    cmd_run_nerfstudio = [
        nerfstudio_python, "run_nerfstudio_1.py",
        "-s", base_dir,
        "--model_id",id,
        "--model_path",model_path,
        "--save_iteration_output_dir", save_iteration_output_dir
        # "--image_path","/home/farsee/dev/3dMass/upload/1744090676990.9229/289/data/"
    ]

    # 运行 subprocess
    try:
        subprocess.run(cmd_run_nerfstudio, check=True)
        CONSOLE.print("[green]Successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        exit(1)





