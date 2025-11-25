import argparse
import os
import subprocess
import sys

from rich.console import Console

from nerfstudio.utils.misc import convert_ply_to_splat

CONSOLE = Console(width=120)
NERFSTUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXECUTABLE = sys.executable


def parse_args():
    parser = argparse.ArgumentParser(description="Process 3D reconstruction pipeline.")
    parser.add_argument(
        "--source_dir", "-s", type=str, required=True, help="Base directory containing data,workspace dir"
    )
    parser.add_argument("--image_path", type=str, default="sparse/0/images")
    parser.add_argument("--model_id", type=str, required=True, help="Dataset ID")
    parser.add_argument("--model_path", type=str, required=True, help="directory for output data")
    parser.add_argument("--save_iteration_output_dir", type=str, default="")
    parser.add_argument("--save_iteration", type=int, default=3000)
    parser.add_argument("--target_points", type=int, default=1000000, help="Target number of points for downsampling")
    parser.add_argument("--init_voxel_size", type=float, default=0.0005, help="Initial voxel size for downsampling")
    return parser.parse_args()


def main():
    args = parse_args()

    dataws_dir = os.path.join(args.source_dir, args.model_id)
    base_dir = os.path.dirname(args.source_dir)  # 回到上一级目录
    data_dir = args.source_dir
    # 路径设置
    image_path = args.image_path
    output_dir = args.model_path
    progress_path = os.path.join(output_dir, "progress_log")
    input_ply = os.path.join(base_dir, "outputs/dense_pc.ply")
    output_ply = os.path.join(base_dir, "outputs/downsampled_output.ply")
    load_config_path = os.path.join(output_dir, "config.yml")
    export_output_dir = os.path.join(args.model_path, "exports")
    crop_reference_ply = output_ply
    crop_input_ply = os.path.join(export_output_dir, "splat.ply")
    crop_output_ply = os.path.join(output_dir, "point_cloud/iteration_30000", "point_cloud.ply")
    splat_output_ply = os.path.join(output_dir, "point_cloud/iteration_30000", "splat.splat")
    
    save_iteration_output_dir = args.save_iteration_output_dir
    save_iteration = args.save_iteration
    # 将激光雷达的点降采样到100w左右
    cmd_preprocess_ply = [
        PYTHON_EXECUTABLE,
        os.path.join(NERFSTUDIO_DIR, "preprocess_ply.py"),
        "--input_file",
        input_ply,
        "--output_file",
        output_ply,
        "--target_points",
        str(args.target_points),
        "--init_voxel_size",
        str(args.init_voxel_size),
        "--sor_enable",
    ]
    try:
        subprocess.run(cmd_preprocess_ply, cwd=NERFSTUDIO_DIR, check=True)
        CONSOLE.print("[green] Downsampling completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        exit(1)
    # 训练
    cmd_train = [
        PYTHON_EXECUTABLE,
        "nerfstudio/scripts/train.py",
        "splatfacto-big",
        "--data",
        data_dir,
        "--output-dir",
        output_dir,
        "--save_only_latest_checkpoint", "False",
        "--save_iteration", str(save_iteration),
        "--save_iteration_output_dir", save_iteration_output_dir,
        "--max-num-iterations", "30000",
        "--pipeline.model.num-downscales=0",
        "--pipeline.model.sh-degree=0",
        "--pipeline.datamanager.cache-images=cpu",
        "--pipeline.model.use_bilateral_grid",
        "True",
        "--pipeline.model.strategy",
        "mcmc",
        "--pipeline.model.max_gs_num",
        "2000000",
        "--pipeline.model.warmup_length",
        "5000",
        "--pipeline.model.refine-every",
        "500",
        "--pipeline.model.stop-split-at",
        "25000",
        "--pipeline.model.progress_path",
        progress_path,
        "--vis",
        "wandb",
        "colmap",
        "--images_path",
        image_path,
        "--colmap_path",
        "sparse/0",
        "--orientation-method",
        "none",
        "--center-method",
        "none",
        "--auto-scale-poses",
        "False",
        "--assume-colmap-world-coordinate-convention",
        "False",
        "--load_ply_path",
        output_ply,
        "--json_cams_path",
        output_dir,
        "--load_combined_ply",
        "True",
    ]
    try:
        subprocess.run(cmd_train, cwd=NERFSTUDIO_DIR, check=True)
        CONSOLE.print("[green] Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        exit(1)
    # 导出
    cmd_export = [
        # PYTHON_EXECUTABLE ,"-m","ns-export", "gaussian-splat",
        PYTHON_EXECUTABLE,
        "nerfstudio/scripts/exporter.py",
        "gaussian-splat",
        "--load-config",
        load_config_path,
        "--output-dir",
        export_output_dir,
    ]
    try:
        subprocess.run(cmd_export, cwd=NERFSTUDIO_DIR, check=True)
        CONSOLE.print("[green] Export completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        exit(1)
    # 裁剪
    cmd_crop = [
        PYTHON_EXECUTABLE,
        "crop_ply_aabb.py",
        "--reference_ply",
        crop_reference_ply,
        "--input_ply",
        crop_input_ply,
        "--output_ply",
        crop_output_ply,
        "--scale_factor",
        "10",
    ]
    try:
        subprocess.run(cmd_crop, cwd=NERFSTUDIO_DIR, check=True)
        CONSOLE.print("[green] Cropping completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        exit(1)

    convert_ply_to_splat(crop_output_ply, splat_output_ply)


if __name__ == "__main__":
    main()
