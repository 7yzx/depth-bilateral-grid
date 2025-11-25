import subprocess
import os
import shutil
from rich.console import Console
import argparse
import numpy as np
from plyfile import PlyData, PlyElement
import sys
import time  # 用于计算时间

CONSOLE = Console(width=120)
NERFSTUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXECUTABLE = sys.executable

# 修改时间输出格式
def format_time(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes} 分 {remaining_seconds} 秒"

def sh2rgb_splat(sh):
    """
    Converts from 0th order spherical harmonics to rgb [0, 255]
    """
    C0 = 0.28209479177387814
    rgb = [sh[i] * C0 + 0.5 for i in range(len(sh))]
    return np.clip(rgb, 0, 1) * 255

def convert_ply_to_splat(input_ply_filename: str, output_splat_filename: str) -> None:
        """
        Converts a provided .ply file to a .splat file. As part of this all information on
        spherical harmonics is thrown out, so view-dependent effects are lost
        
        Args:
            input_ply_filename: The path to the .ply file that we want to convert to a .splat file
            output_splat_filename: The path where we'd like to save the output .splat file
        Returns:
            None
        """

        plydata = PlyData.read(input_ply_filename)
        with open(output_splat_filename, "wb") as splat_file:
            for i in range(plydata.elements[0].count):
                # Ply file format
                # xyz Position (Float32)
                # nx, ny, nz Normal vector (Float32) (for planes, not relevant for Gaussian Splatting)
                # f_dc_0, f_dc_1, f_dc_2 "Direct current" (Float32) first 3 spherical harmonic coefficients
                # f_rest_0, f_rest_1, ... f_rest_n "Rest" (Float32) of the spherical harmonic coefficients
                # opacity (Float32)
                # scale_0, scale_1, scale_2 Scale (Float32) in the x, y, and z directions
                # rot_0, rot_1, rot_2, rot_3 Rotation (Float32) Quaternion rotation vector
                
                # Splat file format
                # XYZ - Position (Float32)
                # XYZ - Scale (Float32)
                # RGBA - Color (uint8)
                # IJKL - Quaternion rotation (uint8)
                
                plydata_row = plydata.elements[0][i]
                
                # Position
                splat_file.write(plydata_row['x'].tobytes())
                splat_file.write(plydata_row['y'].tobytes())
                splat_file.write(plydata_row['z'].tobytes())
                
                # Scale
                for i in range(3):
                    splat_file.write(np.exp(plydata_row[f'scale_{i}']).tobytes())
                
                # Color
                sh = [plydata_row[f"f_dc_{i}"] for i in range(3)]
                rgb = sh2rgb_splat(sh)
                for color in rgb:
                    splat_file.write(color.astype(np.uint8).tobytes())

                # Opacity
                opac = 1.0 + np.exp(-plydata_row['opacity'])
                opacity = np.clip((1.0/opac) * 255, 0, 255)
                splat_file.write(opacity.astype(np.uint8).tobytes())
                
                # Quaternion rotation
                rot = np.array([plydata_row[f"rot_{i}"] for i in range(4)])
                rot = np.clip(rot * 128 + 128, 0, 255)
                for i in range(4):
                    splat_file.write(rot[i].astype(np.uint8).tobytes())


def parse_args():
    parser = argparse.ArgumentParser(description="Process 3D reconstruction pipeline.")
    parser.add_argument("--source_dir", "-s", type=str, required=True, help="Base directory containing data,workspace dir")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input images")
    parser.add_argument("--model_id", type=str, required=True, help="Dataset ID")
    parser.add_argument("--model_path", type=str, required=True, help="directory for output data")
    parser.add_argument("--iterations", type=int, default=5000, help="Number of training iterations")
    parser.add_argument("--target_points", type=int, default=1000000, help="Target number of points for downsampling")
    parser.add_argument("--init_voxel_size", type=float, default=0.0005, help="Initial voxel size for downsampling")
    return parser.parse_args()

def main():
    args = parse_args()
    # 开始记录总时间
    total_start_time = time.time()
    # 直接使用用户提供的路径
    data_dir = args.source_dir
    image_path = args.image_path
    output_dir = args.model_path
    progress_path = os.path.join(output_dir, "progress_log")
    input_ply = os.path.join(args.source_dir, "outputs/dense_pc.ply")
    output_ply = os.path.join(args.source_dir, "outputs/downsampled_output.ply")
    load_config_path = os.path.join(output_dir, "config.yml")
    export_output_dir = os.path.join(args.model_path, "exports")
    crop_reference_ply = output_ply
    crop_input_ply = os.path.join(export_output_dir, "splat.ply")
    crop_output_ply = os.path.join(output_dir, f"point_cloud/iteration_{args.iterations}", "point_cloud.ply")
    splat_output_ply = os.path.join(output_dir, f"point_cloud/iteration_{args.iterations}", "splat.splat")
    # 将激光雷达的点降采样到100w左右
    cmd_preprocess_ply = [
        PYTHON_EXECUTABLE, "preprocess_ply.py",
        "--input_file", input_ply,
        "--output_file", output_ply,
        "--target_points", str(args.target_points),
        "--init_voxel_size", str(args.init_voxel_size),
        "--sor_enable"
    ]
    try:
        subprocess.run(cmd_preprocess_ply, cwd=NERFSTUDIO_DIR, check=True)
        CONSOLE.print("[green]1 Downsampling completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        exit(1)
    # 训练
    cmd_train = [
        PYTHON_EXECUTABLE, "nerfstudio/scripts/train.py",
        "splatfacto-big",
        "--data", data_dir,
        "--output-dir", output_dir,
        "--experiment-name","splatfacto-quick",
        "--max-num-iterations", str(args.iterations),
        "--pipeline.model.camera-optimizer.mode","SO3xR3",
        "--pipeline.model.num-downscales=0",
        "--pipeline.model.sh-degree=0",
        "--pipeline.datamanager.cache-images=cpu",
        "--pipeline.model.use_bilateral_grid", "True",
        # "--pipeline.model.strategy", "mcmc",
        # "--pipeline.model.max_gs_num", "1000000",
        "--pipeline.model.warmup_length", "30000",  # 不分裂
        # "--pipeline.model.refine-every", "500",
        # "--pipeline.model.stop-split-at","25000",
        "--pipeline.model.progress_path", progress_path,
        "--vis", "wandb",
        "colmap",
        "--downscale_factor","8", # 低分辨率
        "--images_path",image_path,
        "--colmap_path", "sparse/0",
        "--orientation-method", "none",
        "--center-method", "none",
        "--auto-scale-poses", "False",
        "--assume-colmap-world-coordinate-convention", "False",
        "--load_ply_path", output_ply,
        # "--load_combined_ply", "True",
        # "--resample", "True"
    ]
    try:
        subprocess.run(cmd_train, cwd=NERFSTUDIO_DIR, check=True)
        CONSOLE.print("[green]2 Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        exit(1)
    # 导出
    cmd_export = [
        # PYTHON_EXECUTABLE ,"-m","ns-export", "gaussian-splat",
        PYTHON_EXECUTABLE, "nerfstudio/scripts/exporter.py",
        "gaussian-splat",
        "--load-config", load_config_path,
        "--output-dir", export_output_dir
    ]
    # 添加安全全局变量以解决PyTorch 2.6 weights_only限制
    import torch.serialization
    torch.serialization.add_safe_globals([getattr(torch.serialization, 'scalar', None)])
    try:
        subprocess.run(cmd_export, cwd=NERFSTUDIO_DIR, check=True)
        CONSOLE.print("[green]3 Export completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        exit(1)
    # 裁剪
    cmd_crop = [
        PYTHON_EXECUTABLE, "crop_ply_aabb.py",
        "--reference_ply", crop_reference_ply,
        "--input_ply", crop_input_ply,
        "--output_ply", crop_output_ply,
        "--scale_factor", "10"
    ]
    try:
        subprocess.run(cmd_crop, cwd=NERFSTUDIO_DIR, check=True)
        CONSOLE.print("[green]4 Cropping completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        exit(1)
    
    convert_ply_to_splat(crop_output_ply, splat_output_ply)
    
    # 总时间
    total_elapsed_time = time.time() - total_start_time
    CONSOLE.print(f"[blue]所有步骤完成，总用时: {format_time(total_elapsed_time)}")

if __name__ == "__main__":
    main()
