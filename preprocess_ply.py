import open3d as o3d
import argparse

def sor_ply(pcd,neighbors=30,std_ratio=3):
    len_before = len(pcd.points)
    cl, _ = pcd.remove_statistical_outlier(nb_neighbors=neighbors, std_ratio=std_ratio)
    len_sor = len(cl.points)
    print("After SOR, the point cloud has {} points, remove {} points".format(len_sor, len_before - len_sor))
    return cl

def adaptive_voxel_down_sample(pcd_path, target_points=1_000_000, init_voxel_size=0.0005,sor_enable = True ,ratio_tolerance=0.1,neighbors=30,std_ratio=3):
    """自适应调整voxel_size，使降采样后的点数接近 target_points"""
    lower_bound, upper_bound = 0.0001, 0.05  # 体素大小范围
    voxel_size = init_voxel_size
    max_iterations = 20  # 限制最大迭代次数
    pcd = o3d.io.read_point_cloud(pcd_path)
    num_points_original = len(pcd.points)
    if num_points_original > target_points + 100_000:
        for _ in range(max_iterations):
            down_pcd = pcd.voxel_down_sample(voxel_size)
            num_points = len(down_pcd.points)
            print(f"Voxel Size: {voxel_size:.6f}, Points: {num_points}")

            # 允许5%误差，即点数在 950k ~ 1050k 之间
            if (1 - ratio_tolerance) * target_points <= num_points <= (1 +0.1+ ratio_tolerance) * target_points:
                if sor_enable:
                    final_pcd = sor_ply(down_pcd)
                    return final_pcd, voxel_size
                else:
                    return down_pcd, voxel_size

            # 调整voxel_size
            if num_points > target_points:  # 点数太多 -> 增大voxel_size
                lower_bound = voxel_size
                voxel_size = (voxel_size + upper_bound) / 2
            else:  # 点数太少 -> 减小voxel_size
                upper_bound = voxel_size
                voxel_size = (voxel_size + lower_bound) / 2
                
    if sor_enable:
        final_pcd = sor_ply(pcd)
        return final_pcd, voxel_size  # 返回最终降采样的点云和voxel_size
    else:
        return pcd, voxel_size

def main():
    parser = argparse.ArgumentParser(description="Adaptive Voxel Downsampling for Point Clouds")
    parser.add_argument("--input_file", required=True, type=str, help="Path to the input PLY file")
    parser.add_argument("--output_file", required=True, type=str, help="Path to save the downsampled PLY file")
    parser.add_argument("--target_points", type=int, default=1_000_000, help="Target number of points after downsampling")
    parser.add_argument("--init_voxel_size", type=float, default=0.0005, help="Initial voxel size for downsampling")
    parser.add_argument("--sor_enable", action="store_true",default=True)

    args = parser.parse_args()

    # 读取点云
    print(f"Loading point cloud from {args.input_file}...")
    
    # 进行自适应降采样
    down_pcd, final_voxel_size = adaptive_voxel_down_sample(args.input_file, args.target_points, args.init_voxel_size,args.sor_enable)

    # 保存结果
    o3d.io.write_point_cloud(args.output_file, down_pcd)
    print(f"Downsampled point cloud saved to {args.output_file}")

    # 可视化

if __name__ == "__main__":
    main()

