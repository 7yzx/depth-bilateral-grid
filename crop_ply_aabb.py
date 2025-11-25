import numpy as np
from plyfile import PlyData, PlyElement
import argparse
import open3d as o3d
import os

def read_ply_with_attributes_o3d(file_path):
    """
    Reads a PLY file using Open3D and extracts xyz coordinates.
    """
    # 使用 Open3D 读取点云
    pcd = o3d.io.read_point_cloud(file_path)

    # 提取 xyz 坐标
    xyz = np.asarray(pcd.points)

    return xyz

def read_ply_with_attributes(file_path, max_sh_degree=0):
    """
    Reads a PLY file and extracts all relevant attributes.
    """
    plydata = PlyData.read(file_path)

    # Extract xyz coordinates
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)

    # Extract other attributes
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3

    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (len(extra_f_names) // 3)))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    return plydata, xyz, opacities, features_dc, features_extra, scales, rots

def get_bounding_box(xyz, scale_factor=2.0):
    """
    Computes the bounding box of the given points and allows scaling the range.

    Parameters:
    - xyz (numpy.ndarray): The point cloud data, shape (N, 3).
    - scale_factor (float): A scaling factor to expand or shrink the bounding box. Default is 1.0.

    Returns:
    - min_coords (numpy.ndarray): The minimum coordinates of the bounding box, shape (3,).
    - max_coords (numpy.ndarray): The maximum coordinates of the bounding box, shape (3,).
    """
    # Compute the original bounding box
    min_coords = xyz.min(axis=0)
    max_coords = xyz.max(axis=0)

    # Compute the center and half-size of the bounding box
    center = (min_coords + max_coords) / 2
    half_size = (max_coords - min_coords) / 2

    # Scale the half-size by the scale factor
    half_size *= scale_factor

    # Recompute the min and max coordinates
    min_coords = center - half_size
    max_coords = center + half_size

    return min_coords, max_coords

def clip_data_by_bounding_box(xyz, min_coords, max_coords, *attributes):
    """
    Clips the data (xyz and other attributes) based on the bounding box.
    """
    mask = np.all((xyz >= min_coords) & (xyz <= max_coords), axis=1)
    clipped_xyz = xyz[mask]
    clipped_attributes = [attr[mask] for attr in attributes]
    return clipped_xyz, clipped_attributes, mask

def construct_list_of_attributes(features_dc, features_rest, scaling, rotation):
    """
    Constructs a list of attribute names based on the provided features.
    """
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(features_dc.shape[1] * features_dc.shape[2]):
        l.append(f'f_dc_{i}')
    for i in range(features_rest.shape[1] * features_rest.shape[2]):
        l.append(f'f_rest_{i}')
    l.append('opacity')
    for i in range(scaling.shape[1]):
        l.append(f'scale_{i}')
    for i in range(rotation.shape[1]):
        l.append(f'rot_{i}')
    return l

def write_clipped_ply(plydata, output_path, xyz, opacities, features_dc, features_extra, scales, rots):
    """
    Writes the clipped data to a new PLY file.
    """
    # Create a new structured array for the clipped data
    normals = np.zeros_like(xyz)
    f_dc = features_dc.reshape(features_dc.shape[0], -1)  # Flatten along the last two dimensions
    f_rest = features_extra.reshape(features_extra.shape[0], -1)  # Flatten along the last two dimensions

    opacities = opacities
    scale = scales
    rotation = rots
    
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(features_dc, features_extra, scales, rots)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)


def crop_gs(output_ply, reference_ply, scale_factor, input_ply, max_sh_degree):
    # 获取文件的上级目录路径
    out_put_dir = os.path.dirname(output_ply)

    # 确保目录存在
    os.makedirs(out_put_dir, exist_ok=True)
    
    # Step 1: Read the reference PLY file and compute its bounding box
    ref_xyz = read_ply_with_attributes_o3d(reference_ply)
    min_coords, max_coords = get_bounding_box(ref_xyz,scale_factor)

    # Step 2: Read the input PLY file
    plydata, input_xyz, opacities, features_dc, features_extra, scales, rots = read_ply_with_attributes(input_ply,max_sh_degree)

    # Step 3: Clip the input data based on the bounding box
    clipped_xyz, clipped_attributes, _ = clip_data_by_bounding_box(
        input_xyz, min_coords, max_coords, opacities, features_dc, features_extra, scales, rots
    )

    # Step 4: Write the clipped data to the output PLY file
    write_clipped_ply(plydata, output_ply, clipped_xyz, *clipped_attributes)

def main():
    parser = argparse.ArgumentParser(description="Clip a PLY file based on a reference PLY file's bounding box.")
    parser.add_argument("--reference_ply", required=True, type=str, help="Path to the reference PLY file")
    parser.add_argument("--input_ply", required=True, type=str, help="Path to the input PLY file")
    parser.add_argument("--output_ply", required=True, type=str, help="Path to save the clipped PLY file")
    parser.add_argument("--scale_factor", type=float, default=2.0, help="Scale factor for the bounding box")
    parser.add_argument("--max_sh_degree", type=int, default=0, help="Maximum SH degree for extra features")
    args = parser.parse_args()
    
    # 获取文件的上级目录路径
    out_put_dir = os.path.dirname(args.output_ply)

    # 确保目录存在
    os.makedirs(out_put_dir, exist_ok=True)
    
    # Step 1: Read the reference PLY file and compute its bounding box
    ref_xyz = read_ply_with_attributes_o3d(args.reference_ply)
    min_coords, max_coords = get_bounding_box(ref_xyz,args.scale_factor)

    # Step 2: Read the input PLY file
    plydata, input_xyz, opacities, features_dc, features_extra, scales, rots = read_ply_with_attributes(args.input_ply,args.max_sh_degree)

    # Step 3: Clip the input data based on the bounding box
    clipped_xyz, clipped_attributes, _ = clip_data_by_bounding_box(
        input_xyz, min_coords, max_coords, opacities, features_dc, features_extra, scales, rots
    )

    # Step 4: Write the clipped data to the output PLY file
    write_clipped_ply(plydata, args.output_ply, clipped_xyz, *clipped_attributes)
    print(f"Clipped PLY file saved to {args.output_ply}")

if __name__ == "__main__":
    main()