# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Miscellaneous helper code.
"""

import platform
import typing
import warnings
from inspect import currentframe
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import torch

import numpy as np
from plyfile import PlyData, PlyElement
import os
import open3d as o3d
T = TypeVar("T")
TKey = TypeVar("TKey")


def get_dict_to_torch(stuff: T, device: Union[torch.device, str] = "cpu", exclude: Optional[List[str]] = None) -> T:
    """Set everything in the dict to the specified torch device.

    Args:
        stuff: things to convert to torch
        device: machine to put the "stuff" on
        exclude: list of keys to skip over transferring to device
    """
    if isinstance(stuff, dict):
        for k, v in stuff.items():
            if exclude and k in exclude:
                stuff[k] = v
            else:
                stuff[k] = get_dict_to_torch(v, device)
        return stuff
    if isinstance(stuff, torch.Tensor):
        return stuff.to(device)
    return stuff


def get_dict_to_cpu(stuff: T) -> T:
    """Set everything in the dict to CPU.

    Args:
        stuff: things to place onto cpu
    """
    if isinstance(stuff, dict):
        for k, v in stuff.items():
            stuff[k] = get_dict_to_cpu(v)
        return stuff
    if isinstance(stuff, torch.Tensor):
        return stuff.detach().cpu()
    return stuff


def get_masked_dict(d: Dict[TKey, torch.Tensor], mask) -> Dict[TKey, torch.Tensor]:
    """Return a masked dictionary.
    TODO(ethan): add more asserts/checks so this doesn't have unpredictable behavior.

    Args:
        d: dict to process
        mask: mask to apply to values in dictionary
    """
    masked_dict = {}
    for key, value in d.items():
        masked_dict[key] = value[mask]
    return masked_dict


class IterableWrapper:
    """A helper that will allow an instance of a class to return multiple kinds of iterables bound
    to different functions of that class.

    To use this, take an instance of a class. From that class, pass in the <instance>.<new_iter_function>
    and <instance>.<new_next_function> to the IterableWrapper constructor. By passing in the instance's
    functions instead of just the class's functions, the self argument should automatically be accounted
    for.

    Args:
        new_iter: function that will be called instead as the __iter__() function
        new_next: function that will be called instead as the __next__() function
        length: length of the iterable. If -1, the iterable will be infinite.


    Attributes:
        new_iter: object's pointer to the function we are calling for __iter__()
        new_next: object's pointer to the function we are calling for __next__()
        length: length of the iterable. If -1, the iterable will be infinite.
        i: current index of the iterable.

    """

    i: int

    def __init__(self, new_iter: Callable, new_next: Callable, length: int = -1):
        self.new_iter = new_iter
        self.new_next = new_next
        self.length = length

    def __next__(self):
        if self.length != -1 and self.i >= self.length:
            raise StopIteration
        self.i += 1
        return self.new_next()

    def __iter__(self):
        self.new_iter()
        self.i = 0
        return self


def scale_dict(dictionary: Dict[Any, Any], coefficients: Dict[str, float]) -> Dict[Any, Any]:
    """Scale a dictionary in-place given a coefficients dictionary.

    Args:
        dictionary: input dict to be scaled.
        coefficients: scalar dict config for holding coefficients.

    Returns:
        Input dict scaled by coefficients.
    """
    for key in dictionary:
        if key in coefficients:
            dictionary[key] *= coefficients[key]
    return dictionary


def step_check(step, step_size, run_at_zero=False) -> bool:
    """Returns true based on current step and step interval."""
    if step_size == 0:
        return False
    return (run_at_zero or step != 0) and step % step_size == 0

def step_iteration_save(step, save_iteration):

    return step == save_iteration

def update_avg(prev_avg: float, new_val: float, step: int) -> float:
    """helper to calculate the running average

    Args:
        prev_avg (float): previous average value
        new_val (float): new value to update the average with
        step (int): current step number

    Returns:
        float: new updated average
    """
    return (step * prev_avg + new_val) / (step + 1)


def strtobool(val) -> bool:
    """Cheap replacement for `distutils.util.strtobool()` which is deprecated
    FMI https://stackoverflow.com/a/715468
    """
    return val.lower() in ("yes", "y", "true", "t", "on", "1")


def torch_compile(*args, **kwargs) -> Any:
    """
    Safe torch.compile with backward compatibility for PyTorch 1.x
    """
    if not hasattr(torch, "compile"):
        # Backward compatibility for PyTorch 1.x
        warnings.warn(
            "PyTorch 1.x will no longer be supported by Nerstudio. Please upgrade to PyTorch 2.x.", DeprecationWarning
        )
        if args and isinstance(args[0], torch.nn.Module):
            return args[0]
        else:
            return torch.jit.script
    elif platform.system() == "Windows":
        # torch.compile is not supported on Windows
        # https://github.com/orgs/pytorch/projects/27
        # TODO: @jkulhanek, remove this once torch.compile is supported on Windows
        warnings.warn(
            "Windows does not yet support torch.compile and the performance will be affected.", RuntimeWarning
        )
        if args and isinstance(args[0], torch.nn.Module):
            return args[0]
        else:
            return lambda x: x
    else:
        return torch.compile(*args, **kwargs)


def get_orig_class(obj, default=None):
    """Returns the __orig_class__ class of `obj` even when it is not initialized in __init__ (Python>=3.8).

    Workaround for https://github.com/python/typing/issues/658.
    Inspired by https://github.com/Stewori/pytypes/pull/53.
    """
    try:
        return object.__getattribute__(obj, "__orig_class__")
    except AttributeError:
        cls = object.__getattribute__(obj, "__class__")
        try:
            is_type_generic = isinstance(cls, typing.GenericMeta)  # type: ignore
        except AttributeError:  # Python 3.8
            is_type_generic = issubclass(cls, typing.Generic)
        if is_type_generic:
            frame = currentframe().f_back.f_back  # type: ignore
            try:
                while frame:
                    try:
                        res = frame.f_locals["self"]
                        if res.__origin__ is cls:
                            return res
                    except (KeyError, AttributeError):
                        frame = frame.f_back
            finally:
                del frame
        return default

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


