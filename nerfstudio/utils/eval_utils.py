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
Evaluation utils
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import torch
import yaml

from nerfstudio.utils.rich_utils import CONSOLE

import typing
from collections import OrderedDict
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import open3d as o3d

def eval_load_checkpoint(config, pipeline, load_step = None) -> Tuple[Path, int]:
    ## TODO: ideally eventually want to get this to be the same as whatever is used to load train checkpoint too
    """Helper function to load checkpointed pipeline

    Args:
        config (DictConfig): Configuration of pipeline to load
        pipeline (Pipeline): Pipeline instance of which to load weights
    Returns:
        A tuple of the path to the loaded checkpoint and the step at which it was saved.
    """
    assert config.load_dir is not None
    if load_step is None:
        CONSOLE.print("Loading latest checkpoint from load_dir")
        # NOTE: this is specific to the checkpoint name format
        if not os.path.exists(config.load_dir):
            CONSOLE.rule("Error", style="red")
            CONSOLE.print(f"No checkpoint directory found at {config.load_dir}, ", justify="center")
            CONSOLE.print(
                "Please make sure the checkpoint exists, they should be generated periodically during training",
                justify="center",
            )
            sys.exit(1)
        load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.load_dir))[-1]
    else:
        load_step = load_step
    load_path = config.load_dir / f"step-{load_step:09d}.ckpt"
    assert load_path.exists(), f"Checkpoint {load_path} does not exist"
    loaded_state = torch.load(load_path, map_location="cpu", weights_only=False)
    pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
    CONSOLE.print(f":white_check_mark: Done loading checkpoint from {load_path}")
    return load_path, load_step


def eval_setup(
    config_path: Path,
    eval_num_rays_per_chunk: Optional[int] = None,
    test_mode: Literal["test", "val", "inference"] = "test",
    update_config_callback = None,
    load_step = None,
):
    """Shared setup for loading a saved pipeline for evaluation.

    Args:
        config_path: Path to config YAML file.
        eval_num_rays_per_chunk: Number of rays per forward pass
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        update_config_callback: Callback to update the config before loading the pipeline


    Returns:
        Loaded config, pipeline module, corresponding checkpoint, and step
    """
    # load save config
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    
    from nerfstudio.engine.trainer import TrainerConfig
    from nerfstudio.pipelines.base_pipeline import Pipeline
    assert isinstance(config, TrainerConfig)
    from nerfstudio.configs.method_configs import all_methods

    config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target
    if eval_num_rays_per_chunk:
        config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk

    if update_config_callback is not None:
        config = update_config_callback(config)

    # load checkpoints from wherever they were saved
    # TODO: expose the ability to choose an arbitrary checkpoint
    config.load_dir = config.get_checkpoint_dir()

    # setup pipeline (which includes the DataManager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode=test_mode)
    assert isinstance(pipeline, Pipeline)
    pipeline.eval()

    # load checkpointed information
    checkpoint_path, step = eval_load_checkpoint(config, pipeline, load_step)

    return config, pipeline, checkpoint_path, step


def write_ply(
    filename: str,
    count: int,
    map_to_tensors: typing.OrderedDict[str, np.ndarray],
):
    """
    Writes a PLY file with given vertex properties and a tensor of float or uint8 values in the order specified by the OrderedDict.
    Note: All float values will be converted to float32 for writing.

    Parameters:
    filename (str): The name of the file to write.
    count (int): The number of vertices to write.
    map_to_tensors (OrderedDict[str, np.ndarray]): An ordered dictionary mapping property names to numpy arrays of float or uint8 values.
        Each array should be 1-dimensional and of equal length matching 'count'. Arrays should not be empty.
    """

    # Ensure count matches the length of all tensors
    if not all(tensor.size == count for tensor in map_to_tensors.values()):
        raise ValueError("Count does not match the length of all tensors")

    # Type check for numpy arrays of type float or uint8 and non-empty
    if not all(
        isinstance(tensor, np.ndarray)
        and (tensor.dtype.kind == "f" or tensor.dtype == np.uint8)
        and tensor.size > 0
        for tensor in map_to_tensors.values()
    ):
        raise ValueError("All tensors must be numpy arrays of float or uint8 type and not empty")

    with open(filename, "wb") as ply_file:
        nerfstudio_version = version("nerfstudio")
        # Write PLY header
        ply_file.write(b"ply\n")
        ply_file.write(b"format binary_little_endian 1.0\n")
        ply_file.write(f"comment Generated by Nerstudio {nerfstudio_version}\n".encode())
        ply_file.write(b"comment Vertical Axis: z\n")
        ply_file.write(f"element vertex {count}\n".encode())

        # Write properties, in order due to OrderedDict
        for key, tensor in map_to_tensors.items():
            data_type = "float" if tensor.dtype.kind == "f" else "uchar"
            ply_file.write(f"property {data_type} {key}\n".encode())

        ply_file.write(b"end_header\n")

        # Write binary data
        # Note: If this is a performance bottleneck consider using numpy.hstack for efficiency improvement
        for i in range(count):
            for tensor in map_to_tensors.values():
                value = tensor[i]
                if tensor.dtype.kind == "f":
                    ply_file.write(np.float32(value).tobytes())
                elif tensor.dtype == np.uint8:
                    ply_file.write(value.tobytes())

def export_gs(output_dir,load_config, output_filename, load_step = None,ply_color_mode="sh_coeffs" ):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir,exist_ok=True)

    _, pipeline, _, _ = eval_setup(load_config, test_mode="inference", load_step=load_step)
    from nerfstudio.models.splatfacto import SplatfactoModel

    assert isinstance(pipeline.model, SplatfactoModel)

    model: SplatfactoModel = pipeline.model
    # model = pipeline.model
    filename = os.path.join(output_dir , output_filename)

    map_to_tensors = OrderedDict()

    with torch.no_grad():
        positions = model.means.cpu().numpy()
        count = positions.shape[0]
        n = count
        map_to_tensors["x"] = positions[:, 0]
        map_to_tensors["y"] = positions[:, 1]
        map_to_tensors["z"] = positions[:, 2]
        map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
        map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
        map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)

        if ply_color_mode == "rgb":
            colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
            colors = (colors * 255).astype(np.uint8)
            map_to_tensors["red"] = colors[:, 0]
            map_to_tensors["green"] = colors[:, 1]
            map_to_tensors["blue"] = colors[:, 2]
        elif ply_color_mode == "sh_coeffs":
            shs_0 = model.shs_0.contiguous().cpu().numpy()
            for i in range(shs_0.shape[1]):
                map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

        if model.config.sh_degree > 0:
            if ply_color_mode == "rgb":
                CONSOLE.print(
                    "Warning: model has higher level of spherical harmonics, ignoring them and only export rgb."
                )
            elif ply_color_mode == "sh_coeffs":
                # transpose(1, 2) was needed to match the sh order in Inria version
                shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
                shs_rest = shs_rest.reshape((n, -1))
                for i in range(shs_rest.shape[-1]):
                    map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]

        map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()

        scales = model.scales.data.cpu().numpy()
        for i in range(3):
            map_to_tensors[f"scale_{i}"] = scales[:, i, None]

        quats = model.quats.data.cpu().numpy()
        for i in range(4):
            map_to_tensors[f"rot_{i}"] = quats[:, i, None]

    # post optimization, it is possible have NaN/Inf values in some attributes
    # to ensure the exported ply file has finite values, we enforce finite filters.
    select = np.ones(n, dtype=bool)
    for k, t in map_to_tensors.items():
        n_before = np.sum(select)
        select = np.logical_and(select, np.isfinite(t).all(axis=-1))
        n_after = np.sum(select)
        if n_after < n_before:
            CONSOLE.print(f"{n_before - n_after} NaN/Inf elements in {k}")
    nan_count = np.sum(select) - n

    # filter gaussians that have opacities < 1/255, because they are skipped in cuda rasterization
    low_opacity_gaussians = (map_to_tensors["opacity"]).squeeze(axis=-1) < -5.5373  # logit(1/255)
    lowopa_count = np.sum(low_opacity_gaussians)
    select[low_opacity_gaussians] = 0

    if np.sum(select) < n:
        CONSOLE.print(
            f"{nan_count} Gaussians have NaN/Inf and {lowopa_count} have low opacity, only export {np.sum(select)}/{n}"
        )
        for k, t in map_to_tensors.items():
            map_to_tensors[k] = map_to_tensors[k][select]
        count = np.sum(select)

    write_ply(str(filename), count, map_to_tensors)