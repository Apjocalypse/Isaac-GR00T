# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np
import os
import tyro

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import BasePolicy, Gr00tPolicy
from gr00t.utils.eval import calc_mse_for_single_trajectory_genie
import gr00t.data.pretrainAe_a2d_pretrain_v6 as a2d_cfg
from gr00t.data.schema import EmbodimentTag

from transformers import AutoTokenizer, AutoProcessor, AutoConfig, AutoImageProcessor


warnings.simplefilter("ignore", category=FutureWarning)

"""
Example command:

NOTE: provide --model_path to load up the model checkpoint in this script,
        else it will use the default host and port via RobotInferenceClient

python scripts/eval_policy.py --plot --model-path nvidia/GR00T-N1.5-3B
"""


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    host: str = "localhost"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    plot: bool = False
    """Whether to plot the images."""

    modality_keys: List[str] = field(default_factory=lambda: ["right_arm", "left_arm"])
    """Modality keys to evaluate."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_only"
    """Data config to use."""

    steps: int = 150
    """Number of steps to evaluate."""

    trajs: int = 1
    """Number of trajectories to evaluate."""

    action_horizon: int = 16
    """Action horizon to evaluate."""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Video backend to use for various codec options. h264: decord or av: torchvision_av"""

    dataset_name: str = "robot_sim.PickNPlace"

    dataset_root_dir: str = f"/home/anpengju/datasets/Manipulation-SimData/"

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """Embodiment tag to use."""

    model_path: str = None
    """Path to the model checkpoint."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    local_log_dir: str = "./experiments/eval_logs"

    window_size: int = 30
    """Window size for the dataset."""

    load_meta: bool = False

    with_proprio: bool = True

    debug: bool = False

    action_horizon: int = 16

    action_dim: int = 16

    save_path: str = "/home/anpengju/Isaac-GR00T-challenge/open_loop.png"


def get_policy(args: ArgsConfig):
    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_configs = data_config.modality_config()
    transforms = data_config.transform()
    args.dataset_root_dir = args.dataset_root_dir + args.dataset_name
    if args.model_path is not None:
        import torch

        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy: BasePolicy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            device="cuda" if torch.cuda.is_available() else "cpu",
            load_meta=args.load_meta
        )
    else:
        policy: BasePolicy = RobotInferenceClient(host=args.host, port=args.port)

    os.makedirs(args.local_log_dir, exist_ok=True)
    # policy.set_action_dim(args.action_dim)

    val_set = {}
    val_set[args.dataset_name] = {
        "use_cam_list": ["head", "hand_right", "hand_left"],
        "label_file_name": f"val.json",
    }

    # Load gensim dataset
    from gr00t.data.agibot_dataset import A2dDataset
    dataset_args = a2d_cfg.DatasetArguments(
        meta_json_dir=args.dataset_root_dir,
        data_root_dir=args.dataset_root_dir,
        dataset_task_cfg=val_set
    )
    data_training_args = a2d_cfg.DataTrainingArguments(force_image_size=224)
    ActionSpacePadder = a2d_cfg.ActionSpacePadderArguments()

    text_tokenizer = AutoTokenizer.from_pretrained(
        "OpenGVLab/InternVL2-2B",
        trust_remote_code=True,
        add_eos_token=False,
    )

    text_tokenizer.model_max_length = 4096

    vla_dataset = A2dDataset(
        # base parmas
        label_file_dir=dataset_args.meta_json_dir, 
        data_root_dir=dataset_args.data_root_dir, 
        valid_episode_txt=dataset_args.valid_episode_txt, 
        sample_rate=dataset_args.train_sample_rate, 
        online_process_mp_cnt=dataset_args.online_process_mp_cnt, 
        # a2d params
        text_tokenizer=text_tokenizer, 
        num_image_token=int((dataset_args.force_image_size // 14) ** 2 * (0.5**2)), 
        is_train=True, 
        image_size=data_training_args.force_image_size, 
        pad2square=data_training_args.pad2square, 
        dynamic_image_size=data_training_args.dynamic_image_size, 
        use_thumbnail=data_training_args.use_thumbnail, 
        min_dynamic_patch=data_training_args.min_dynamic_patch, 
        max_dynamic_patch=data_training_args.max_dynamic_patch, 
        normalize_type=data_training_args.normalize_type, 
        action_chunk_size=data_training_args.action_chunk_size, 
        # use_real_state=data_training_args.use_real_state, 
        use_real_state=True, 
        conversation_type=data_training_args.conversation_type, 
        vis_frame=False, 
        vis_dir="", 
        ActionSpacePadder=ActionSpacePadder, 
        min_window_size=args.window_size, 
        max_window_size=args.window_size + 1, 
        # image_transform=image_processor.apply_transform, 
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        transforms=transforms,
        # modality_configs=modality_configs
    )

    vla_dataset.generate_task_infos(
        dataset_args.dataset_task_cfg,
        task_episode_processors_cfg=dataset_args.episode_processors,
        task_dataset_processors_cfg=dataset_args.dataset_processors,
        task_runtime_processors_cfg=dataset_args.runtime_processors,
        shuffle=True,
        statistic=True,
        debug_one_episode=args.debug,
        # debug_one_episode=False,
    )

    return policy, vla_dataset

    # Get the supported modalities for the policy
    # modality = policy.get_modality_config()
    # print("Current modality config: \n", modality)

    # # Create the dataset
    # dataset = LeRobotSingleDataset(
    #     dataset_path=args.dataset_path,
    #     modality_configs=modality,
    #     video_backend=args.video_backend,
    #     video_backend_kwargs=None,
    #     transforms=None,  # We'll handle transforms separately through the policy
    #     embodiment_tag=args.embodiment_tag,
    # )

    # print(len(dataset))
    # # Make a prediction
    # obs = dataset[0]
    # for k, v in obs.items():
    #     if isinstance(v, np.ndarray):
    #         print(k, v.shape)
    #     else:
    #         print(k, v)

    # for k, v in dataset.get_step_data(0, 0).items():
    #     if isinstance(v, np.ndarray):
    #         print(k, v.shape)
    #     else:
    #         print(k, v)

    # print("Total trajectories:", len(dataset.trajectory_lengths))
    # print("All trajectories:", dataset.trajectory_lengths)
    # print("Running on all trajs with modality keys:", args.modality_keys)

    # all_mse = []
    # for traj_id in range(args.trajs):
    #     print("Running trajectory:", traj_id)
    #     mse = calc_mse_for_single_trajectory(
    #         policy,
    #         dataset,
    #         traj_id,
    #         modality_keys=args.modality_keys,
    #         steps=args.steps,
    #         action_horizon=args.action_horizon,
    #         plot=args.plot,
    #     )
    #     print("MSE:", mse)
    #     all_mse.append(mse)
    # print("Average MSE across all trajs:", np.mean(all_mse))
    # print("Done")
    # exit()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    policy, dataset = get_policy(config)
    mse = calc_mse_for_single_trajectory_genie(
        policy,
        dataset,
        config,
        traj_id=0,
        steps=100,
        action_horizon=16,
        action_dim=config.action_dim,
        plot=True
    )
    print("MSE loss for trajectory 0:", mse)
