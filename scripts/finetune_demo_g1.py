import os
import torch

from transformers import TrainingArguments

from gr00t.data.schema import EmbodimentTag
from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform import VideoToTensor, VideoCrop, VideoResize, VideoColorJitter, VideoToNumpy
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.model.transforms import GR00TTransform
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.gr00t_n1 import GR00T_N1
from gr00t.experiment.runner import TrainRunner

dataset_path = "/home/jiyuheng/Isaac-GR00T/demo_data/G1_BlockStacking_Dataset"  # change this to your dataset path
embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

# select the modality keys you want to use for finetuning
video_modality = ModalityConfig(
    delta_indices=[0],
    modality_keys=["video.cam_right_high"],
)

state_modality = ModalityConfig(
    delta_indices=[0],
    modality_keys=["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand"],
)

action_modality = ModalityConfig(
    delta_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    modality_keys=["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"],
)

language_modality = ModalityConfig(
    delta_indices=[0],
    modality_keys=["annotation.human.task_description"],
)

modality_configs = {
    "video": video_modality,
    "state": state_modality,
    "action": action_modality,
    "language": language_modality,
}



# select the transforms you want to apply to the data
to_apply_transforms = ComposedModalityTransform(
    transforms=[
        # video transforms
        VideoToTensor(apply_to=video_modality.modality_keys, backend="torchvision"),
        VideoCrop(apply_to=video_modality.modality_keys, scale=0.95, backend="torchvision"),
        VideoResize(apply_to=video_modality.modality_keys, height=224, width=224, interpolation="linear", backend="torchvision" ),
        VideoColorJitter(apply_to=video_modality.modality_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08, backend="torchvision"),
        VideoToNumpy(apply_to=video_modality.modality_keys),

        # state transforms
        StateActionToTensor(apply_to=state_modality.modality_keys),
        StateActionTransform(apply_to=state_modality.modality_keys, normalization_modes={
            "state.left_arm": "min_max",
            "state.right_arm": "min_max",
            "state.left_hand": "min_max",
            "state.right_hand": "min_max",
        }),

        # action transforms
        StateActionToTensor(apply_to=action_modality.modality_keys),
        StateActionTransform(apply_to=action_modality.modality_keys, normalization_modes={
            "action.right_arm": "min_max",
            "action.left_arm": "min_max",
            "action.right_hand": "min_max",
            "action.left_hand": "min_max",
        }),

        # ConcatTransform
        ConcatTransform(
            video_concat_order=video_modality.modality_keys,
            state_concat_order=state_modality.modality_keys,
            action_concat_order=action_modality.modality_keys,
        ),
        # model-specific transform
        GR00TTransform(
            state_horizon=len(state_modality.delta_indices),
            action_horizon=len(action_modality.delta_indices),
            max_state_dim=64,
            max_action_dim=32,
        ),
    ]
)


# train_dataset = LeRobotSingleDataset(
#     dataset_path=dataset_path,
#     modality_configs=modality_configs,
#     embodiment_tag=embodiment_tag,
#     video_backend="torchvision_av",
# )


# # use matplotlib to visualize the images
# import matplotlib.pyplot as plt
# import numpy as np

# print(train_dataset[0].keys())

# images = []
# for i in range(5):
#     image = train_dataset[i]["video.cam_right_high"][0]
#     # image is in HWC format, convert it to CHW format
#     image = image.transpose(2, 0, 1)
#     images.append(image)   

# fig, axs = plt.subplots(1, 5, figsize=(20, 5))
# for i, image in enumerate(images):
#     axs[i].imshow(np.transpose(image, (1, 2, 0)))
#     axs[i].axis("off")
# plt.show()

train_dataset = LeRobotSingleDataset(
    dataset_path=dataset_path,
    modality_configs=modality_configs,
    embodiment_tag=embodiment_tag,
    video_backend="torchvision_av",
    transforms=to_apply_transforms,
)


device = "cuda" if torch.cuda.is_available() else "cpu"



BASE_MODEL_PATH = "/home/jiyuheng/groot_n1"
TUNE_LLM = False            # Whether to tune the LLM
TUNE_VISUAL = True          # Whether to tune the visual encoder
TUNE_PROJECTOR = True       # Whether to tune the projector
TUNE_DIFFUSION_MODEL = True # Whether to tune the diffusion model

model = GR00T_N1.from_pretrained(
    pretrained_model_name_or_path=BASE_MODEL_PATH,
    tune_llm=TUNE_LLM,  # backbone's LLM
    tune_visual=TUNE_VISUAL,  # backbone's vision tower
    tune_projector=TUNE_PROJECTOR,  # action head's projector
    tune_diffusion_model=TUNE_DIFFUSION_MODEL,  # action head's DiT
)

# Set the model's compute_dtype to bfloat16
model.compute_dtype = "bfloat16"
model.config.compute_dtype = "bfloat16"
model.to(device)

output_dir = "/home/jiyuheng/n1_ckpt/g1_demo"    # CHANGE THIS ACCORDING TO YOUR LOCAL PATH
per_device_train_batch_size = 8     # CHANGE THIS ACCORDING TO YOUR GPU MEMORY
max_steps = 5000                      # CHANGE THIS ACCORDING TO YOUR NEEDS
report_to = "wandb"
dataloader_num_workers = 8

training_args = TrainingArguments(
    output_dir=output_dir,
    run_name=None,
    remove_unused_columns=False,
    deepspeed="",
    gradient_checkpointing=False,
    bf16=True,
    tf32=True,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=1,
    dataloader_num_workers=dataloader_num_workers,
    dataloader_pin_memory=False,
    dataloader_persistent_workers=True,
    optim="adamw_torch",
    adam_beta1=0.95,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    learning_rate=1e-4,
    weight_decay=1e-5,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=10.0,
    num_train_epochs=300,
    max_steps=max_steps,
    save_strategy="steps",
    save_steps=1000,
    evaluation_strategy="no",
    save_total_limit=8,
    report_to=report_to,
    seed=42,
    do_eval=False,
    ddp_find_unused_parameters=False,
    ddp_bucket_cap_mb=100,
    torch_compile_mode=None,
)


experiment = TrainRunner(
    train_dataset=train_dataset,
    model=model,
    training_args=training_args,
)

experiment.train()