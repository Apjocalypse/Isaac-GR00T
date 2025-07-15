import os
import torch
import tyro
import subprocess
import sys
from pathlib import Path

from transformers import TrainingArguments
from dataclasses import dataclass

from transformers import (
    AutoModelForVision2Seq, 
    AutoProcessor, 
    BitsAndBytesConfig, 
    AutoConfig, 
    AutoImageProcessor,
    AutoTokenizer,
)
from gr00t.data.transform.processing_prismatic import PrismaticImageProcessor
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.experiment.runner import TrainRunner
import gr00t.data.pretrainAe_a2d_pretrain_v6 as a2d_cfg
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP

@dataclass
class FinetuneConfig:
    vla_path: str = "nvidia/GR00T-N1.5-3B"
    
    # dataset_name: str = "restock_supermarket_items"
    dataset_path: str = "/robot/embodied-perception-data/user/hb/data/Manipulation-SimData"

    tune_llm: bool = False            # Whether to tune the LLM
    tune_visual: bool = True          # Whether to tune the visual encoder
    tune_projector: bool = True       # Whether to tune the projector
    tune_diffusion_model: bool = True # Whether to tune the diffusion model

    output_dir: str = "/home/anpengju/runs/gr00t_challenge"    # CHANGE THIS ACCORDING TO YOUR LOCAL PATH
    per_device_train_batch_size: int = 8     # CHANGE THIS ACCORDING TO YOUR GPU MEMORY
    max_steps: int = 5000                      # CHANGE THIS ACCORDING TO YOUR NEEDS
    report_to: str = "wandb"    # or tensorboard or none
    dataloader_num_workers: int = 8
    window_size: int = 30

    data_config: str = "agibot_genie1"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    debug: bool = False

    num_gpus: int = 1

def finetune(cfg: FinetuneConfig) -> None:

    # cfg.dataset_path = cfg.dataset_path + cfg.dataset_name

    data_config_cls = DATA_CONFIG_MAP[cfg.data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # image_processor = PrismaticImageProcessor(
    #     use_fused_vision_backbone=True,
    #     image_resize_strategy='resize-naive',
    #     interpolations=['bicubic']
    # )

    # train_set = {}
    # train_set[cfg.dataset_name] = {
    #     "use_cam_list": ["head", "hand_right", "hand_left"],
    #     "label_file_name": f"train.json",
    # }

    from gr00t.data.agibot_dataset import A2dDataset
    dataset_args = a2d_cfg.DatasetArguments(
        meta_json_dir=cfg.dataset_path,
        data_root_dir=cfg.dataset_path,
        # dataset_task_cfg=train_set
    )
    data_training_args = a2d_cfg.DataTrainingArguments(force_image_size=224)
    ActionSpacePadder = a2d_cfg.ActionSpacePadderArguments()

    text_tokenizer = AutoTokenizer.from_pretrained(
        "OpenGVLab/InternVL2-2B",
        trust_remote_code=True,
        add_eos_token=False,
    )

    text_tokenizer.model_max_length = 4096
    embodiment_tag = EmbodimentTag("agibot_genie1")

    vla_dataset = A2dDataset(
        # base parmas
        label_file_dir=dataset_args.meta_json_dir, 
        data_root_dir=dataset_args.data_root_dir, 
        # valid_episode_txt=dataset_args.valid_episode_txt, 
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
        min_window_size=cfg.window_size, 
        max_window_size=cfg.window_size + 1, 
        # image_transform=image_processor.apply_transform, 
        embodiment_tag=embodiment_tag,
        transforms=transforms,
        # modality_configs=modality_configs
    )

    vla_dataset.generate_task_infos(
        # dataset_args.dataset_task_cfg,
        task_episode_processors_cfg=dataset_args.episode_processors,
        task_dataset_processors_cfg=dataset_args.dataset_processors,
        task_runtime_processors_cfg=dataset_args.runtime_processors,
        shuffle=True,
        statistic=True,
        debug_one_episode=cfg.debug,
        # debug_one_episode=False,
    )

    # train_dataset = LeRobotSingleDataset(
    #     dataset_path=dataset_path,
    #     modality_configs=modality_configs,
    #     embodiment_tag=embodiment_tag,
    #     video_backend="torchvision_av",
    #     transforms=to_apply_transforms,
    # )

    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=cfg.vla_path,
        tune_llm=cfg.tune_llm,  # backbone's LLM
        tune_visual=cfg.tune_visual,  # backbone's vision tower
        tune_projector=cfg.tune_projector,  # action head's projector
        tune_diffusion_model=cfg.tune_diffusion_model,  # action head's DiT
    )

    # Set the model's compute_dtype to bfloat16
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"
    model.to(cfg.device)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=cfg.dataloader_num_workers,
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
        max_steps=cfg.max_steps,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=8,
        report_to=cfg.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )


    experiment = TrainRunner(
        train_dataset=vla_dataset,
        model=model,
        training_args=training_args,
    )

    experiment.train()

if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(FinetuneConfig)

    # Print the tyro config
    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        # Single GPU mode - set CUDA_VISIBLE_DEVICES=0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Run the script normally
        finetune(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            finetune(config)
        else:
            # Multi-GPU mode - use torchrun
            script_path = Path(__file__).absolute()
            # Remove any existing CUDA_VISIBLE_DEVICES from environment
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            # Use subprocess.run instead of os.system
            cmd = [     
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",  # default to 1 node for now
                str(script_path),
            ]

            # Convert config to command line arguments
            for key, value in vars(config).items():
                if isinstance(value, bool):
                    # For boolean values, use --flag or --no-flag format
                    if value:
                        cmd.append(f"--{key.replace('_', '-')}")
                    else:
                        cmd.append(f"--no-{key.replace('_', '-')}")
                else:
                    # For non-boolean values, use --key value format
                    cmd.append(f"--{key.replace('_', '-')}")

                    # if the value is a list (e.g. dataset_path), we need to add each element in the list
                    if isinstance(value, list):
                        for v in value:
                            cmd.append(str(v))
                    else:
                        cmd.append(str(value))
            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)