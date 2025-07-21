source /root/miniconda3/activate /home/anpengju/envs/gr00t

cd /home/anpengju/Isaac-GR00T-challenge

export PYTHONPATH=/home/anpengju/Isaac-GR00T-challenge

python scripts/finetune_genie.py \
    --vla_path nvidia/GR00T-N1.5-3B \
    --dataset_path /robot/embodied-perception-data/user/hb/data/Manipulation-SimData \
    --tune_llm False \
    --tune_visual True \
    --tune_projector True \
    --tune_diffusion_model True \
    --output_dir /home/anpengju/runs/gr00t_challenge \
    --per_device_train_batch_size 8 \
    --max_steps 5000 \
    --report_to none \
    --dataloader_num_workers 8 \
    --window_size 30 \
    --data_config agibot_genie1 \
