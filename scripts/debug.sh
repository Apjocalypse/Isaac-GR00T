# conda activate /home/jiyuheng/env/gr00t/

BASE_MODEL=/home/jiyuheng/groot_n1
DATASET=/home/robot_data/pi0_data/agilex/HuaihaiLyu/groceries
EXP_NAME=agi-1_agilex_groceries
OUTPUT_DIR=/home/jiyuheng/n1_ckpt/${EXP_NAME}
ROBOT_CONFIG=aloha

export NO_ALBUMENTATIONS_UPDATE=1

export http_proxy=http://192.168.0.3:1080
export https_proxy=http://192.168.0.3:1080
export WANDB_MODE=offline

TUNE_LLM="--no-tune-llm"
TUNE_VISUAL="--tune-visual"
TUNE_PROJECTOR="--tune-projector"
TUNE_DIFFUSION_MODEL="--tune-diffusion-model"

MAX_STEP=5000
NUM_GPU=1

BATCHSIZE=16
SAVE_STEP=500
LR=0.0001
WEIGHT_DECAY=1e-05
WARMUP=0.05

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir "$OUTPUT_DIR"
fi

# CUDA_VISIBLE_DEVICES=7

CUDA_VISIBLE_DEVICES=7 python /home/jiyuheng/Isaac-GR00T/scripts/gr00t_finetune.py \
    --data_config $ROBOT_CONFIG \
    --dataset-path $DATASET \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCHSIZE \
    --max-steps $MAX_STEP \
    --save-steps $SAVE_STEP \
    --num-gpus $NUM_GPU \
    --base-model-path $BASE_MODEL \
    $TUNE_LLM \
    $TUNE_VISUAL \
    $TUNE_PROJECTOR \
    $TUNE_DIFFUSION_MODEL \
    --no-resume \
    --learning-rate $LR \
    --weight-decay $WEIGHT_DECAY \
    --warmup-ratio $WARMUP \
    --dataloader-num-workers 8 \
    --embodiment-tag new_embodiment