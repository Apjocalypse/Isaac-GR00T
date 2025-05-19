cd /share/project/jiyuheng/Isaac-GR00T
export PYTHONPATH=$(pwd)

BASE_MODEL=/share/project/jiyuheng/groot_n1
DATASET=/share/project/jiyuheng/Isaac-GR00T/demo_data/Pick_up_the_left_bread_and_put_it_into_the_plate_zuoshou_4position_total100
EXP_NAME=num8_1_G1_Dex3_bread_zuoshou4_25
OUTPUT_DIR=/share/project/jiyuheng/n1_ckpt/${EXP_NAME}
ROBOT_CONFIG=baai_g1_dex3

export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_PROJECT=GR00T-N1-Reproduction
export WANDB_API_KEY="327df07cca0b4918195078945539c819acc6ac97"
export WANDB_RUN_NAME=${EXP_NAME}-$(date +%Y-%m-%d-%H-%M-%S)
wandb login $WANDB_API_KEY
# export WANDB_MODE=offline

export NO_ALBUMENTATIONS_UPDATE=1

TUNE_LLM="--no-tune-llm"
TUNE_VISUAL="--no-tune-visual"
TUNE_PROJECTOR="--tune-projector"
TUNE_DIFFUSION_MODEL="--tune-diffusion-model"

MAX_STEP=50000
NUM_GPU=4

BATCHSIZE=16
SAVE_STEP=10000
LR=0.0001
WEIGHT_DECAY=1e-05
WARMUP=0.05

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir "$OUTPUT_DIR"
fi

nohup python /share/project/jiyuheng/Isaac-GR00T/scripts/gr00t_finetune.py \
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
    --embodiment-tag new_embodiment \
    > "$OUTPUT_DIR/train_log.txt" 2>&1 &