DATA_DIR=/home/jiyuheng/Isaac-GR00T/demo_data/raw/G1_Dex3_grasp_apple_place_basket_push_forward
ROBOT_TYPE=baai_G1_Dex3
TASK="grasp_apple_place_basket_push_forward"
REPO_ID=yuheng2000/G1_Dex3_grasp_apple_place_basket_push_forward
CACHE_DIR=/root/.cache/huggingface/lerobot/$REPO_ID
TARGET_DIR=/home/jiyuheng/Isaac-GR00T/demo_data

conda activate /home/jiyuheng/env/unitree_lerobot
cd /home/jiyuheng/unitree_IL_lerobot

python unitree_lerobot/utils/sort_and_rename_folders.py --data_dir $DATA_DIR

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir $DATA_DIR \
    --robot_type $ROBOT_TYPE \
    --task "$TASK" \
    --no-push-to-hub \
    --repo-id $REPO_ID

mv $CACHE_DIR $TARGET_DIR