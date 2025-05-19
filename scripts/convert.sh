DATA_DIR=/share/project/hejingyang/G1Data/G1Data_zuoshou_4x25
ROBOT_TYPE=baai_G1_Dex3_3_camera
TASK="Pick_up_the_left_bread_and_put_it_into_the_plate"
REPO_ID=yuheng2000/Pick_up_the_left_bread_and_put_it_into_the_plate_zuoshou_4position_total100
DIR_NAME=Pick_up_the_left_bread_and_put_it_into_the_plate_zuoshou_4position_total100
INST="pick up the left bread and put it into the plate."
CACHE_DIR=/root/.cache/huggingface/lerobot/$REPO_ID
TARGET_DIR=/share/project/jiyuheng/Isaac-GR00T/demo_data

# conda activate /share/project/jiyuheng/env/unitree_lerobot
cd /share/project/jiyuheng/unitree_IL_lerobot

python unitree_lerobot/utils/sort_and_rename_folders.py --data_dir $DATA_DIR

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir $DATA_DIR \
    --robot_type $ROBOT_TYPE \
    --task "$TASK" \
    --no-push-to-hub \
    --repo-id $REPO_ID

mv $CACHE_DIR $TARGET_DIR


python /share/project/jiyuheng/Isaac-GR00T/scripts/convert_task_inst.py \
    --folder_path $TARGET_DIR/$DIR_NAME \
    --task_name "$INST"