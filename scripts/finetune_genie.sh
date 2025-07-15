source /root/miniconda3/activate /home/anpengju/envs/gr00t

cd /home/anpengju/Isaac-GR00T-challenge

export PYTHONPATH=/home/anpengju/Isaac-GR00T-challenge

python scripts/finetune_genie.py \
    --dataset_name restock_supermarket_items