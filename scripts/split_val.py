import json
import random
import os

task_names = os.listdir('/home/anpengju/AgiBot-World/Manipulation-SimData/')

# print(task_names)

for task_name in task_names:

    prefix = f"/home/anpengju/AgiBot-World/Manipulation-SimData/{task_name}"

    # 1. 读取原始数据
    with open(f'{prefix}/task_train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 打乱顺序
    random.shuffle(data)

    # 3. 按比例分割
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # 4. 写入train.json
    with open(f'{prefix}/train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    # 5. 写入val.json
    with open(f'{prefix}/val.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"train.json: {len(train_data)} samples")
    print(f"val.json: {len(val_data)} samples")
