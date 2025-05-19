from pathlib import Path
import shutil

parent_folder = "/share/project/hejingyang/G1Data/G1Data_zuoshou_4x25"
root_dir = Path(parent_folder)

# 获取一级子文件夹名称（如 B, C, D）
subfolder_names = [f.name for f in root_dir.iterdir() if f.is_dir()]

for folder_name in subfolder_names:
    parent_path = root_dir / folder_name
    prefix = folder_name + '_'

    if not parent_path.exists():
        print(f"路径 {parent_path} 不存在")
        continue

    # 第一步：重命名 parent_path 下的所有子文件夹
    for child in parent_path.iterdir():
        if child.is_dir():
            new_name = prefix + child.name
            new_path = child.parent / new_name
            if new_path.exists():
                print(f"目标路径 {new_path} 已存在，跳过: {child}")
                continue
            child.rename(new_path)
            print(f"重命名: {child} -> {new_path}")

    # 第二步：将 parent_path 中的所有内容（文件+文件夹）移动到 root_dir 下
    for item in parent_path.iterdir():
        dest = root_dir / item.name

        # 如果目标已存在，加后缀防止覆盖
        counter = 1
        while dest.exists():
            dest = root_dir / f"{item.stem}_{counter}{item.suffix}"
            counter += 1

        shutil.move(str(item), str(dest))
        print(f"移动: {item} -> {dest}")

    # 第三步：删除 now empty 的 parent_path 文件夹
    try:
        parent_path.rmdir()
        print(f"✅ 删除空文件夹: {parent_path}")
    except OSError as e:
        print(f"⚠️ 文件夹非空，未删除 {parent_path}: {e}")