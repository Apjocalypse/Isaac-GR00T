import os
import json
import argparse

def modify_jsonl_file(folder_path: str, task_name: str) -> None:
    """
    用指定的task_name覆盖JSONL文件中每条记录的tasks字段
    
    Args:
        folder_path: 包含meta文件夹的路径
        task_name: 用于覆盖的任务名称字符串
    """
    # 构建完整的文件路径
    file_path = os.path.join(folder_path, "meta", "episodes.jsonl")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 读取并修改JSONL文件
    modified_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析JSON对象
            obj = json.loads(line)
            
            # 用task_name覆盖tasks字段
            obj["tasks"] = [task_name]
            
            # 添加修改后的行
            modified_lines.append(obj)
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        for obj in modified_lines:
            f.write(json.dumps(obj) + '\n')
    
    print(f"成功修改文件: {file_path}")
    print(f"共处理 {len(modified_lines)} 条记录")

def main():
    """脚本主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="用指定任务名称覆盖JSONL文件中的tasks字段")
    parser.add_argument("--folder_path", required=True, help="包含meta文件夹的路径")
    parser.add_argument("--task_name", required=True, help="用于覆盖的任务名称字符串")
    
    # 解析参数
    args = parser.parse_args()
    
    try:
        # 执行修改操作
        modify_jsonl_file(args.folder_path, args.task_name)
    except Exception as e:
        print(f"执行过程中发生错误: {e}")

if __name__ == "__main__":
    main()    

# python /share/project/jiyuheng/Isaac-GR00T/scripts/convert_task_inst.py --folder_path /share/project/jiyuheng/Isaac-GR00T/demo_data/Pick_up_the_left_bread_and_put_it_into_the_plate_zuoshou_4position_total100 --task_name "pick up the left bread and put it into the plate.