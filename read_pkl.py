import pickle

# 读取 .pkl 文件的函数
def read_pkl_file(file_path):
    try:
        # 打开文件并加载内容
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到！")
    except pickle.UnpicklingError:
        print("文件内容无法反序列化，可能不是有效的 .pkl 文件！")

# 示例用法
file_path = "/robot/embodied-perception-data/user/hb/data/Manipulation-SimData/ann_file/all_ann.pkl"  # 替换为你的 .pkl 文件路径
data = read_pkl_file(file_path)

if data is not None:
    print("读取到的数据：", data)
