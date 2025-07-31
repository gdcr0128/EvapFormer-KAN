import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import pandas as pd

# 加载数据函数
def load_npz_files(folder_path, key_name):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])
    loaded_files = []
    for file in files:
        data = np.load(os.path.join(folder_path, file), allow_pickle=True)
        loaded_files.append(data[key_name])
    return loaded_files

# 提取所有头部对最后一个输出token的注意力权重平均值函数
def get_avg_attn_across_heads(attn_files):
    all_attn_across_heads = np.concatenate(
        [file.mean(axis=1) for file in attn_files], axis=0
    )
    avg_attn_across_heads = np.mean(all_attn_across_heads, axis=0)
    return avg_attn_across_heads

# 可视化函数
def visualize_attention(avg_attn, title, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_attn, cmap='viridis', annot=False)
    plt.title(title)
    plt.xlabel("Input Tokens")
    plt.ylabel("Output Seq")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 特征排序函数，使用从CSV文件中加载的标签名称
def print_feature_ranking(avg_attn, labels):
    attn_weights = avg_attn.mean(axis=0)
    sorted_indices = np.argsort(-attn_weights)
    print("特征排序（按平均注意力权重大小）:")
    for i in sorted_indices:
        # 使用min函数确保索引i不会超出labels列表的范围
        print(f"{labels[min(i, len(labels)-1)]}: {attn_weights[i]:.4f}")

# 清空文件夹内容
def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# 从CSV文件的第一行加载标签并排除第一个标签的函数
def load_labels_from_csv_first_row(csv_path):
    df = pd.read_csv(csv_path, nrows=0)  # 只读取第一行
    labels = df.columns.tolist()[1:]  # 排除第一个标签
    return labels

# 主执行函数
def main():
    attn_folder = '../attns'  # 注意力权重文件夹路径
    attn_key = 'attn'  # 从上面检查中我们得到的正确的键名
    csv_path = '../data/ERA5.csv'  # 替换为你的CSV数据集文件路径

    attn_files = load_npz_files(attn_folder, attn_key)  # 加载注意力权重文件
    labels = load_labels_from_csv_first_row(csv_path)  # 从CSV文件的第一行加载标签并排除第一个

    avg_attn_across_heads = get_avg_attn_across_heads(attn_files)

    visualize_attention(avg_attn_across_heads, "Average Attention Weights Across All Heads for Last Token", "average_attention_weights_across_heads.png")
    print_feature_ranking(avg_attn_across_heads, labels)
    clear_folder(attn_folder)

# 执行主函数
main()
