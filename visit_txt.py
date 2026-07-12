# import sys
#
# def view_txt(file_path, head_num=20, tail_num=10):
#     # 统计总行数
#     total_lines = 0
#     all_lines = []
#     with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
#         for line in f:
#             line = line.strip()
#             all_lines.append(line)
#             total_lines += 1
#     print("="*60)
#     print(f"文件路径: {file_path}")
#     print(f"总行数: {total_lines:,}")
#     print("="*60)
#
#     # 打印前N行
#     print(f"\n【前 {head_num} 行内容】")
#     for idx, line in enumerate(all_lines[:head_num]):
#         print(f"行{idx+1}: {line}")
#
#     # 打印后N行
#     if total_lines > head_num:
#         print(f"\n【后 {tail_num} 行内容】")
#         start = max(head_num, total_lines - tail_num)
#         for idx in range(start, total_lines):
#             print(f"行{idx+1}: {all_lines[idx]}")
#
#     # 抽样打印中间5行（超大文件专用）
#     if total_lines > head_num + tail_num + 10:
#         print("\n【中间随机抽样5行】")
#         step = total_lines // 5
#         for i in range(1, 6):
#             pos = i * step
#             print(f"行{pos}: {all_lines[pos]}")
#
#     # 查看单行字段分隔（判断制表符/逗号分隔）
#     if all_lines:
#         sample = all_lines[0]
#         if "\t" in sample:
#             split_data = sample.split("\t")
#             print(f"\n【字段分隔符：制表符\\t，一行字段总数：{len(split_data)}】")
#             print("字段示例：", split_data)
#         elif "," in sample:
#             split_data = sample.split(",")
#             print(f"\n【字段分隔符：逗号,，一行字段总数：{len(split_data)}】")
#             print("字段示例：", split_data)
#
# if __name__ == "__main__":
#     # 用法1：直接指定文件名
#     if len(sys.argv) >= 2:
#         txt_file = sys.argv[1]
#         txt_file='checkins-gowalla_semantic.pkl'
#         view_txt(txt_file, head_num=30, tail_num=10)
#     else:
#         # 无参数时提示使用方法
#         print("使用方式：python view_txt.py 你的文件.txt")
#         print("示例：python view_txt.py loc-gowalla_totalCheckins.txt")


from collections import defaultdict
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

input_file = "data/checkins-gowalla.txt"
output_file = "data/poi_semantics_gowalla_behavior.txt"

num_semantics = 100

poi_stats = {}

def get_item(poi_id):
    if poi_id not in poi_stats:
        poi_stats[poi_id] = {
            "lat": 0.0,
            "lng": 0.0,
            "count": 0,
            "hour_hist": np.zeros(24, dtype=np.float32),
            "weekend": 0,
        }
    return poi_stats[poi_id]

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue

        user_id = parts[0]
        time_str = parts[1]
        lat = float(parts[2])
        lng = float(parts[3])
        poi_id = parts[4]

        try:
            dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue

        item = get_item(poi_id)
        item["lat"] = lat
        item["lng"] = lng
        item["count"] += 1
        item["hour_hist"][dt.hour] += 1
        if dt.weekday() >= 5:
            item["weekend"] += 1

poi_ids = []
features = []

for poi_id, item in poi_stats.items():
    count = max(item["count"], 1)

    hour_hist = item["hour_hist"] / count
    weekend_ratio = item["weekend"] / count
    log_count = np.log1p(count)

    feat = np.concatenate([
        np.array([
            item["lat"],
            item["lng"],
            weekend_ratio,
            log_count
        ], dtype=np.float32),
        hour_hist
    ])

    poi_ids.append(poi_id)
    features.append(feat)

features = np.vstack(features)
features = StandardScaler().fit_transform(features)

kmeans = KMeans(
    n_clusters=num_semantics,
    random_state=0,
    n_init=10
)

labels = kmeans.fit_predict(features)

with open(output_file, "w", encoding="utf-8") as f:
    for poi_id, sem_id in zip(poi_ids, labels):
        f.write(f"{poi_id}\t{sem_id}\n")

print("POI number:", len(poi_ids))
print("semantic number:", num_semantics)
print("saved:", output_file)