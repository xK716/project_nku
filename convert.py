# import os
# import pandas as pd
#
# input_csv = "data/dataset_TSMC2014_NYC.csv"
#
# out_checkins = "data/checkins-4sq-nyc.txt"
# out_offset = "data/checkins_4sq_nyc_time_offset.txt"
# out_semantic = "data/poi_semantics_4sq_nyc.txt"
#
# out_user_map = "data/user_id_map_4sq_nyc.csv"
# out_poi_map = "data/poi_id_map_4sq_nyc.csv"
# out_sem_map = "data/semantic_id_map_4sq_nyc.csv"
#
# if not os.path.exists(input_csv):
#     raise FileNotFoundError(f"找不到文件：{input_csv}")
#
# df = pd.read_csv(input_csv, encoding="utf-8")
#
# print("原始列名：")
# print(df.columns.tolist())
#
# # 兼容常见列名
# rename_map = {}
# for c in df.columns:
#     c_strip = c.strip()
#     rename_map[c] = c_strip
# df = df.rename(columns=rename_map)
#
# required_cols = [
#     "userId",
#     "venueId",
#     "venueCategory",
#     "latitude",
#     "longitude",
#     "timezoneOffset",
#     "utcTimestamp",
# ]
#
# for col in required_cols:
#     if col not in df.columns:
#         raise ValueError(f"缺少列：{col}，当前列名为：{df.columns.tolist()}")
#
# df = df.dropna(subset=required_cols).copy()
#
# # 原始 venueId 是字符串，原代码会 int(tokens[4])，所以必须重新映射为整数
# df["user_raw"] = df["userId"].astype(str)
# df["venue_raw"] = df["venueId"].astype(str)
# df["cat_raw"] = df["venueCategory"].astype(str)
#
# # 解析 UTC 时间。原始格式类似：Tue Apr 03 18:00:09 +0000 2012
# df["utc_dt"] = pd.to_datetime(df["utcTimestamp"], utc=True, errors="coerce")
# df = df.dropna(subset=["utc_dt"]).copy()
#
# # 重新编码 user / poi / semantic
# user_values = sorted(df["user_raw"].unique())
# venue_values = sorted(df["venue_raw"].unique())
# cat_values = sorted(df["cat_raw"].unique())
#
# user2id = {u: i for i, u in enumerate(user_values)}
# venue2id = {v: i for i, v in enumerate(venue_values)}
# cat2id = {c: i for i, c in enumerate(cat_values)}
#
# df["user_int"] = df["user_raw"].map(user2id)
# df["poi_int"] = df["venue_raw"].map(venue2id)
# df["sem_int"] = df["cat_raw"].map(cat2id)
#
# # 关键：DPRL 的 read_pois 假设同一用户连续，且文件内每个用户按时间倒序；
# # 代码内部会 insert(0)，所以这里按 user 升序、时间降序排序。
# df = df.sort_values(["user_int", "utc_dt"], ascending=[True, False])
#
# # 输出 checkins 文件和 offset 文件，二者必须逐行对应
# with open(out_checkins, "w", encoding="utf-8") as f_check, \
#      open(out_offset, "w", encoding="utf-8") as f_offset:
#
#     for _, row in df.iterrows():
#         time_str = row["utc_dt"].strftime("%Y-%m-%dT%H:%M:%SZ")
#
#         f_check.write(
#             f"{int(row['user_int'])}\t"
#             f"{time_str}\t"
#             f"{float(row['latitude'])}\t"
#             f"{float(row['longitude'])}\t"
#             f"{int(row['poi_int'])}\n"
#         )
#
#         f_offset.write(f"{int(row['timezoneOffset'])}\n")
#
# # 每个 POI 一个语义类别。
# # 如果同一 POI 出现多个类别，取出现次数最多的类别。
# poi_sem = (
#     df.groupby("poi_int")["sem_int"]
#     .agg(lambda x: x.value_counts().idxmax())
#     .reset_index()
# )
#
# with open(out_semantic, "w", encoding="utf-8") as f:
#     for _, row in poi_sem.iterrows():
#         f.write(f"{int(row['poi_int'])}\t{int(row['sem_int'])}\n")
#
# # 保存映射表，方便论文说明类别含义
# pd.DataFrame({
#     "raw_user_id": list(user2id.keys()),
#     "user_int": list(user2id.values())
# }).to_csv(out_user_map, index=False, encoding="utf-8-sig")
#
# pd.DataFrame({
#     "raw_venue_id": list(venue2id.keys()),
#     "poi_int": list(venue2id.values())
# }).to_csv(out_poi_map, index=False, encoding="utf-8-sig")
#
# pd.DataFrame({
#     "venueCategory": list(cat2id.keys()),
#     "sem_int": list(cat2id.values())
# }).to_csv(out_sem_map, index=False, encoding="utf-8-sig")
#
# print("转换完成")
# print("checkins:", out_checkins)
# print("offset:", out_offset)
# print("semantic:", out_semantic)
# print("user 数:", df["user_int"].nunique())
# print("POI 数:", df["poi_int"].nunique())
# print("semantic 类别数:", df["sem_int"].nunique())
# print("check-in 数:", len(df))



# import os
# import pandas as pd
#
# input_csv = "data/dataset_TSMC2014_TKY.csv"
#
# out_checkins = "data/checkins-4sq-tky.txt"
# out_offset = "data/checkins_4sq_tky_time_offset.txt"
# out_semantic = "data/poi_semantics_4sq_tky.txt"
#
# out_user_map = "data/user_id_map_4sq_tky.csv"
# out_poi_map = "data/poi_id_map_4sq_tky.csv"
# out_sem_map = "data/semantic_id_map_4sq_tky.csv"
#
# if not os.path.exists(input_csv):
#     raise FileNotFoundError(f"找不到文件：{input_csv}")
#
# df = pd.read_csv(input_csv, encoding="utf-8")
# df = df.rename(columns={c: c.strip() for c in df.columns})
#
# required_cols = [
#     "userId",
#     "venueId",
#     "venueCategory",
#     "latitude",
#     "longitude",
#     "timezoneOffset",
#     "utcTimestamp",
# ]
#
# for col in required_cols:
#     if col not in df.columns:
#         raise ValueError(f"缺少列：{col}，当前列名为：{df.columns.tolist()}")
#
# df = df.dropna(subset=required_cols).copy()
#
# df["user_raw"] = df["userId"].astype(str)
# df["venue_raw"] = df["venueId"].astype(str)
# df["cat_raw"] = df["venueCategory"].astype(str)
#
# df["utc_dt"] = pd.to_datetime(df["utcTimestamp"], utc=True, errors="coerce")
# df = df.dropna(subset=["utc_dt"]).copy()
#
# user_values = sorted(df["user_raw"].unique())
# venue_values = sorted(df["venue_raw"].unique())
# cat_values = sorted(df["cat_raw"].unique())
#
# user2id = {u: i for i, u in enumerate(user_values)}
# venue2id = {v: i for i, v in enumerate(venue_values)}
# cat2id = {c: i for i, c in enumerate(cat_values)}
#
# df["user_int"] = df["user_raw"].map(user2id)
# df["poi_int"] = df["venue_raw"].map(venue2id)
# df["sem_int"] = df["cat_raw"].map(cat2id)
#
# # DPRL 要求同一用户连续；这里按 user 升序、时间降序排
# df = df.sort_values(["user_int", "utc_dt"], ascending=[True, False])
#
# with open(out_checkins, "w", encoding="utf-8") as f_check, \
#      open(out_offset, "w", encoding="utf-8") as f_offset:
#
#     for _, row in df.iterrows():
#         time_str = row["utc_dt"].strftime("%Y-%m-%dT%H:%M:%SZ")
#
#         f_check.write(
#             f"{int(row['user_int'])}\t"
#             f"{time_str}\t"
#             f"{float(row['latitude'])}\t"
#             f"{float(row['longitude'])}\t"
#             f"{int(row['poi_int'])}\n"
#         )
#
#         f_offset.write(f"{int(row['timezoneOffset'])}\n")
#
# poi_sem = (
#     df.groupby("poi_int")["sem_int"]
#     .agg(lambda x: x.value_counts().idxmax())
#     .reset_index()
# )
#
# with open(out_semantic, "w", encoding="utf-8") as f:
#     for _, row in poi_sem.iterrows():
#         f.write(f"{int(row['poi_int'])}\t{int(row['sem_int'])}\n")
#
# pd.DataFrame({
#     "raw_user_id": list(user2id.keys()),
#     "user_int": list(user2id.values())
# }).to_csv(out_user_map, index=False, encoding="utf-8-sig")
#
# pd.DataFrame({
#     "raw_venue_id": list(venue2id.keys()),
#     "poi_int": list(venue2id.values())
# }).to_csv(out_poi_map, index=False, encoding="utf-8-sig")
#
# pd.DataFrame({
#     "venueCategory": list(cat2id.keys()),
#     "sem_int": list(cat2id.values())
# }).to_csv(out_sem_map, index=False, encoding="utf-8-sig")
#
# print("转换完成")
# print("checkins:", out_checkins)
# print("offset:", out_offset)
# print("semantic:", out_semantic)
# print("user 数:", df["user_int"].nunique())
# print("POI 数:", df["poi_int"].nunique())
# print("semantic 类别数:", df["sem_int"].nunique())
# print("check-in 数:", len(df))





# import os
# import json
# import argparse
# from datetime import datetime
# from collections import defaultdict, Counter
# import pandas as pd
#
#
# def parse_args():
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument(
#         "--business-file",
#         default="data/yelp/yelp_academic_dataset_business.json",
#         type=str,
#         help="Yelp business.json path"
#     )
#
#     parser.add_argument(
#         "--review-file",
#         default="data/yelp/yelp_academic_dataset_review.json",
#         type=str,
#         help="Yelp review.json path"
#     )
#
#     parser.add_argument(
#         "--out-dir",
#         default="data",
#         type=str,
#         help="Output directory"
#     )
#
#     parser.add_argument(
#         "--min-user-records",
#         default=101,
#         type=int,
#         help="Minimum records per user. DPRL default min_checkins is 101."
#     )
#
#     parser.add_argument(
#         "--city",
#         default=None,
#         type=str,
#         help="Optional city filter, e.g., Philadelphia, Tampa, Las Vegas. Default: use all cities."
#     )
#
#     parser.add_argument(
#         "--max-records",
#         default=0,
#         type=int,
#         help="Optional max filtered records for debugging. 0 means no limit."
#     )
#
#     parser.add_argument(
#         "--category-mode",
#         default="first",
#         choices=["first", "top"],
#         help="How to choose Yelp category. first: use first category string; top: use most frequent category per business."
#     )
#
#     return parser.parse_args()
#
#
# def check_file(path):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"File not found: {path}")
#
#     if path.endswith(".zip"):
#         raise RuntimeError(
#             f"{path} looks like a zip file. Please unzip it first and use the .json file."
#         )
#
#     return True
#
#
# def safe_json_loads(line):
#     try:
#         return json.loads(line)
#     except Exception:
#         return None
#
#
# def parse_yelp_time(date_str):
#     # Yelp review date format is usually: "YYYY-MM-DD HH:MM:SS"
#     try:
#         return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
#     except Exception:
#         return None
#
#
# def main():
#     args = parse_args()
#
#     business_file = args.business_file
#     review_file = args.review_file
#     out_dir = args.out_dir
#
#     check_file(business_file)
#     check_file(review_file)
#     os.makedirs(out_dir, exist_ok=True)
#
#     out_checkins = os.path.join(out_dir, "checkins-4sq-yelp.txt")
#     out_offset = os.path.join(out_dir, "checkins_4sq_yelp_time_offset.txt")
#     out_semantic = os.path.join(out_dir, "poi_semantics_yelp.txt")
#
#     out_user_map = os.path.join(out_dir, "user_id_map_yelp.csv")
#     out_poi_map = os.path.join(out_dir, "poi_id_map_yelp.csv")
#     out_sem_map = os.path.join(out_dir, "semantic_id_map_yelp.csv")
#     out_city_stat = os.path.join(out_dir, "yelp_city_stat.csv")
#
#     print("=" * 80)
#     print("Step 1: Loading business.json")
#     print("=" * 80)
#
#     business_info = {}
#     city_counter = Counter()
#     category_counter = Counter()
#
#     with open(business_file, "r", encoding="utf-8") as f:
#         for line in f:
#             obj = safe_json_loads(line)
#             if obj is None:
#                 continue
#
#             bid = obj.get("business_id")
#             lat = obj.get("latitude")
#             lng = obj.get("longitude")
#             cats = obj.get("categories")
#             city = obj.get("city")
#
#             if bid is None or lat is None or lng is None or cats is None:
#                 continue
#
#             city = str(city).strip() if city is not None else "unknown"
#
#             if args.city is not None:
#                 if city.lower() != args.city.lower():
#                     continue
#
#             cat_list = [c.strip() for c in str(cats).split(",") if c.strip()]
#             if len(cat_list) == 0:
#                 continue
#
#             if args.category_mode == "first":
#                 main_cat = cat_list[0]
#             else:
#                 # For a single business, Yelp does not provide category frequency.
#                 # Here top mode still temporarily uses first; real top category is derived later if needed.
#                 main_cat = cat_list[0]
#
#             business_info[bid] = {
#                 "lat": float(lat),
#                 "lng": float(lng),
#                 "city": city,
#                 "category": main_cat,
#                 "all_categories": cat_list,
#             }
#
#             city_counter[city] += 1
#             category_counter[main_cat] += 1
#
#     print(f"Valid businesses: {len(business_info)}")
#     print(f"Raw semantic categories: {len(category_counter)}")
#
#     if len(business_info) == 0:
#         raise RuntimeError("No valid businesses loaded. Check business.json or city filter.")
#
#     pd.DataFrame(
#         [{"city": c, "business_count": n} for c, n in city_counter.most_common()]
#     ).to_csv(out_city_stat, index=False, encoding="utf-8-sig")
#
#     print(f"City statistics saved to: {out_city_stat}")
#     print("Top 10 cities:")
#     for c, n in city_counter.most_common(10):
#         print(f"  {c}: {n}")
#
#     print("=" * 80)
#     print("Step 2: Counting valid review records per user")
#     print("=" * 80)
#
#     user_count = defaultdict(int)
#     total_reviews = 0
#     valid_review_business = 0
#
#     with open(review_file, "r", encoding="utf-8") as f:
#         for line in f:
#             total_reviews += 1
#             obj = safe_json_loads(line)
#             if obj is None:
#                 continue
#
#             uid = obj.get("user_id")
#             bid = obj.get("business_id")
#             date = obj.get("date")
#
#             if uid is None or bid is None or date is None:
#                 continue
#
#             if bid not in business_info:
#                 continue
#
#             dt = parse_yelp_time(date)
#             if dt is None:
#                 continue
#
#             valid_review_business += 1
#             user_count[uid] += 1
#
#     valid_users = {u for u, c in user_count.items() if c >= args.min_user_records}
#
#     print(f"Total reviews scanned: {total_reviews}")
#     print(f"Reviews with valid business: {valid_review_business}")
#     print(f"Users before filtering: {len(user_count)}")
#     print(f"Valid users with >= {args.min_user_records} records: {len(valid_users)}")
#
#     if len(valid_users) == 0:
#         raise RuntimeError(
#             "No valid users after filtering. "
#             "Try lower threshold, e.g. --min-user-records 50 or --min-user-records 20."
#         )
#
#     print("=" * 80)
#     print("Step 3: Building filtered records")
#     print("=" * 80)
#
#     records = []
#     used_poi_set = set()
#     used_cat_set = set()
#
#     with open(review_file, "r", encoding="utf-8") as f:
#         for line in f:
#             obj = safe_json_loads(line)
#             if obj is None:
#                 continue
#
#             uid = obj.get("user_id")
#             bid = obj.get("business_id")
#             date = obj.get("date")
#
#             if uid not in valid_users:
#                 continue
#
#             if bid not in business_info:
#                 continue
#
#             dt = parse_yelp_time(date)
#             if dt is None:
#                 continue
#
#             b = business_info[bid]
#             cat = b["category"]
#
#             records.append({
#                 "user_raw": uid,
#                 "poi_raw": bid,
#                 "dt": dt,
#                 "lat": b["lat"],
#                 "lng": b["lng"],
#                 "category": cat,
#                 "city": b["city"],
#             })
#
#             used_poi_set.add(bid)
#             used_cat_set.add(cat)
#
#             if args.max_records > 0 and len(records) >= args.max_records:
#                 break
#
#     print(f"Filtered records: {len(records)}")
#     print(f"Used POIs: {len(used_poi_set)}")
#     print(f"Used semantic categories: {len(used_cat_set)}")
#
#     if len(records) == 0:
#         raise RuntimeError("No records generated. Try reducing --min-user-records.")
#
#     print("=" * 80)
#     print("Step 4: Mapping raw IDs to integer IDs")
#     print("=" * 80)
#
#     df = pd.DataFrame(records)
#
#     user_values = sorted(df["user_raw"].unique())
#     poi_values = sorted(df["poi_raw"].unique())
#     cat_values = sorted(df["category"].unique())
#
#     user2id = {u: i for i, u in enumerate(user_values)}
#     poi2id = {p: i for i, p in enumerate(poi_values)}
#     cat2id = {c: i for i, c in enumerate(cat_values)}
#
#     df["user_int"] = df["user_raw"].map(user2id)
#     df["poi_int"] = df["poi_raw"].map(poi2id)
#     df["sem_int"] = df["category"].map(cat2id)
#
#     # DPRL dataloader expects the same user's records to be continuous.
#     # The original code inserts each check-in at the front, so we sort by descending time.
#     df = df.sort_values(["user_int", "dt"], ascending=[True, False])
#
#     print(f"Final users: {df['user_int'].nunique()}")
#     print(f"Final POIs: {df['poi_int'].nunique()}")
#     print(f"Final semantic classes: {df['sem_int'].nunique()}")
#     print(f"Final records: {len(df)}")
#
#     print("=" * 80)
#     print("Step 5: Writing DPRL format files")
#     print("=" * 80)
#
#     with open(out_checkins, "w", encoding="utf-8") as f_check, \
#          open(out_offset, "w", encoding="utf-8") as f_offset:
#
#         for _, row in df.iterrows():
#             time_str = row["dt"].strftime("%Y-%m-%dT%H:%M:%SZ")
#
#             f_check.write(
#                 f"{int(row['user_int'])}\t"
#                 f"{time_str}\t"
#                 f"{float(row['lat'])}\t"
#                 f"{float(row['lng'])}\t"
#                 f"{int(row['poi_int'])}\n"
#             )
#
#             # Yelp review data does not provide precise timezone per review.
#             # Use 0 as UTC offset.
#             f_offset.write("0\n")
#
#     poi_sem = (
#         df.groupby("poi_int")["sem_int"]
#         .agg(lambda x: x.value_counts().idxmax())
#         .reset_index()
#     )
#
#     with open(out_semantic, "w", encoding="utf-8") as f:
#         for _, row in poi_sem.iterrows():
#             f.write(f"{int(row['poi_int'])}\t{int(row['sem_int'])}\n")
#
#     pd.DataFrame({
#         "raw_user_id": list(user2id.keys()),
#         "user_int": list(user2id.values())
#     }).to_csv(out_user_map, index=False, encoding="utf-8-sig")
#
#     pd.DataFrame({
#         "raw_business_id": list(poi2id.keys()),
#         "poi_int": list(poi2id.values())
#     }).to_csv(out_poi_map, index=False, encoding="utf-8-sig")
#
#     pd.DataFrame({
#         "category": list(cat2id.keys()),
#         "sem_int": list(cat2id.values())
#     }).to_csv(out_sem_map, index=False, encoding="utf-8-sig")
#
#     print("Done.")
#     print(f"Saved checkins: {out_checkins}")
#     print(f"Saved offset: {out_offset}")
#     print(f"Saved semantic file: {out_semantic}")
#     print(f"Saved user map: {out_user_map}")
#     print(f"Saved POI map: {out_poi_map}")
#     print(f"Saved semantic map: {out_sem_map}")
#
#     print("=" * 80)
#     print("Next commands")
#     print("=" * 80)
#     print("Delete old cache first:")
#     print("rm -f data/4sq_3000_48.pickle")
#     print("rm -f data/4sq_graph.pickle")
#     print()
#     print("Run baseline:")
#     print(
#         "python train.py "
#         "--dataset checkins-4sq-yelp.txt "
#         "--offset checkins_4sq_yelp_time_offset.txt "
#         "--hidden-dim 30 "
#         "--weight_decay 1e-5 "
#         "--gpu 0 "
#         "--validate-epoch 5"
#     )
#     print()
#     print("Run Sem-DPRL:")
#     print(
#         "python train.py "
#         "--dataset checkins-4sq-yelp.txt "
#         "--offset checkins_4sq_yelp_time_offset.txt "
#         "--hidden-dim 30 "
#         "--weight_decay 1e-5 "
#         "--gpu 0 "
#         "--validate-epoch 5 "
#         "--use-sem-branch "
#         "--filter-candidate "
#         "--semantic-file poi_semantics_yelp.txt "
#         "--sem-weight 0.2 "
#         "--topm-candidate 5000 "
#         "--semantic-fusion-weight 0.0 "
#         "--sem-label-smoothing 0.05 "
#         "--sem-warmup-epochs 5 "
#         "--candidate-mode score "
#         "--candidate-sem-weight 0.1 "
#         "--candidate-region-weight 0.05"
#     )
#
#
# if __name__ == "__main__":
#     main()




import argparse
from collections import Counter, defaultdict


def read_poi_semantic(path):
    poi2sem = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            poi = int(parts[0])
            sem = int(parts[1])
            poi2sem[poi] = sem

    return poi2sem


def count_semantic_by_checkins(checkin_file, poi2sem):
    """
    按 check-in 频次统计每个 semantic 类别的重要性。
    比单纯按 POI 数量统计更合理，因为高频访问类别更重要。
    """
    sem_counter = Counter()
    poi_counter = Counter()

    with open(checkin_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            poi = int(parts[4])
            poi_counter[poi] += 1

            if poi in poi2sem:
                sem_counter[poi2sem[poi]] += 1

    return sem_counter, poi_counter


def merge_semantic_topk(checkin_file, semantic_file, output_file, map_file, top_k):
    poi2sem = read_poi_semantic(semantic_file)
    sem_counter, poi_counter = count_semantic_by_checkins(checkin_file, poi2sem)

    print("Original semantic classes:", len(sem_counter))
    print("Original POIs with semantic:", len(poi2sem))

    # 取 check-in 频次最高的 top_k 个语义类别
    top_semantics = [sem for sem, _ in sem_counter.most_common(top_k)]
    top_semantic_set = set(top_semantics)

    # 重新映射为连续 ID：0,1,2,...
    old2new = {}
    for new_id, old_sem in enumerate(top_semantics):
        old2new[old_sem] = new_id

    other_id = len(top_semantics)

    print("Keep top semantic classes:", len(top_semantics))
    print("Other semantic id:", other_id)

    # 输出新的 poi_semantics 文件
    kept_poi = 0
    other_poi = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for poi in sorted(poi2sem.keys()):
            old_sem = poi2sem[poi]

            if old_sem in top_semantic_set:
                new_sem = old2new[old_sem]
                kept_poi += 1
            else:
                new_sem = other_id
                other_poi += 1

            f.write(f"{poi}\t{new_sem}\n")

    # 输出 old semantic 到 new semantic 的映射，方便论文说明
    with open(map_file, "w", encoding="utf-8") as f:
        f.write("old_semantic_id,new_semantic_id,checkin_count,type\n")

        for old_sem in sorted(sem_counter.keys()):
            if old_sem in top_semantic_set:
                new_sem = old2new[old_sem]
                t = "kept"
            else:
                new_sem = other_id
                t = "other"

            f.write(f"{old_sem},{new_sem},{sem_counter[old_sem]},{t}\n")

    print("Saved:", output_file)
    print("Saved:", map_file)
    print("POIs kept in top classes:", kept_poi)
    print("POIs merged into Other:", other_poi)
    print("New semantic classes:", other_id + 1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkins", type=str, required=True)
    parser.add_argument("--semantic-file", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--map-output", type=str, default="semantic_merge_map.csv")
    parser.add_argument("--top-k", type=int, default=50)

    args = parser.parse_args()

    merge_semantic_topk(
        checkin_file=args.checkins,
        semantic_file=args.semantic_file,
        output_file=args.output,
        map_file=args.map_output,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()