"""
Shopee Vietnamese Review - Text Mining + Association Rule Mining
----------------------------------------------------------------
Script dùng cho báo cáo Chủ đề 4:
- Gộp các file data1.csv, data2.csv, data3.csv.
- Làm sạch dữ liệu, bỏ trùng bình luận.
- Chuyển mỗi review thành một transaction gồm các item/aspect và item nhãn cảm xúc.
- Chạy Apriori tự cài đặt để tìm frequent itemsets và association rules.
- Xuất các bảng CSV và hình minh họa trong thư mục results/.

Chạy nhanh:
python code/shopee_arm_text_mining.py --data-dir data --output-dir results
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Set

import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# 1. Cấu hình xử lý văn bản
# -------------------------
STOPWORDS = set(
    "và là thì mà của cho với ở trên dưới này kia đó đây mình tôi em anh chị bạn "
    "rất hơi quá cũng đã đang sẽ vẫn nên nữa ạ nhé nha luôn lắm thôi về từ ra vào "
    "cái con hàng sản phẩm shopee được kh không ko k kg vs đ â y ra sao gì có còn "
    "nếu hoặc để trong đó nó một các b"
    .split()
)

NORMALIZE_MAP = {
    "ko": "không",
    "k": "không",
    "kh": "không",
    "kg": "không",
    "hok": "không",
    "dc": "được",
    "đc": "được",
    "sp": "sản_phẩm",
    "san_pham": "sản_phẩm",
    "ship": "giao",
    "shipper": "người_giao_hàng",
    "ok": "tốt",
    "oke": "tốt",
    "xinh": "đẹp",
    "xịn": "tốt",
    "mn": "mọi_người",
    "mik": "mình",
}

# Các khía cạnh/aspect thường gặp trong đánh giá thương mại điện tử.
ASPECTS = [
    ("đúng_mô_tả", ["đúng mô_tả", "đúng_mô_tả", "mô_tả đúng", "y hình", "y_hình", "như hình", "giống hình", "giống ảnh", "đúng mẫu", "đúng màu"]),
    ("không_giống_mô_tả", ["không đúng mô_tả", "sai mô_tả", "khác hình", "khác ảnh", "sai màu", "sai mẫu", "không giống", "chưa đúng"]),
    ("giao_hàng_nhanh", ["giao nhanh", "giao_hàng nhanh", "giao hàng nhanh", "ship nhanh", "nhận hàng nhanh", "giao đúng"]),
    ("giao_hàng_chậm", ["giao lâu", "giao chậm", "ship chậm", "chờ lâu", "lâu giao", "giao_hàng chậm"]),
    ("đóng_gói_tốt", ["đóng_gói cẩn_thận", "đóng gói cẩn thận", "đóng_gói tốt", "gói kỹ", "bọc kỹ", "đóng_gói kỹ"]),
    ("đóng_gói_kém", ["đóng_gói sơ_sài", "đóng gói sơ sài", "móp hộp", "vỡ hộp", "hộp móp", "gói kém"]),
    ("chất_lượng_tốt", ["chất_lượng tốt", "chất lượng tốt", "sản_phẩm tốt", "dùng tốt", "hàng tốt", "nghe tốt", "khá ổn", "tạm ổn", "ổn áp"]),
    ("chất_lượng_kém", ["chất_lượng kém", "hàng kém", "chất_lượng không tốt", "không tốt", "dùng chán", "tệ", "chán", "kém"]),
    ("sản_phẩm_đẹp", ["sản_phẩm đẹp", "hàng đẹp", "dép đẹp", "áo đẹp", "túi đẹp", "mẫu đẹp", "màu đẹp", "đẹp lắm", "đẹp"]),
    ("giá_rẻ_hợp_lý", ["giá rẻ", "giá tốt", "đáng tiền", "đáng mua", "phù_hợp giá", "hợp giá", "với giá", "giá tiền", "săn sale", "sa le"]),
    ("giá_cao", ["giá cao", "hơi đắt", "đắt", "không đáng tiền"]),
    ("shop_tốt", ["shop nhiệt_tình", "tư_vấn nhiệt_tình", "shop thân_thiện", "shop dễ_thương", "cảm_ơn shop", "ủng_hộ shop", "shop tốt"]),
    ("shop_kém", ["shop không phản_hồi", "shop rep chậm", "không trả_lời", "tư_vấn kém", "shop kém"]),
    ("màu_sắc_đẹp", ["màu_sắc", "màu đẹp", "màu ok", "màu xinh", "màu chuẩn", "đúng màu"]),
    ("chất_liệu_tốt", ["chất_liệu tốt", "chất liệu tốt", "vải đẹp", "chất vải", "vải ok", "vải mềm"]),
    ("chất_liệu_mỏng", ["vải mỏng", "áo mỏng", "mỏng", "chất mỏng", "vải hơi mỏng"]),
    ("kích_thước_vừa", ["vừa chân", "vừa_vặn", "mang vừa", "size vừa", "đúng size", "fit"]),
    ("sản_phẩm_lỗi_hỏng", ["bị lỗi", "hàng lỗi", "lỗi", "hỏng", "vỡ", "rách", "móp", "bể"]),
    ("âm_thanh_tốt", ["nghe êm", "âm_thanh", "êm tai", "nghe tốt", "bass", "tai nghe"]),
    ("mùi_hương_tốt", ["mùi hương", "thơm", "lưu hương", "mùi thơm"]),
    ("review_nhận_xu", ["nhận xu", "lấy xu", "đánh_giá lấy xu", "comment nhận xu", "mang tính_chất"]),
    ("sẽ_mua_lại", ["lần sau", "mua lại", "ủng_hộ tiếp", "sẽ ủng_hộ", "quay lại"]),
]

GENERIC_KEEP = {
    "đẹp", "tốt", "nhanh", "rẻ", "ổn", "bền", "êm", "mềm", "thơm",
    "mỏng", "lỗi", "chậm", "kém", "đắt", "hỏng", "rách", "móp", "vừa", "chuẩn", "xấu"
}

# -------------------------
# 2. Đọc và chuẩn hóa dữ liệu
# -------------------------
def read_dataset(data_dir: str) -> pd.DataFrame:
    data_path = Path(data_dir)
    frames = []

    data1_path = data_path / "data1.csv"
    data2_path = data_path / "data2.csv"
    data3_path = data_path / "data3.csv"

    if data1_path.exists():
        df = pd.read_csv(data1_path)
        frames.append(df[["text", "label"]].assign(source="data1"))

    if data2_path.exists():
        df = pd.read_csv(data2_path)
        frames.append(df[["text", "label"]].assign(source="data2"))

    if data3_path.exists():
        df = pd.read_csv(data3_path).rename(columns={"txt": "text", "lbl": "label", "prob_positve": "prob_positive"})
        keep_cols = ["text", "label", "source"]
        df = df.assign(source="data3")
        if "prob_positive" in df.columns:
            keep_cols.append("prob_positive")
        frames.append(df[keep_cols])

    if not frames:
        raise FileNotFoundError("Không tìm thấy data1.csv, data2.csv hoặc data3.csv trong data/.")

    df_all = pd.concat(frames, ignore_index=True, sort=False)
    df_all = df_all.dropna(subset=["text"]).copy()
    df_all["text"] = df_all["text"].astype(str).str.strip().str.lower()
    df_all = df_all[df_all["text"] != ""].copy()
    return df_all


def clean_tokens(text: str) -> List[str]:
    text = unicodedata.normalize("NFC", str(text).lower())
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹ_\s]", " ", text)
    text = re.sub(r"\d+", " ", text)

    tokens = []
    for token in re.findall(r"[a-zA-ZÀ-ỹ_]+", text):
        token = NORMALIZE_MAP.get(token, token)
        if token in STOPWORDS or len(token) <= 1:
            continue
        # Bỏ các chuỗi nhiễu như 'aa', 'dd', 'zz'.
        if re.fullmatch(r"[a-z]{1,2}", token):
            continue
        tokens.append(token)
    return tokens


def build_transaction(text: str, label: int) -> List[str]:
    raw = str(text).lower()
    raw_clean = re.sub(r"[^0-9a-zA-ZÀ-ỹ_\s]", " ", raw)
    raw_space = raw_clean.replace("_", " ")
    searchable = raw_clean + " " + raw_space

    items: Set[str] = set()

    for aspect_name, patterns in ASPECTS:
        if any(pattern in searchable for pattern in patterns):
            items.add(aspect_name)

    for token in clean_tokens(text):
        if token in GENERIC_KEEP:
            items.add("kw_" + token)

    # Trong dataset này, label=1 tương ứng với tích cực, label=0 tương ứng với nhóm không tích cực/trung tính.
    items.add("sentiment_positive" if int(label) == 1 else "sentiment_nonpositive")
    return sorted(items)

# -------------------------
# 3. Apriori tự cài đặt
# -------------------------
def count_itemsets(transactions: List[List[str]], max_len: int) -> Counter:
    counter: Counter = Counter()
    for trans in transactions:
        trans = sorted(set(trans))
        for k in range(1, min(max_len, len(trans)) + 1):
            for comb in itertools.combinations(trans, k):
                counter[comb] += 1
    return counter


def generate_rules(counts: Counter, n_transactions: int, min_support: float, min_confidence: float, min_lift: float) -> pd.DataFrame:
    frequent = {items: cnt for items, cnt in counts.items() if cnt / n_transactions >= min_support}
    rules = []
    for itemset, itemset_count in frequent.items():
        if len(itemset) < 2:
            continue
        itemset_set = set(itemset)
        for r in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, r):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(itemset_set - set(antecedent)))
                antecedent_count = counts[antecedent]
                consequent_count = counts[consequent]
                support = itemset_count / n_transactions
                confidence = itemset_count / antecedent_count
                lift = confidence / (consequent_count / n_transactions)
                if confidence >= min_confidence and lift >= min_lift:
                    rules.append({
                        "antecedents": ", ".join(antecedent),
                        "consequents": ", ".join(consequent),
                        "support": support,
                        "confidence": confidence,
                        "lift": lift,
                        "count": itemset_count,
                        "antecedent_count": antecedent_count,
                        "consequent_count": consequent_count,
                    })
    if not rules:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift", "count"])
    return pd.DataFrame(rules).sort_values(["lift", "confidence", "support"], ascending=False)

# -------------------------
# 4. Trực quan hóa
# -------------------------
def save_barh(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, out_path: Path, top_n: int = 15):
    plot_df = df.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df[y_col], plot_df[x_col])
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_all = read_dataset(args.data_dir)
    df_all.to_csv(Path(args.data_dir) / "full_shopee_reviews_raw.csv", index=False, encoding="utf-8-sig")

    # Bỏ trùng theo text để giảm nhiễu từ các câu lặp lại nguyên văn.
    df_unique = df_all.drop_duplicates(subset=["text"], keep="first").copy()
    df_unique.to_csv(Path(args.data_dir) / "full_shopee_reviews_unique.csv", index=False, encoding="utf-8-sig")

    transactions = []
    valid_rows = []
    for idx, row in df_unique.iterrows():
        trans = build_transaction(row["text"], row["label"])
        # Cần có ít nhất một item nội dung ngoài item sentiment.
        content_items = [item for item in trans if not item.startswith("sentiment_")]
        if len(content_items) >= args.min_content_items:
            transactions.append(trans)
            valid_rows.append(idx)

    df_valid = df_unique.loc[valid_rows].copy()
    df_valid.to_csv(Path(args.data_dir) / "full_shopee_reviews_valid_transactions.csv", index=False, encoding="utf-8-sig")

    n = len(transactions)
    counts = count_itemsets(transactions, args.max_len)
    item_counter = Counter(item for trans in transactions for item in trans)

    # Top item/keyword
    top_items = pd.DataFrame(item_counter.most_common(), columns=["item", "frequency"])
    top_items["support"] = top_items["frequency"] / n
    top_items.to_csv(output_dir / "top_keywords.csv", index=False, encoding="utf-8-sig")

    # Frequent itemsets
    frequent_rows = []
    for itemset, cnt in counts.items():
        support = cnt / n
        if support >= args.min_support:
            frequent_rows.append({"itemsets": ", ".join(itemset), "support": support, "count": cnt, "length": len(itemset)})
    frequent_df = pd.DataFrame(frequent_rows).sort_values(["support", "count"], ascending=False)
    frequent_df.to_csv(output_dir / "frequent_itemsets.csv", index=False, encoding="utf-8-sig")

    # Association rules
    rules_df = generate_rules(counts, n, args.min_support, args.min_confidence, args.min_lift)
    rules_df.to_csv(output_dir / "association_rules.csv", index=False, encoding="utf-8-sig")

    # Rules có consequent là sentiment để phân tích trong báo cáo.
    sentiment_rules = rules_df[
        rules_df["consequents"].isin(["sentiment_positive", "sentiment_nonpositive"])
        & ~rules_df["antecedents"].str.startswith("sentiment_")
    ].copy()
    sentiment_rules.to_csv(output_dir / "sentiment_rules.csv", index=False, encoding="utf-8-sig")

    # Summary
    summary = {
        "raw_rows": len(df_all),
        "unique_reviews": len(df_unique),
        "valid_transactions": n,
        "num_unique_items": len(item_counter),
        "num_frequent_itemsets": len(frequent_df),
        "num_association_rules": len(rules_df),
        "num_sentiment_rules": len(sentiment_rules),
        "min_support": args.min_support,
        "min_confidence": args.min_confidence,
        "min_lift": args.min_lift,
        "max_len": args.max_len,
        "label_0_raw": int((df_all["label"] == 0).sum()),
        "label_1_raw": int((df_all["label"] == 1).sum()),
        "label_0_unique": int((df_unique["label"] == 0).sum()),
        "label_1_unique": int((df_unique["label"] == 1).sum()),
        "label_0_valid": int((df_valid["label"] == 0).sum()),
        "label_1_valid": int((df_valid["label"] == 1).sum()),
    }
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")

    # Label/source statistics
    pd.crosstab(df_all["source"], df_all["label"]).to_csv(output_dir / "source_label_distribution_raw.csv", encoding="utf-8-sig")
    pd.crosstab(df_unique["source"], df_unique["label"]).to_csv(output_dir / "source_label_distribution_unique.csv", encoding="utf-8-sig")

    # Figures
    fig_dir = Path("figures")
    # Nếu chạy từ thư mục project, lưu hình vào figures/; nếu không có thì lưu trong output.
    if not fig_dir.exists():
        fig_dir = output_dir
    save_barh(top_items[~top_items["item"].str.startswith("sentiment_")], "frequency", "item", "Top item/khía cạnh xuất hiện nhiều nhất", "Tần suất", fig_dir / "top_items.png", 15)

    freq_len2 = frequent_df[frequent_df["length"] >= 2].head(15).copy()
    if len(freq_len2) > 0:
        save_barh(freq_len2, "support", "itemsets", "Top frequent itemsets", "Support", fig_dir / "top_frequent_itemsets.png", 12)

    if len(sentiment_rules) > 0:
        plt.figure(figsize=(8, 5.5))
        plt.scatter(sentiment_rules["confidence"], sentiment_rules["lift"], s=sentiment_rules["support"] * 3000)
        plt.xlabel("Confidence")
        plt.ylabel("Lift")
        plt.title("Phân bố luật kết hợp theo confidence và lift")
        plt.tight_layout()
        plt.savefig(fig_dir / "rules_confidence_lift.png", dpi=220)
        plt.close()

    label_df = pd.DataFrame({
        "label": ["0 - nonpositive", "1 - positive"],
        "count": [summary["label_0_unique"], summary["label_1_unique"]],
    })
    plt.figure(figsize=(6.5, 4.5))
    plt.bar(label_df["label"], label_df["count"])
    plt.ylabel("Số review sau bỏ trùng")
    plt.title("Phân bố nhãn sentiment trong dữ liệu")
    plt.tight_layout()
    plt.savefig(fig_dir / "sentiment_distribution.png", dpi=220)
    plt.close()

    print("========== HOÀN THÀNH THỰC NGHIỆM ==========")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"Kết quả lưu tại: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--min-support", type=float, default=0.025)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--min-lift", type=float, default=1.0)
    parser.add_argument("--max-len", type=int, default=3)
    parser.add_argument("--min-content-items", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
