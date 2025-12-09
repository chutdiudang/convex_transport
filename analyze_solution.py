#python analyze_solution.py --data_dir dataset_large --solution solution.csv

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(data_dir, solution_file):
    df_w = pd.read_csv(os.path.join(data_dir, "warehouses.csv"))
    df_c = pd.read_csv(os.path.join(data_dir, "customers.csv"))
    df_sol = pd.read_csv(os.path.join(data_dir, solution_file))

    return df_w, df_c, df_sol


# ============================================================================
# 1. HEATMAP phân bố hàng (m x n)
# ============================================================================
def plot_heatmap(df_w, df_c, df_sol, title="Heatmap phân bố hàng"):
    m = df_w.shape[0]
    n = df_c.shape[0]

    heat = np.zeros((m, n))

    for _, row in df_sol.iterrows():
        i = int(row["i"]) - 1
        j = int(row["j"]) - 1
        heat[i][j] = row["x_ij"]

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        heat,
        cmap="Blues",
        cbar=True,
        linewidths=0.2,
        linecolor="gray"
    )
    plt.title(title)
    plt.xlabel("Customers")
    plt.ylabel("Warehouses")
    plt.tight_layout()
    plt.show()


# ============================================================================
# 2. HIỂN THỊ CÁC TUYẾN ĐƯỢC SỬ DỤNG (flow > 0)
#    Vẽ kho – khách hàng trên mặt phẳng, và nối các tuyến được sử dụng.
# ============================================================================
def plot_used_routes(df_w, df_c, df_sol, threshold=1e-6):
    plt.figure(figsize=(12, 10))

    # Vẽ vị trí khách và kho
    plt.scatter(df_c["x"], df_c["y"], c="red", label="Customers", s=50)
    plt.scatter(df_w["x"], df_w["y"], c="blue", label="Warehouses", s=80)

    # Annotate id
    for _, r in df_w.iterrows():
        plt.text(r["x"] + 0.5, r["y"] + 0.5, f"W{int(r['id'])}", color="blue")

    for _, r in df_c.iterrows():
        plt.text(r["x"] + 0.5, r["y"] + 0.5, f"C{int(r['id'])}", color="red")

    # Vẽ các tuyến được dùng
    for _, row in df_sol.iterrows():
        if row["x_ij"] > threshold:
            wi = df_w.iloc[int(row["i"]) - 1]
            cj = df_c.iloc[int(row["j"]) - 1]

            # độ đậm tuyến tỉ lệ với lượng flow
            lw = max(0.5, row["x_ij"] / df_sol["x_ij"].max() * 3)

            plt.plot(
                [wi["x"], cj["x"]],
                [wi["y"], cj["y"]],
                color="gray",
                linewidth=lw,
                alpha=0.7
            )

    plt.title("Các tuyến vận tải được sử dụng (flow > 0)")
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================================
# 3. TÍNH TẢI CỦA TỪNG KHO
#    load_i = tổng lượng xuất từ kho i
# ============================================================================
def compute_warehouse_load(df_w, df_sol):
    m = df_w.shape[0]
    load = np.zeros(m)

    for _, row in df_sol.iterrows():
        i = int(row["i"]) - 1
        load[i] += row["x_ij"]

    df_load = df_w.copy()
    df_load["load"] = load
    df_load["utilization"] = load / df_w["supply"]  # % tải

    print("\n===== TẢI KHO =====")
    print(df_load[["id", "supply", "load", "utilization"]])

    # Vẽ biểu đồ
    plt.figure(figsize=(12, 5))
    sns.barplot(x=df_load["id"], y=df_load["load"], color="blue")
    plt.title("Tải của từng kho")
    plt.xlabel("Warehouse ID")
    plt.ylabel("Lượng hàng xuất")
    plt.show()

    return df_load


# ============================================================================
# 4. TỈ LỆ KHÁCH NHẬN TỪ BAO NHIÊU KHO
# ============================================================================
def compute_customer_sources(df_c, df_sol, threshold=1e-6):
    n = df_c.shape[0]
    source_count = np.zeros(n, dtype=int)

    for _, row in df_sol.iterrows():
        if row["x_ij"] > threshold:
            j = int(row["j"]) - 1
            source_count[j] += 1

    df_src = df_c.copy()
    df_src["num_sources"] = source_count

    print("\n===== SỐ KHO CUNG CẤP CHO MỖI KHÁCH =====")
    print(df_src[["id", "num_sources"]])

    # Biểu đồ phân phối
    plt.figure(figsize=(10, 5))
    sns.countplot(x=df_src["num_sources"])
    plt.title("Phân phối: Khách nhận hàng từ bao nhiêu kho")
    plt.xlabel("Số kho phục vụ")
    plt.ylabel("Số khách")
    plt.show()

    return df_src


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset_small")
    parser.add_argument("--solution", type=str, default="solution.csv")
    args = parser.parse_args()

    df_w, df_c, df_sol = load_data(args.data_dir, args.solution)

    # 1. Heatmap dòng vận tải
    plot_heatmap(df_w, df_c, df_sol)

    # 2. Vẽ các tuyến được sử dụng
    plot_used_routes(df_w, df_c, df_sol)

    # 3. Tính tải kho
    compute_warehouse_load(df_w, df_sol)

    # 4. Tính số nguồn cung cấp mỗi khách
    compute_customer_sources(df_c, df_sol)
