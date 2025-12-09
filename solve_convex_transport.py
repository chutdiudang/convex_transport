# python solve_convex_transport.py --data_dir dataset_large --out solution.csv
# import argparse
# import os

# import numpy as np
# import pandas as pd
# import cvxpy as cp


# def solve_convex_transport(data_dir="dataset_small", output_file="solution.csv"):
#     """
#     Giải bài toán vận tải với chi phí lồi:
#         C_ij(x) = a_ij * x^2 + b_ij * x
#     trên mạng sparse (chỉ các tuyến có trong cost_params.csv).
#     """

#     # ---------------------------------------------------------
#     # 1. Đọc dữ liệu
#     # ---------------------------------------------------------
#     warehouses_path = os.path.join(data_dir, "warehouses.csv")
#     customers_path = os.path.join(data_dir, "customers.csv")
#     cost_params_path = os.path.join(data_dir, "cost_params.csv")

#     df_w = pd.read_csv(warehouses_path)
#     df_c = pd.read_csv(customers_path)
#     df_cost = pd.read_csv(cost_params_path)

#     # Số kho, số khách
#     m = df_w.shape[0]
#     n = df_c.shape[0]

#     # Map id kho/khách -> index (0-based) để tiện truy cập
#     # (phòng trường hợp id không trùng với 1..m)
#     warehouse_ids = df_w["id"].to_numpy()
#     customer_ids = df_c["id"].to_numpy()

#     id_to_w_idx = {wid: idx for idx, wid in enumerate(warehouse_ids)}
#     id_to_c_idx = {cid: idx for idx, cid in enumerate(customer_ids)}

#     # Cung & cầu
#     S = df_w["supply"].to_numpy()   # shape (m,)
#     D = df_c["demand"].to_numpy()   # shape (n,)

#     # Các cạnh (tuyến hợp lệ): mỗi dòng là i, j, a_ij, b_ij, distance
#     edges_i = df_cost["i"].to_numpy(dtype=int)      # id kho
#     edges_j = df_cost["j"].to_numpy(dtype=int)      # id khách
#     a_ij = df_cost["a_ij"].to_numpy()
#     b_ij = df_cost["b_ij"].to_numpy()

#     num_edges = df_cost.shape[0]

#     print(f"Số kho (m) = {m}, số khách (n) = {n}, số tuyến hợp lệ = {num_edges}")
#     print(f"Tổng cung   = {S.sum():.4f}")
#     print(f"Tổng cầu    = {D.sum():.4f}")

#     # ---------------------------------------------------------
#     # 2. Tạo biến quyết định cho từng tuyến
#     # ---------------------------------------------------------
#     # x_e >= 0 với e = 0..num_edges-1
#     x = cp.Variable(num_edges, nonneg=True)

#     # ---------------------------------------------------------
#     # 3. Xây dựng ràng buộc
#     # ---------------------------------------------------------
#     constraints = []

#     # 3.1 Ràng buộc cung: sum_{j} x_{ij} <= S_i
#     for wid in warehouse_ids:
#         w_idx = id_to_w_idx[wid]
#         # chọn các cạnh xuất phát từ kho wid
#         edge_indices = np.where(edges_i == wid)[0]
#         if edge_indices.size > 0:
#             constraints.append(cp.sum(x[edge_indices]) <= S[w_idx])
#         else:
#             # Nếu kho này không có tuyến nào, ràng buộc trivially: 0 <= S_i
#             constraints.append(0 <= S[w_idx])

#     # 3.2 Ràng buộc cầu: sum_{i} x_{ij} = D_j
#     for cid in customer_ids:
#         c_idx = id_to_c_idx[cid]
#         # chọn các cạnh đi vào khách cid
#         edge_indices = np.where(edges_j == cid)[0]
#         if edge_indices.size == 0:
#             raise ValueError(f"Customer {cid} không có tuyến nào kết nối! Dataset không hợp lệ.")
#         constraints.append(cp.sum(x[edge_indices]) == D[c_idx])

#     # ---------------------------------------------------------
#     # 4. Hàm mục tiêu (chi phí lồi)
#     #     Z = sum_e (a_e * x_e^2 + b_e * x_e)
#     # ---------------------------------------------------------
#     objective = cp.Minimize(cp.sum(a_ij * cp.square(x) + b_ij * x))

#     # ---------------------------------------------------------
#     # 5. Giải bài toán với CVXPY
#     # ---------------------------------------------------------
#     prob = cp.Problem(objective, constraints)

#     # Bạn có thể chọn solver khác nếu có (MOSEK, OSQP, ECOS,...)
#     print("Đang giải bài toán tối ưu lồi...")
#     #prob.solve(solver=cp.ECOS, verbose=False)
#     prob.solve(solver=cp.OSQP, verbose=False)
#     print(f"Trạng thái lời giải: {prob.status}")
#     if prob.status not in ["optimal", "optimal_inaccurate"]:
#         print("⚠ Không tìm được nghiệm tối ưu.")
#         return

#     print(f"Giá trị chi phí tối ưu Z* = {prob.value:.4f}")

#     x_opt = x.value  # numpy array, shape (num_edges,)

#     # ---------------------------------------------------------
#     # 6. Xuất kết quả ra CSV
#     # ---------------------------------------------------------
#     # Gộp vào dataframe: i, j, x_ij, a_ij, b_ij, distance
#     df_solution = df_cost.copy()
#     df_solution["x_ij"] = x_opt

#     out_path = os.path.join(data_dir, output_file)
#     df_solution.to_csv(out_path, index=False)

#     print(f"✔ Đã lưu nghiệm tối ưu vào: {out_path}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_dir", type=str, default="dataset_small",
#                         help="Thư mục chứa các file CSV dataset")
#     parser.add_argument("--out", type=str, default="solution.csv",
#                         help="Tên file output (trong thư mục data_dir)")

#     args = parser.parse_args()

#     solve_convex_transport(
#         data_dir=args.data_dir,
#         output_file=args.out
#     )
import argparse
import os

import numpy as np
import pandas as pd
import cvxpy as cp


def solve_convex_transport(data_dir="dataset_small", output_file="solution.csv"):
    """
    Giải bài toán vận tải chi phí lồi trên mạng sparse:
        C_ij(x) = a_ij * x^2 + b_ij * x
    Sạch cảnh báo và tối ưu cho CVXPY.
    """

    # ---------------------------------------------------------
    # 1. Đọc dữ liệu
    # ---------------------------------------------------------
    df_w = pd.read_csv(os.path.join(data_dir, "warehouses.csv"))
    df_c = pd.read_csv(os.path.join(data_dir, "customers.csv"))
    df_cost = pd.read_csv(os.path.join(data_dir, "cost_params.csv"))

    m = df_w.shape[0]
    n = df_c.shape[0]

    warehouse_ids = df_w["id"].to_numpy()
    customer_ids = df_c["id"].to_numpy()

    # Map id → index
    id_to_w_idx = {wid: idx for idx, wid in enumerate(warehouse_ids)}
    id_to_c_idx = {cid: idx for idx, cid in enumerate(customer_ids)}

    # Supply / Demand
    S = df_w["supply"].to_numpy()
    D = df_c["demand"].to_numpy()

    # Edges
    edges_i = df_cost["i"].to_numpy(dtype=int)
    edges_j = df_cost["j"].to_numpy(dtype=int)
    a_ij = df_cost["a_ij"].to_numpy()
    b_ij = df_cost["b_ij"].to_numpy()
    num_edges = len(a_ij)

    print(f"Số kho = {m}, số khách = {n}, số tuyến = {num_edges}")
    print(f"Tổng cung  = {S.sum():.4f}")
    print(f"Tổng cầu   = {D.sum():.4f}")

    # ---------------------------------------------------------
    # 2. Biến quyết định
    # ---------------------------------------------------------
    x = cp.Variable(num_edges, nonneg=True)

    # ---------------------------------------------------------
    # 3. Ràng buộc
    # ---------------------------------------------------------
    constraints = []

    # (1) Cung: sum_j x_ij <= S_i
    for wid in warehouse_ids:
        w_idx = id_to_w_idx[wid]
        idx_edges = np.where(edges_i == wid)[0]
        if len(idx_edges) > 0:
            constraints.append(cp.sum(x[idx_edges]) <= S[w_idx])
        else:
            constraints.append(0 <= S[w_idx])

    # (2) Cầu: sum_i x_ij = D_j
    for cid in customer_ids:
        c_idx = id_to_c_idx[cid]
        idx_edges = np.where(edges_j == cid)[0]
        if len(idx_edges) == 0:
            raise ValueError(f"Customer {cid} không có tuyến kết nối!")
        constraints.append(cp.sum(x[idx_edges]) == D[c_idx])

    # ---------------------------------------------------------
    # 4. Hàm mục tiêu (sạch cảnh báo)
    # ---------------------------------------------------------
    # Thay vì a * x * x → dùng cp.multiply()
    quad_term = cp.multiply(a_ij, cp.square(x))
    lin_term  = cp.multiply(b_ij, x)

    objective = cp.Minimize(cp.sum(quad_term + lin_term))

    # ---------------------------------------------------------
    # 5. Giải bài toán
    # ---------------------------------------------------------
    prob = cp.Problem(objective, constraints)

    print("\nĐang giải bài toán tối ưu lồi (OSQP)...")
    prob.solve(solver=cp.OSQP, verbose=False)
    #prob.solve(solver=cp.ECOS, verbose=False)
    print(f"Trạng thái lời giải: {prob.status}")
    print(f"Chi phí tối ưu Z* = {prob.value:.4f}")

    x_opt = x.value

    # ---------------------------------------------------------
    # 6. Xuất kết quả
    # ---------------------------------------------------------
    df_out = df_cost.copy()
    df_out["x_ij"] = x_opt

    save_path = os.path.join(data_dir, output_file)
    df_out.to_csv(save_path, index=False)

    print(f"✔ Đã lưu nghiệm tối ưu vào: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset_small")
    parser.add_argument("--out", type=str, default="solution.csv")
    args = parser.parse_args()

    solve_convex_transport(args.data_dir, args.out)
