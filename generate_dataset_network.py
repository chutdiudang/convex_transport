import numpy as np
import pandas as pd
import argparse
import os

# ======================================================================
# Generate Convex Transportation Dataset with Distance-Based Network
# ======================================================================

def generate_dataset(size="small", R=30, seed=42, output_dir="dataset"):
    """
    Generate dataset for Convex Transportation Problem
    with sparse network: warehouse connects to customer
    only if distance <= R.
    """

    np.random.seed(seed)

    # ---------------------------------------------------------
    # 1. Dataset sizes
    # ---------------------------------------------------------
    if size == "small":
        m = 5      # warehouses
        n = 10     # customers
    elif size == "medium":
        m = 20
        n = 40
    elif size == "large":
        m = 60
        n = 150
    else:
        raise ValueError("size must be small, medium or large")

    print(f"\n=== Generating dataset ({size}) with R = {R} ===\n")

    # Create folder
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 2. Generate coordinates
    # ---------------------------------------------------------
    # Place warehouses and customers in 2D grid 0–100
    W_xy = np.random.uniform(0, 100, size=(m, 2))
    C_xy = np.random.uniform(0, 100, size=(n, 2))

    # ---------------------------------------------------------
    # 3. Generate demands (customer)
    # ---------------------------------------------------------
    D = np.random.uniform(10, 50, size=n)
    total_demand = D.sum()

    # ---------------------------------------------------------
    # 4. Generate supplies (warehouse)
    # ---------------------------------------------------------
    mean_supply = total_demand / m
    S = np.random.uniform(0.8 * mean_supply, 1.2 * mean_supply, size=m)

    # Normalize supply so total = total_demand
    S = S * (total_demand / S.sum())

    # ---------------------------------------------------------
    # 5. Compute distances
    # ---------------------------------------------------------
    dist = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            dist[i, j] = np.linalg.norm(W_xy[i] - C_xy[j])

    # ---------------------------------------------------------
    # 6. Build network connections based on R
    # ---------------------------------------------------------
    connections = []
    for i in range(m):
        for j in range(n):
            if dist[i, j] <= R:
                connections.append([i + 1, j + 1])

    # Ensure every customer has at least 1 connection
    customers_with_conn = {j for (_, j) in connections}

    for j in range(1, n + 1):
        if j not in customers_with_conn:
            # assign the nearest warehouse
            nearest_i = np.argmin(dist[:, j - 1]) + 1
            connections.append([nearest_i, j])

    print(f"Total valid connections: {len(connections)}")

    # ---------------------------------------------------------
    # 7. Generate convex cost parameters a_ij and b_ij
    # Only for valid edges (i,j)
    # ---------------------------------------------------------
    rows = []

    c1 = 1.0     # cost per km per ton
    c2 = 0.05    # convexity factor
    Q  = 50      # scaling

    for (i, j) in connections:
        d_ij = dist[i - 1, j - 1]
        b_ij = c1 * d_ij
        a_ij = c2 * (d_ij / (Q ** 2))
        rows.append([i, j, a_ij, b_ij, d_ij])

    # ---------------------------------------------------------
    # 8. SAVE CSV FILES
    # ---------------------------------------------------------

    # warehouses.csv
    df_w = pd.DataFrame({
        "id": np.arange(1, m + 1),
        "x": W_xy[:, 0],
        "y": W_xy[:, 1],
        "supply": S
    })
    df_w.to_csv(f"{output_dir}/warehouses.csv", index=False)

    # customers.csv
    df_c = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "x": C_xy[:, 0],
        "y": C_xy[:, 1],
        "demand": D
    })
    df_c.to_csv(f"{output_dir}/customers.csv", index=False)

    # network_connections.csv
    df_net = pd.DataFrame(connections, columns=["warehouse", "customer"])
    df_net.to_csv(f"{output_dir}/network_connections.csv", index=False)

    # cost_params.csv
    df_cost = pd.DataFrame(rows,
        columns=["i", "j", "a_ij", "b_ij", "distance"])
    df_cost.to_csv(f"{output_dir}/cost_params.csv", index=False)

    # metadata.csv
    df_meta = pd.DataFrame({
        "dataset_size": [size],
        "m_warehouses": [m],
        "n_customers": [n],
        "R_distance_limit": [R],
        "total_supply": [S.sum()],
        "total_demand": [D.sum()],
        "seed": [seed]
    })
    df_meta.to_csv(f"{output_dir}/metadata.csv", index=False)

    print(f"✔ Dataset generated successfully in folder: {output_dir}")
    print("Files:")
    print(" - warehouses.csv")
    print(" - customers.csv")
    print(" - network_connections.csv")
    print(" - cost_params.csv")
    print(" - metadata.csv")


# ======================================================================
# Main program
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="small",
                        choices=["small", "medium", "large"])
    parser.add_argument("--R", type=float, default=30,
                        help="connection radius (distance limit)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="dataset")

    args = parser.parse_args()

    generate_dataset(
        size=args.size,
        R=args.R,
        seed=args.seed,
        output_dir=args.out
    )
#python generate_dataset_network.py --size small --R 30 --out dataset_small
#python generate_dataset_network.py --size medium --R 40 --out dataset_medium
#python generate_dataset_network.py --size large --R 50 --out dataset_large
#python generate_dataset_network.py --size small --R 30 --seed 123
