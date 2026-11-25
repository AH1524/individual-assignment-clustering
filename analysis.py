"""
Gym member k-means segments and $10 extras-credit targeting.

Reproduces every number used in the assignment slides.
Run:  .venv/bin/python analysis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.30
CREDIT_COST = 10.0
SPEND_CUTOFF = 20.0
LIFT_MARGIN = 15.0  # $25 extra spend * 60% margin
NET_IF_LOW_SPEND = LIFT_MARGIN - CREDIT_COST  # +5
NET_IF_HIGH_SPEND = -CREDIT_COST  # -10

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "outputs"

FEATURE_COLS = [
    "age",
    "visits_per_month",
    "avg_visit_minutes",
    "classes_per_month",
    "months_member",
    "monthly_extras_spend",
]
LEFT_OUT = ["member_id"]

# Filled after inspecting centroids (cluster id -> name / credit).
# Names are set from the fitted centers in original units.
SEGMENT_META = {}


def inertia_from_centers(X: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
    """Sum of squared distances to assigned centers (no refit)."""
    return float(((X - centers[labels]) ** 2).sum())


def member_credit_net(spend: float, gets_credit: bool) -> float:
    """Finance assumptions applied to one member, not a segment average."""
    if not gets_credit:
        return 0.0
    if spend < SPEND_CUTOFF:
        return NET_IF_LOW_SPEND
    return NET_IF_HIGH_SPEND


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)


def segment_action(
    age: float,
    visits_per_month: float,
    avg_visit_minutes: float,
    classes_per_month: float,
    months_member: float,
    monthly_extras_spend: float,
    scaler: StandardScaler,
    model: KMeans,
    segment_meta: dict,
) -> dict:
    """Assign one member to a fitted segment and the credit decision for that segment."""
    row = pd.DataFrame(
        [
            {
                "age": age,
                "visits_per_month": visits_per_month,
                "avg_visit_minutes": avg_visit_minutes,
                "classes_per_month": classes_per_month,
                "months_member": months_member,
                "monthly_extras_spend": monthly_extras_spend,
            }
        ],
        columns=FEATURE_COLS,
    )
    scaled = scaler.transform(row)
    cluster = int(model.predict(scaled)[0])
    distances = np.sqrt(((scaled - model.cluster_centers_) ** 2).sum(axis=1))
    meta = segment_meta[cluster]
    return {
        "cluster": cluster,
        "segment": meta["name"],
        "gets_credit": meta["gets_credit"],
        "distances": distances,
    }


def name_from_centroid(row: pd.Series) -> tuple[str, str]:
    """Names come from the centroid in original units, not the cluster id."""
    spend = row["monthly_extras_spend"]
    visits = row["visits_per_month"]
    classes = row["classes_per_month"]
    minutes = row["avg_visit_minutes"]
    tenure = row["months_member"]
    age = row["age"]

    if spend >= 35 and classes >= 5:
        return "Class spenders", "Many classes, high extras, mid-length visits"
    if visits >= 12 and minutes >= 55 and classes < 3:
        return "Solo grinders", "Many long visits, almost no classes"
    if visits <= 5 and spend < 20 and tenure <= 20:
        return "Light newcomers", "Few visits, short tenure, low extras"
    if age >= 50 and tenure >= 40:
        return "Veteran regulars", "Older, long membership, moderate visits"
    return f"Segment", "See centroid row"


def credit_from_train(train: pd.DataFrame) -> dict[int, bool]:
    """
    A segment gets the credit if, on training members only, giving everyone
    in that segment the credit is net-positive under finance's rules.
    """
    out = {}
    for cid, part in train.groupby("cluster"):
        net = sum(member_credit_net(s, True) for s in part["monthly_extras_spend"])
        out[int(cid)] = net > 0
        print(f"  train cluster {cid}: n={len(part)}  net if credited=${net:.0f}")
    return out


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(ROOT / "gym_members.csv")
    print("=== Data ===")
    print(f"rows, cols: {df.shape}")
    print(f"columns: {list(df.columns)}")
    print(f"mean monthly extras spend: {df['monthly_extras_spend'].mean():.2f}")
    print(f"left out of the fit: {LEFT_OUT}")
    print(f"features: {FEATURE_COLS}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print(df[FEATURE_COLS].describe())

    train, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print("\n=== Split ===")
    print(f"train: {len(train)} ({len(train) / len(df):.0%})")
    print(f"test:  {len(test)} ({len(test) / len(df):.0%})")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLS])
    X_test = scaler.transform(test[FEATURE_COLS])

    ks = list(range(1, 9))
    train_inertias = []
    test_inertias = []
    models = {}

    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        km.fit(X_train)
        models[k] = km
        train_inertias.append(float(km.inertia_))
        test_labels = km.predict(X_test)
        test_inertias.append(
            inertia_from_centers(X_test, km.cluster_centers_, test_labels)
        )
        print(
            f"k={k}  train inertia={train_inertias[-1]:.1f}  "
            f"test inertia={test_inertias[-1]:.1f}"
        )

    drops_train = np.diff(train_inertias)
    drops_test = np.diff(test_inertias)
    print("\nTrain inertia drops (k to k+1):", np.round(drops_train, 1))
    print("Test inertia drops (k to k+1): ", np.round(drops_test, 1))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(ks, train_inertias, marker="o", color="#1f4e79", label="Training members")
    ax.plot(ks, test_inertias, marker="s", color="#c45911", label="Test members")
    ax.set_xlabel("k (number of groups)")
    ax.set_ylabel("Within-group spread (sum of squared distances)")
    ax.set_xticks(ks)
    style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "elbow.png", dpi=200)
    plt.close(fig)

    # Training drops stay large through 3→4 (−276) then flatten (4→5 is −26).
    # The test curve has the same bend. Ops asked for 3–5 groups.
    chosen_k = 4
    print(f"\nChosen k from the elbow: {chosen_k}")

    model = models[chosen_k]
    centers_orig = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=FEATURE_COLS,
    )
    centers_orig.index.name = "cluster"
    print("\n=== Cluster centers (original units) ===")
    print(centers_orig.round(1))

    train = train.copy()
    test = test.copy()
    df_all = df.copy()
    train["cluster"] = model.predict(X_train)
    test["cluster"] = model.predict(X_test)
    df_all["cluster"] = model.predict(scaler.transform(df_all[FEATURE_COLS]))

    counts = df_all["cluster"].value_counts().sort_index()
    print("\nMember counts (all 330, assigned with fitted model):")
    print(counts.to_string())

    print("\nShare with extras spend < $20 by cluster:")
    for cid in sorted(df_all["cluster"].unique()):
        part = df_all[df_all["cluster"] == cid]
        share = (part["monthly_extras_spend"] < SPEND_CUTOFF).mean()
        print(
            f"  cluster {cid}: n={len(part)}  "
            f"mean extras=${part['monthly_extras_spend'].mean():.1f}  "
            f"share < $20={share:.1%}"
        )

    print("\nTrain-set net from crediting every member in a cluster:")
    credit_flags = credit_from_train(train)

    global SEGMENT_META
    SEGMENT_META = {}
    for cid, row in centers_orig.iterrows():
        name, traits = name_from_centroid(row)
        SEGMENT_META[int(cid)] = {
            "name": name,
            "traits": traits,
            "gets_credit": credit_flags[int(cid)],
        }

    print("\n=== Segment names and credit ===")
    for cid, meta in SEGMENT_META.items():
        print(
            f"  {cid}: {meta['name']} | {meta['traits']} | "
            f"credit={meta['gets_credit']}"
        )

    k1_test = test_inertias[0]
    vk_test = test_inertias[chosen_k - 1]
    print("\n=== Test spread: baseline vs v1 ===")
    print(f"test spread k=1 (baseline): {k1_test:.1f}")
    print(f"test spread k={chosen_k} (v1):     {vk_test:.1f}")
    print(f"reduction vs baseline: {(k1_test - vk_test) / k1_test:.1%}")

    new_member = dict(
        age=30,
        visits_per_month=4,
        avg_visit_minutes=42,
        classes_per_month=1,
        months_member=5,
        monthly_extras_spend=10,
    )
    decision = segment_action(
        **new_member, scaler=scaler, model=model, segment_meta=SEGMENT_META
    )
    print("\n=== New member ===")
    print(new_member)
    print(f"distances to centers: {np.round(decision['distances'], 2)}")
    print(
        f"segment={decision['segment']} (cluster {decision['cluster']})  "
        f"gets_credit={decision['gets_credit']}"
    )

    df_all["gets_credit_v1"] = df_all["cluster"].map(
        lambda c: SEGMENT_META[int(c)]["gets_credit"]
    )
    df_all["net_baseline"] = [
        member_credit_net(s, True) for s in df_all["monthly_extras_spend"]
    ]
    df_all["net_v1"] = [
        member_credit_net(s, g)
        for s, g in zip(df_all["monthly_extras_spend"], df_all["gets_credit_v1"])
    ]

    baseline_net = float(df_all["net_baseline"].sum())
    v1_net = float(df_all["net_v1"].sum())
    n_credit_v1 = int(df_all["gets_credit_v1"].sum())

    print("\n=== Monthly net from the $10 credit ===")
    print(f"baseline (credit to all {len(df_all)}): ${baseline_net:.0f}")
    print(f"v1 (credit to {n_credit_v1} members in chosen segments): ${v1_net:.0f}")
    print(f"v1 minus baseline: ${v1_net - baseline_net:.0f}")

    summary_rows = []
    for cid in sorted(SEGMENT_META):
        part = df_all[df_all["cluster"] == cid]
        summary_rows.append(
            {
                "cluster": cid,
                "segment": SEGMENT_META[cid]["name"],
                "n_members": len(part),
                "age": part["age"].mean(),
                "visits_per_month": part["visits_per_month"].mean(),
                "avg_visit_minutes": part["avg_visit_minutes"].mean(),
                "classes_per_month": part["classes_per_month"].mean(),
                "months_member": part["months_member"].mean(),
                "monthly_extras_spend": part["monthly_extras_spend"].mean(),
                "share_spend_under_20": (part["monthly_extras_spend"] < SPEND_CUTOFF).mean(),
                "gets_credit": SEGMENT_META[cid]["gets_credit"],
                "traits": SEGMENT_META[cid]["traits"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "segment_summary.csv", index=False)
    centers_orig.to_csv(OUT_DIR / "cluster_centers.csv")
    pd.DataFrame(
        {"k": ks, "train_inertia": train_inertias, "test_inertia": test_inertias}
    ).to_csv(OUT_DIR / "elbow_inertias.csv", index=False)
    df_all.to_csv(OUT_DIR / "members_with_segments.csv", index=False)

    with open(OUT_DIR / "slide_numbers.txt", "w") as f:
        f.write(f"n_members={len(df_all)}\n")
        f.write(f"n_train={len(train)}\n")
        f.write(f"n_test={len(test)}\n")
        f.write(f"mean_extras={df['monthly_extras_spend'].mean():.2f}\n")
        f.write(f"chosen_k={chosen_k}\n")
        f.write(f"test_inertia_k1={k1_test:.1f}\n")
        f.write(f"test_inertia_vk={vk_test:.1f}\n")
        f.write(f"baseline_net={baseline_net:.0f}\n")
        f.write(f"v1_net={v1_net:.0f}\n")
        f.write(f"v1_minus_baseline={v1_net - baseline_net:.0f}\n")
        f.write(f"n_credit_v1={n_credit_v1}\n")
        f.write(f"new_member_segment={decision['segment']}\n")
        f.write(f"new_member_gets_credit={decision['gets_credit']}\n")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = {0: "#1f4e79", 1: "#5b9bd5", 2: "#c45911", 3: "#70ad47"}
    for cid, part in df_all.groupby("cluster"):
        ax.scatter(
            part["visits_per_month"],
            part["monthly_extras_spend"],
            s=22,
            alpha=0.75,
            color=colors.get(int(cid), "gray"),
            label=SEGMENT_META[int(cid)]["name"],
        )
    ax.axhline(SPEND_CUTOFF, color="#888888", linestyle="--", linewidth=1)
    ax.set_xlabel("Visits per month")
    ax.set_ylabel("Monthly extras spend ($)")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "visits_vs_extras.png", dpi=200)
    plt.close(fig)

    print(f"\nWrote figures to {FIG_DIR}")
    print(f"Wrote tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
