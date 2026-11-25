# Gym member segments: who should get the monthly offer?

K-means clustering of 330 gym members, then a targeting rule for a $10 extras credit.

## Reproduce the numbers

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python analysis.py
```

`analysis.py` splits the members, scales on the training set only, fits k-means for k = 1 through 8, assigns held-out members to the training centers, names the chosen segments, scores a new member, and writes the monthly net from the credit.

Outputs:

- `figures/elbow.png` — training and test within-group spread vs k
- `figures/visits_vs_extras.png` — visits vs extras spend, colored by segment
- `outputs/elbow_inertias.csv`
- `outputs/cluster_centers.csv`
- `outputs/segment_summary.csv`
- `outputs/members_with_segments.csv`
- `outputs/slide_numbers.txt`

The random split uses `random_state=42` (70% train / 30% test). K-means uses `n_init=10` and `random_state=42`. Every printed number is determined by those seeds.

## Data

`gym_members.csv` — 330 members, one row each, averages over the last 3 months.

Features used: `age`, `visits_per_month`, `avg_visit_minutes`, `classes_per_month`, `months_member`, `monthly_extras_spend`.

Left out: `member_id` (an identifier, not a description of the member).
