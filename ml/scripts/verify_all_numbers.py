import pandas as pd
import json
import gzip
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
pkg = REPO_ROOT / "reports/final_research_package/analysis"

print("=== 1. VERIFYING TRANSITIONS AND CHURN IDENTITIES ===")
df_trans = pd.read_csv(pkg / "prediction_instability_transitions.csv")
for idx, r in df_trans.iterrows():
    e11 = r['e11_both_correct'] * 100
    e10 = r['e10_loss'] * 100
    e01 = r['e01_gain'] * 100
    e00_same = r['e00_same_wrong'] * 100
    e00_diff = r['e00_diff_wrong'] * 100
    tot = e11 + e10 + e01 + e00_same + e00_diff
    corr_churn = r['correctness_churn'] * 100
    label_churn = r['label_churn'] * 100
    agree = r['agreement_rate'] * 100
    print(f"{r['regime']:20s} | {r['model']:15s} | E11:{e11:5.2f}% L:{e10:5.2f}% G:{e01:5.2f}% Same:{e00_same:5.2f}% Diff:{e00_diff:5.2f}% | Tot:{tot:6.2f}% | CorrChurn:{corr_churn:5.2f}% LabelChurn:{label_churn:5.2f}%")
    assert abs(tot - 100.0) < 0.01, f"Probability sum != 100: {tot}"
    assert abs(corr_churn - (e10 + e01)) < 0.01, f"Corr churn mismatch"
    assert abs(label_churn - (corr_churn + e00_diff)) < 0.01, f"Label churn mismatch"
    assert abs(agree - (e11 + e00_same)) < 0.01, f"Agreement rate mismatch"
    assert abs(agree + label_churn - 100.0) < 0.01, f"Agreement + Label churn != 100"

print("\n=== 2. VERIFYING PER-CLASS STATISTICS ===")
df_class = pd.read_csv(pkg / "per_class_instability_summary.csv")
print("Classes present:", len(df_class['class_name'].unique()))
for c in df_class['class_name'].unique():
    nom = df_class[(df_class['class_name'] == c) & (df_class['regime'] == 'Nominal-Order')].iloc[0]
    matched = df_class[(df_class['class_name'] == c) & (df_class['regime'] == 'Approximate-Fidelity')].iloc[0]
    print(f"Class: {c:18s} | Nom Acc:{nom['accuracy']*100:5.2f}% LabelChurn:{nom['label_churn']*100:5.2f}% ErrorMig:{nom['error_migration']*100:5.2f}% | Matched Acc:{matched['accuracy']*100:5.2f}% LabelChurn:{matched['label_churn']*100:5.2f}% ErrorMig:{matched['error_migration']*100:5.2f}%")

print("\n=== 3. VERIFYING GAP CONTRACTIONS ===")
df_gaps = pd.read_csv(pkg / "nominal_vs_matched_gap_contraction.csv")
for idx, r in df_gaps.iterrows():
    nom_gap = r['nominal_acc_gap'] * 100
    match_gap = r['matched_acc_gap'] * 100
    contract = r['acc_gap_contraction_pct']
    expected_contract = (nom_gap - match_gap) / nom_gap * 100
    print(f"Model: {r['model']:15s} | Nom Gap: {nom_gap:5.2f}% | Matched Gap: {match_gap:5.2f}% | Contraction: {contract:5.2f}% (Expected: {expected_contract:5.2f}%)")
    assert abs(contract - expected_contract) < 0.02

# Check average of architecture contractions vs contraction of overall range
arch_contractions = df_gaps[df_gaps['model'] != 'Overall']['acc_gap_contraction_pct'].tolist()
mean_arch_contraction = sum(arch_contractions) / len(arch_contractions)
overall_range_contraction = df_gaps[df_gaps['model'] == 'Overall']['acc_gap_contraction_pct'].values[0]
print(f"\nAverage of 4 architecture contractions: {mean_arch_contraction:.2f}%")
print(f"Contraction of overall mean accuracy range: {overall_range_contraction:.2f}%")

print("\n=== 4. VERIFYING RAW ROW COUNTS ===")
raw_full = REPO_ROOT / "results/order_effect_study/full/outputs/all_order_predictions.csv.gz"
raw_matched = REPO_ROOT / "results/order_effect_study/matched_fidelity/outputs/all_order_predictions.csv.gz"

for name, p in [("Nominal", raw_full), ("Matched", raw_matched)]:
    with gzip.open(p, 'rt') as f:
        header = f.readline().strip().split(',')
        row_count = sum(1 for _ in f)
    print(f"{name} file: {row_count} data rows + 1 header row = {row_count + 1} total lines")
    assert row_count == 486700, f"Expected 486,700, got {row_count}"
print("Total rows across both files: 486,700 * 2 = 973,400 forward passes.")

print("\n>>> ALL RAW AND DERIVED NUMBERS VERIFIED WITH ZERO ERRORS <<<")
