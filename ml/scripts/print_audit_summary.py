import json
import pandas as pd

df = pd.read_csv("results/matched_fidelity_study/grouped_cv_splines_comparison.csv")
print("=" * 95)
print("OUT-OF-SAMPLE GROUPED CV AUDIT: LINEAR vs QUADRATIC vs SPLINE vs EXTENDED")
print("=" * 95)

for _, r in df.iterrows():
    arch = r["architecture"]
    print(f"\nArchitecture: {arch.upper()}")
    print("-" * 60)
    print(f"  R2 OOS:   Linear={r['r2_oos_linear']:.4f} | Quad={r['r2_oos_quadratic']:.4f} | Spline={r['r2_oos_spline']:.4f} | Extended={r['r2_oos_extended']:.4f}")
    print(f"  MAE OOS:  Linear={r['mae_oos_linear']:.4f} | Quad={r['mae_oos_quadratic']:.4f} | Spline={r['mae_oos_spline']:.4f} | Extended={r['mae_oos_extended']:.4f}")
    print(f"  Gain vs Quadratic Baseline:")
    print(f"    Delta R2:  {r['delta_r2_vs_quad']:+.4f} [95% CI: {r['ci_delta_r2_quad_low']:+.4f}, {r['ci_delta_r2_quad_high']:+.4f}] (P(gain>0)={r['prop_boot_r2_gain_quad']*100:.1f}%)")
    print(f"    Delta MAE: {r['delta_mae_vs_quad']:+.4f} [95% CI: {r['ci_delta_mae_quad_low']:+.4f}, {r['ci_delta_mae_quad_high']:+.4f}] (P(gain>0)={r['prop_boot_mae_gain_quad']*100:.1f}%, red={r['mae_reduction_pct_quad']:.2f}%)")
    print(f"  Gain vs Spline Baseline:")
    print(f"    Delta R2:  {r['delta_r2_vs_spline']:+.4f} [95% CI: {r['ci_delta_r2_spline_low']:+.4f}, {r['ci_delta_r2_spline_high']:+.4f}] (P(gain>0)={r['prop_boot_r2_gain_spline']*100:.1f}%)")
    print(f"    Delta MAE: {r['delta_mae_vs_spline']:+.4f} [95% CI: {r['ci_delta_mae_spline_low']:+.4f}, {r['ci_delta_mae_spline_high']:+.4f}] (P(gain>0)={r['prop_boot_mae_gain_spline']*100:.1f}%, red={r['mae_reduction_pct_spline']:.2f}%)")
    print(f"  Condition Numbers: Quad={r['mean_kappa_quad']:.1f} | Spline={r['mean_kappa_spline']:.1f} | Ext={r['mean_kappa_ext']:.1f}")

with open("results/matched_fidelity_study/grouped_cv_splines_diagnostics.json", "r") as f:
    diag = json.load(f)

print("\n" + "=" * 95)
print("FOLD-LEVEL RANK AND CONDITION NUMBER AUDIT (FIRST ARCHITECTURE: CONVNEXT):")
print("=" * 95)
for fd in diag["architectures"][0]["fold_diagnostics"]:
    f_num = fd["fold"]
    m1 = fd["model1_linear"]
    m2 = fd["model2_quadratic"]
    m3 = fd["model3_spline"]
    m4 = fd["model4_extended"]
    print(f"Fold {f_num}: Rows Tr={fd['train_rows']}/Te={fd['test_rows']} | Cols: Lin={m1['cols']} (rk {m1['rank']}, k {m1['condition_number']}) | Quad={m2['cols']} (rk {m2['rank']}, k {m2['condition_number']}) | Spline={m3['cols']} (rk {m3['rank']}, k {m3['condition_number']}) | Ext={m4['cols']} (rk {m4['rank']}, k {m4['condition_number']})")
