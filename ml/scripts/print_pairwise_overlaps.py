import json

with open('c:/STATISTICS & CV/robustlens/results/matched_fidelity_study/empirical_common_support_overlaps.json') as f:
    d = json.load(f)

print('Pairwise Overlaps Table (15 Pairs):')
print(f"{'Pair':<12} | {'Final Ops':<16} | {'Same Final?':<11} | {'Overlap Area':<12} | {'Jaccard':<8} | {'PSNR Range (dB)':<18} | {'SSIM Range':<16}")
print('-'*110)
for k, v in d['pairwise_overlaps_15'].items():
    b = v['overlap_bounds']
    p_str = f"[{b.get('psnr_min', 'N/A')}, {b.get('psnr_max', 'N/A')}]" if b else 'None'
    s_str = f"[{b.get('ssim_min', 'N/A')}, {b.get('ssim_max', 'N/A')}]" if b else 'None'
    same_str = 'YES' if v['same_final_op'] else 'NO'
    ops = f"{v['final_op_1']} vs {v['final_op_2']}"
    print(f"{k:<12} | {ops:<16} | {same_str:<11} | {v['overlap_area']:<12.3f} | {v['jaccard_overlap_index']:<8.3f} | {p_str:<18} | {s_str:<16}")
