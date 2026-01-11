#!/usr/bin/env python3
"""
누락된 그림 파일 생성
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.io import arff
from pathlib import Path

# 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# 출력 디렉토리
output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# 데이터 로드
def load_arff_data(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode('utf-8')
    return df

print("Loading data...")
df = load_arff_data("dataset.arff")
loss = df['loss'].values

# =============================================================================
# 메트릭 계산 함수
# =============================================================================
def compute_mks(Y_real, Y_synth):
    ks_stat, _ = stats.ks_2samp(Y_real, Y_synth)
    return ks_stat

def compute_cks(df_real, df_synth, loss_col="loss", condition_col="cat79"):
    conditions = df_real[condition_col].unique()
    ks_values, weights = [], []
    for z in conditions:
        Y_real_z = df_real[df_real[condition_col] == z][loss_col].values
        Y_synth_z = df_synth[df_synth[condition_col] == z][loss_col].values
        if len(Y_real_z) == 0 or len(Y_synth_z) == 0:
            continue
        ks_stat, _ = stats.ks_2samp(Y_real_z, Y_synth_z)
        ks_values.append(ks_stat)
        weights.append(len(Y_real_z))
    return np.average(ks_values, weights=weights) if ks_values else np.nan

def compute_tcks(df_real, df_synth, tau_q, loss_col="loss", condition_col="cat79", min_tail_n=30):
    conditions = df_real[condition_col].unique()
    ks_values, weights = [], []
    for z in conditions:
        Y_real_tail = df_real[(df_real[condition_col] == z) & (df_real[loss_col] > tau_q)][loss_col].values
        Y_synth_tail = df_synth[(df_synth[condition_col] == z) & (df_synth[loss_col] > tau_q)][loss_col].values
        if len(Y_real_tail) < min_tail_n or len(Y_synth_tail) < min_tail_n:
            continue
        ks_stat, _ = stats.ks_2samp(Y_real_tail, Y_synth_tail)
        ks_values.append(ks_stat)
        weights.append(len(Y_real_tail))
    return np.average(ks_values, weights=weights) if ks_values else np.nan

def inject_tail_scaling(df, scale_factor, q, target_condition="D", loss_col="loss", condition_col="cat79"):
    df_distorted = df.copy()
    tau_q = df[loss_col].quantile(q)
    mask = (df_distorted[condition_col] == target_condition) & (df_distorted[loss_col] > tau_q)
    df_distorted.loc[mask, loss_col] = tau_q + (df_distorted.loc[mask, loss_col] - tau_q) * scale_factor
    return df_distorted

# =============================================================================
# Figure: metric_comparison_q0.95.png
# =============================================================================
print("Generating metric_comparison_q0.95.png...")
q = 0.95
tau_q = df['loss'].quantile(q)
scale_factors = [1.0, 0.8, 0.6, 0.4, 0.2]

results = []
for sf in scale_factors:
    df_synth = inject_tail_scaling(df, sf, q)
    mks = compute_mks(df['loss'].values, df_synth['loss'].values)
    cks = compute_cks(df, df_synth)
    tcks = compute_tcks(df, df_synth, tau_q)
    results.append({'scale': sf, 'mKS': mks, 'cKS': cks, 'T-cKS': tcks})

results_df = pd.DataFrame(results)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(scale_factors))
width = 0.25

ax.bar(x - width, results_df['mKS'], width, label='mKS', color='#3498db', edgecolor='black')
ax.bar(x, results_df['cKS'], width, label='cKS', color='#2ecc71', edgecolor='black')
ax.bar(x + width, results_df['T-cKS'], width, label='T-cKS', color='#e74c3c', edgecolor='black')

ax.set_xlabel('Scale Factor (Tail Compression)', fontsize=12, fontweight='bold')
ax.set_ylabel('KS Distance', fontsize=12, fontweight='bold')
ax.set_title(f'Metric Comparison by Distortion Strength (q={q})', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{sf:.1f}' for sf in scale_factors])
ax.legend(fontsize=10)
ax.set_ylim(0, 0.4)

plt.tight_layout()
plt.savefig(output_dir / "metric_comparison_q0.95.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ metric_comparison_q0.95.png saved")

# =============================================================================
# Figure: q_sensitivity_winsor.png (Quantile sensitivity)
# =============================================================================
print("Generating q_sensitivity_winsor.png...")
q_values = [0.85, 0.90, 0.95, 0.99]
scale_factor = 0.2

q_results = []
for q in q_values:
    tau_q = df['loss'].quantile(q)
    df_synth = inject_tail_scaling(df, scale_factor, q)
    tcks = compute_tcks(df, df_synth, tau_q)
    cks = compute_cks(df, df_synth)
    mks = compute_mks(df['loss'].values, df_synth['loss'].values)
    q_results.append({'q': q, 'mKS': mks, 'cKS': cks, 'T-cKS': tcks})

q_results_df = pd.DataFrame(q_results)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(q_results_df['q'], q_results_df['mKS'], 'o-', label='mKS', linewidth=2, markersize=8)
ax.plot(q_results_df['q'], q_results_df['cKS'], 's-', label='cKS', linewidth=2, markersize=8)
ax.plot(q_results_df['q'], q_results_df['T-cKS'], '^-', label='T-cKS', linewidth=2, markersize=8, color='#e74c3c')

ax.set_xlabel('Quantile (q)', fontsize=12, fontweight='bold')
ax.set_ylabel('KS Distance', fontsize=12, fontweight='bold')
ax.set_title(f'Quantile Sensitivity Analysis (scale={scale_factor})', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "q_sensitivity_winsor.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ q_sensitivity_winsor.png saved")

# =============================================================================
# Figure: detection_comparison_q095.png
# =============================================================================
print("Generating detection_comparison_q095.png...")
q = 0.95
tau_q = df['loss'].quantile(q)
df_synth = inject_tail_scaling(df, 0.2, q)

# 조건별 계산
conditions = ['A', 'B', 'C', 'D']
cks_by_cond = []
tcks_by_cond = []

for z in conditions:
    Y_real_z = df[df['cat79'] == z]['loss'].values
    Y_synth_z = df_synth[df_synth['cat79'] == z]['loss'].values
    ks_stat, _ = stats.ks_2samp(Y_real_z, Y_synth_z)
    cks_by_cond.append(ks_stat)
    
    Y_real_tail = df[(df['cat79'] == z) & (df['loss'] > tau_q)]['loss'].values
    Y_synth_tail = df_synth[(df_synth['cat79'] == z) & (df_synth['loss'] > tau_q)]['loss'].values
    if len(Y_real_tail) >= 30 and len(Y_synth_tail) >= 30:
        ks_tail, _ = stats.ks_2samp(Y_real_tail, Y_synth_tail)
        tcks_by_cond.append(ks_tail)
    else:
        tcks_by_cond.append(0)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(conditions))
width = 0.35

ax.bar(x - width/2, cks_by_cond, width, label='Conditional KS', color='#2ecc71', edgecolor='black')
ax.bar(x + width/2, tcks_by_cond, width, label='Tail-Conditional KS', color='#e74c3c', edgecolor='black')

ax.set_xlabel('Condition (cat79)', fontsize=12, fontweight='bold')
ax.set_ylabel('KS Distance', fontsize=12, fontweight='bold')
ax.set_title('Detection Comparison: cKS vs T-cKS by Condition (q=0.95, scale=0.2)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.legend(fontsize=10)

# D 조건 강조
ax.annotate('Distorted!', xy=(3, max(tcks_by_cond[3], cks_by_cond[3]) + 0.02), 
            fontsize=11, fontweight='bold', color='red', ha='center')

plt.tight_layout()
plt.savefig(output_dir / "detection_comparison_q095.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ detection_comparison_q095.png saved")

# =============================================================================
# Figure: heatmap_tcks_winsorization.png
# =============================================================================
print("Generating heatmap_tcks_winsorization.png...")
q_values = [0.85, 0.90, 0.95, 0.99]
scale_factors = [1.0, 0.8, 0.6, 0.4, 0.2]

heatmap_data = np.zeros((len(q_values), len(scale_factors)))

for i, q in enumerate(q_values):
    tau_q = df['loss'].quantile(q)
    for j, sf in enumerate(scale_factors):
        df_synth = inject_tail_scaling(df, sf, q)
        tcks = compute_tcks(df, df_synth, tau_q)
        heatmap_data[i, j] = tcks if not np.isnan(tcks) else 0

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='Reds',
            xticklabels=[f'{sf:.1f}' for sf in scale_factors],
            yticklabels=[f'{q:.2f}' for q in q_values],
            ax=ax, cbar_kws={'label': 'T-cKS'})

ax.set_xlabel('Scale Factor', fontsize=12, fontweight='bold')
ax.set_ylabel('Quantile (q)', fontsize=12, fontweight='bold')
ax.set_title('T-cKS Heatmap: Quantile vs Distortion Strength', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "heatmap_tcks_winsorization.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ heatmap_tcks_winsorization.png saved")

print("\n" + "="*50)
print("All missing figures generated!")
print("="*50)

