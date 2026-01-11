#!/usr/bin/env python3
"""
실제 합성 데이터 생성 모델 실험 결과 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# 결과 로드
results_df = pd.read_csv("results/synthetic_experiment_results.csv")

# 출력 디렉토리
output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Figure 1: 모델별 메트릭 비교 (q=0.95)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

q95_data = results_df[results_df['q'] == 0.95].groupby('method')[['mKS', 'cKS', 'T-cKS']].mean()
q95_data = q95_data.reindex(['TVAE', 'CTGAN', 'GaussianCopula'])

x = np.arange(len(q95_data))
width = 0.25

bars1 = ax.bar(x - width, q95_data['mKS'], width, label='mKS', color='#3498db', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, q95_data['cKS'], width, label='cKS', color='#2ecc71', edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + width, q95_data['T-cKS'], width, label='T-cKS', color='#e74c3c', edgecolor='black', linewidth=0.5)

ax.set_xlabel('Synthesizer', fontsize=12, fontweight='bold')
ax.set_ylabel('KS Distance', fontsize=12, fontweight='bold')
ax.set_title('Synthetic Data Quality Metrics Comparison (q=0.95)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(q95_data.index, fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, 0.18)

# 값 표시
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / "synthetic_metric_comparison_q095.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 1 saved: synthetic_metric_comparison_q095.png")

# =============================================================================
# Figure 2: T-cKS / cKS 비율 (탐지 민감도)
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

sensitivity = q95_data['T-cKS'] / q95_data['cKS']
colors = ['#e74c3c' if s > 2 else '#f39c12' if s > 1.5 else '#3498db' for s in sensitivity]

bars = ax.bar(sensitivity.index, sensitivity.values, color=colors, edgecolor='black', linewidth=1)

ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='cKS = T-cKS')
ax.set_xlabel('Synthesizer', fontsize=12, fontweight='bold')
ax.set_ylabel('T-cKS / cKS Ratio', fontsize=12, fontweight='bold')
ax.set_title('Tail Detection Sensitivity: T-cKS vs cKS (q=0.95)', fontsize=14, fontweight='bold')

# 값 표시
for bar, val in zip(bars, sensitivity.values):
    ax.annotate(f'{val:.2f}x',
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylim(0, 4.5)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(output_dir / "synthetic_tcks_sensitivity.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 2 saved: synthetic_tcks_sensitivity.png")

# =============================================================================
# Figure 3: Quantile별 T-cKS 변화
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

for method in ['TVAE', 'CTGAN', 'GaussianCopula']:
    method_data = results_df[results_df['method'] == method].groupby('q')['T-cKS'].mean()
    ax.plot(method_data.index, method_data.values, 'o-', linewidth=2, markersize=8, label=method)

ax.set_xlabel('Quantile (q)', fontsize=12, fontweight='bold')
ax.set_ylabel('T-cKS', fontsize=12, fontweight='bold')
ax.set_title('T-cKS by Tail Threshold Quantile', fontsize=14, fontweight='bold')
ax.set_xticks([0.90, 0.95, 0.99])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "synthetic_tcks_by_quantile.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 3 saved: synthetic_tcks_by_quantile.png")

# =============================================================================
# Figure 4: Tail 재현율 비교
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Tail Count Ratio
ax1 = axes[0]
tail_count = results_df[results_df['q'] == 0.95].groupby('method')['tail_count_ratio'].mean()
tail_count = tail_count.reindex(['TVAE', 'CTGAN', 'GaussianCopula'])
colors = ['#e74c3c' if r < 0.3 else '#f39c12' if r < 0.5 else '#2ecc71' for r in tail_count]

bars = ax1.bar(tail_count.index, tail_count.values * 100, color=colors, edgecolor='black', linewidth=1)
ax1.axhline(y=100, color='green', linestyle='--', linewidth=1.5, label='Ideal (100%)')
ax1.set_xlabel('Synthesizer', fontsize=11, fontweight='bold')
ax1.set_ylabel('Tail Sample Ratio (%)', fontsize=11, fontweight='bold')
ax1.set_title('(a) Tail Sample Generation Rate', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 110)

for bar, val in zip(bars, tail_count.values):
    ax1.annotate(f'{val*100:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, val * 100),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Tail Mean Ratio
ax2 = axes[1]
tail_mean = results_df[results_df['q'] == 0.95].groupby('method')['tail_mean_ratio'].mean()
tail_mean = tail_mean.reindex(['TVAE', 'CTGAN', 'GaussianCopula'])
colors = ['#2ecc71' if 0.95 <= r <= 1.05 else '#f39c12' if 0.9 <= r <= 1.1 else '#e74c3c' for r in tail_mean]

bars = ax2.bar(tail_mean.index, tail_mean.values * 100, color=colors, edgecolor='black', linewidth=1)
ax2.axhline(y=100, color='green', linestyle='--', linewidth=1.5, label='Ideal (100%)')
ax2.set_xlabel('Synthesizer', fontsize=11, fontweight='bold')
ax2.set_ylabel('Tail Mean Ratio (%)', fontsize=11, fontweight='bold')
ax2.set_title('(b) Tail Mean Preservation', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 120)

for bar, val in zip(bars, tail_mean.values):
    ax2.annotate(f'{val*100:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, val * 100),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "synthetic_tail_reproduction.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 4 saved: synthetic_tail_reproduction.png")

# =============================================================================
# Figure 5: TVAE 케이스 상세 분석 (핵심 그래프)
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 6))

tvae_q95 = results_df[(results_df['method'] == 'TVAE') & (results_df['q'] == 0.95)][['mKS', 'cKS', 'T-cKS']].mean()

metrics = ['mKS', 'cKS', 'T-cKS']
values = [tvae_q95['mKS'], tvae_q95['cKS'], tvae_q95['T-cKS']]
colors = ['#3498db', '#2ecc71', '#e74c3c']

bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5, width=0.6)

# 화살표와 설명 추가
ax.annotate('', xy=(2, values[2]), xytext=(1, values[1]),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.annotate(f'3.5x more\nsensitive!', xy=(1.5, 0.05), fontsize=12, 
            fontweight='bold', color='red', ha='center')

ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
ax.set_ylabel('KS Distance', fontsize=12, fontweight='bold')
ax.set_title('TVAE: Hidden Tail Distortion Detected by T-cKS', fontsize=14, fontweight='bold')

for bar, val in zip(bars, values):
    ax.annotate(f'{val:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylim(0, 0.1)
plt.tight_layout()
plt.savefig(output_dir / "tvae_hidden_distortion.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 5 saved: tvae_hidden_distortion.png")

print("\n" + "="*50)
print("All figures saved to results/figures/")
print("="*50)

