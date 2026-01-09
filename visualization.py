"""
T-cKS 실험 결과 시각화
======================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

# 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# 색상 팔레트
COLORS = {
    'mKS': '#2ecc71',      # 녹색
    'cKS': '#3498db',      # 파랑
    'T-cKS': '#e74c3c',    # 빨강
}


def plot_metric_comparison(results_df: pd.DataFrame,
                           q: float = 0.95,
                           method: Optional[str] = None,
                           save_path: Optional[str] = None):
    """
    왜곡 강도별 메트릭 비교 플롯
    
    Args:
        results_df: 실험 결과 데이터프레임
        q: 사용할 quantile 값
        method: 특정 방법만 표시 (None이면 모두)
        save_path: 저장 경로
    """
    # 필터링
    df = results_df[results_df['q'] == q].copy()
    if method:
        df = df[df['method'] == method]
    
    # 강도별 평균
    grouped = df.groupby(['method', 'strength'])[['mKS', 'cKS', 'T-cKS']].agg(['mean', 'std'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, m in enumerate(['winsorization', 'thinning']):
        ax = axes[idx]
        
        method_data = grouped.loc[m] if m in grouped.index else None
        if method_data is None:
            continue
        
        strengths = method_data.index.values
        
        for metric, color in COLORS.items():
            means = method_data[(metric, 'mean')].values
            stds = method_data[(metric, 'std')].values
            
            ax.plot(strengths, means, 'o-', label=metric, color=color, linewidth=2, markersize=8)
            ax.fill_between(strengths, means - stds, means + stds, alpha=0.2, color=color)
        
        ax.set_xlabel('Distortion Strength')
        ax.set_ylabel('KS Distance')
        ax.set_title(f'{m.capitalize()} (q={q})')
        ax.legend(loc='upper left')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, None)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_q_sensitivity(results_df: pd.DataFrame,
                       method: str = 'winsorization',
                       strength: float = 0.8,
                       save_path: Optional[str] = None):
    """
    q 값 변화에 따른 T-cKS 민감도
    
    Args:
        results_df: 실험 결과
        method: 왜곡 방법
        strength: 왜곡 강도
        save_path: 저장 경로
    """
    df = results_df[
        (results_df['method'] == method) & 
        (results_df['strength'] == strength)
    ].copy()
    
    grouped = df.groupby('q')[['mKS', 'cKS', 'T-cKS']].agg(['mean', 'std'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    q_values = grouped.index.values
    
    for metric, color in COLORS.items():
        means = grouped[(metric, 'mean')].values
        stds = grouped[(metric, 'std')].values
        
        ax.bar(
            [str(q) for q in q_values], 
            means, 
            yerr=stds,
            label=metric, 
            color=color, 
            alpha=0.7,
            width=0.25
        )
    
    ax.set_xlabel('Quantile (q)')
    ax.set_ylabel('KS Distance')
    ax.set_title(f'Metric Sensitivity to q ({method}, strength={strength})')
    ax.legend()
    
    # x축 위치 조정 (그룹 바 차트)
    x = np.arange(len(q_values))
    width = 0.25
    
    ax.clear()
    for i, (metric, color) in enumerate(COLORS.items()):
        means = grouped[(metric, 'mean')].values
        stds = grouped[(metric, 'std')].values
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, label=metric, color=color, alpha=0.7)
    
    ax.set_xlabel('Quantile (q)')
    ax.set_ylabel('KS Distance')
    ax.set_title(f'Metric Sensitivity to q ({method}, strength={strength})')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{q}' for q in q_values])
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_heatmap(results_df: pd.DataFrame,
                 metric: str = 'T-cKS',
                 method: str = 'winsorization',
                 save_path: Optional[str] = None):
    """
    q vs strength 히트맵
    
    Args:
        results_df: 실험 결과
        metric: 표시할 메트릭
        method: 왜곡 방법
        save_path: 저장 경로
    """
    df = results_df[results_df['method'] == method].copy()
    
    pivot = df.pivot_table(
        values=metric, 
        index='q', 
        columns='strength', 
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    sns.heatmap(
        pivot, 
        annot=True, 
        fmt='.3f', 
        cmap='Reds',
        ax=ax,
        cbar_kws={'label': f'{metric} Distance'}
    )
    
    ax.set_title(f'{metric} Heatmap ({method})')
    ax.set_xlabel('Distortion Strength')
    ax.set_ylabel('Quantile (q)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_detection_comparison(results_df: pd.DataFrame,
                              q: float = 0.95,
                              save_path: Optional[str] = None):
    """
    탐지 능력 비교: T-cKS vs 기존 지표
    
    핵심 논문 그래프 - 왜곡 시 각 지표의 반응 비교
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, method in enumerate(['winsorization', 'thinning']):
        df = results_df[(results_df['q'] == q) & (results_df['method'] == method)]
        grouped = df.groupby('strength')[['mKS', 'cKS', 'T-cKS']].mean()
        
        # 왼쪽: 절대값 비교
        ax1 = axes[idx, 0]
        for metric, color in COLORS.items():
            ax1.plot(grouped.index, grouped[metric], 'o-', 
                    label=metric, color=color, linewidth=2.5, markersize=10)
        
        ax1.set_xlabel('Distortion Strength')
        ax1.set_ylabel('KS Distance')
        ax1.set_title(f'{method.capitalize()}: Absolute Values (q={q})')
        ax1.legend()
        ax1.set_xlim(-0.05, 1.05)
        
        # 오른쪽: 상대 변화율 (strength=0 대비)
        ax2 = axes[idx, 1]
        baseline = grouped.loc[0.0] if 0.0 in grouped.index else grouped.iloc[0]
        
        for metric, color in COLORS.items():
            # 상대 변화율: (current - baseline) / baseline * 100
            if baseline[metric] > 0:
                relative_change = (grouped[metric] - baseline[metric]) / baseline[metric] * 100
            else:
                relative_change = grouped[metric] * 100
            
            ax2.plot(grouped.index, relative_change, 'o-', 
                    label=metric, color=color, linewidth=2.5, markersize=10)
        
        ax2.set_xlabel('Distortion Strength')
        ax2.set_ylabel('Relative Change (%)')
        ax2.set_title(f'{method.capitalize()}: Relative Change from Baseline')
        ax2.legend()
        ax2.set_xlim(-0.05, 1.05)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def create_summary_table(results_df: pd.DataFrame, 
                         save_path: Optional[str] = None) -> pd.DataFrame:
    """
    논문용 요약 테이블 생성
    
    Args:
        results_df: 실험 결과
        save_path: 저장 경로
        
    Returns:
        pd.DataFrame: 요약 테이블
    """
    # 방법 × 강도 × q별 평균 및 표준편차
    summary = results_df.groupby(['method', 'strength', 'q']).agg({
        'mKS': ['mean', 'std'],
        'cKS': ['mean', 'std'],
        'T-cKS': ['mean', 'std']
    }).round(4)
    
    # 컬럼명 정리
    summary.columns = [f'{col[0]}_{col[1]}' for col in summary.columns]
    
    if save_path:
        summary.to_csv(save_path)
        print(f"Summary table saved to: {save_path}")
    
    return summary


def generate_all_figures(results_path: str = "results/experiment_1_results.csv",
                         output_dir: str = "results/figures"):
    """
    모든 시각화 생성
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 결과 로드
    results_df = pd.read_csv(results_path)
    
    print("Generating figures...")
    
    # 1. 메트릭 비교 (각 q별)
    for q in [0.90, 0.95, 0.99]:
        plot_metric_comparison(
            results_df, q=q,
            save_path=str(output_dir / f"metric_comparison_q{q}.png")
        )
    
    # 2. q 민감도
    plot_q_sensitivity(
        results_df, method='winsorization', strength=0.8,
        save_path=str(output_dir / "q_sensitivity_winsor.png")
    )
    plot_q_sensitivity(
        results_df, method='thinning', strength=0.8,
        save_path=str(output_dir / "q_sensitivity_thinning.png")
    )
    
    # 3. 히트맵
    for method in ['winsorization', 'thinning']:
        plot_heatmap(
            results_df, metric='T-cKS', method=method,
            save_path=str(output_dir / f"heatmap_tcks_{method}.png")
        )
    
    # 4. 탐지 비교 (핵심 그래프)
    plot_detection_comparison(
        results_df, q=0.95,
        save_path=str(output_dir / "detection_comparison_q095.png")
    )
    
    # 5. 요약 테이블
    create_summary_table(
        results_df,
        save_path=str(output_dir / "summary_table.csv")
    )
    
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    generate_all_figures()

