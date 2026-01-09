"""
Tail-Conditional KS Distance (T-cKS) 실험 파이프라인
=====================================================
보험 합성 데이터 평가를 위한 T-cKS 지표 검증 실험

Author: [Your Name]
Date: 2026-01
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import arff
import warnings
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import json
from pathlib import Path

warnings.filterwarnings('ignore')

# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """실험 설정을 관리하는 데이터 클래스"""
    # Data settings
    data_path: str = "dataset.arff"
    loss_col: str = "loss"
    condition_col: str = "cat79"  # 조건 변수 (나중에 탐색 후 변경 가능)
    
    # Tail settings
    q_values: Tuple[float, ...] = (0.90, 0.95, 0.99)
    min_tail_n: int = 30  # 최소 tail 샘플 수
    
    # Distortion settings
    distortion_methods: Tuple[str, ...] = ("winsorization", "thinning")
    distortion_strengths: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    
    # Experiment settings
    random_seeds: Tuple[int, ...] = (42, 123, 456, 789, 1024)
    
    # Output settings
    output_dir: str = "results"


# =============================================================================
# 2. Data Loading & Preprocessing
# =============================================================================

def load_arff_data(file_path: str) -> pd.DataFrame:
    """
    ARFF 파일을 DataFrame으로 로드
    
    Args:
        file_path: ARFF 파일 경로
        
    Returns:
        pd.DataFrame: 로드된 데이터프레임
    """
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    
    # bytes를 string으로 변환 (ARFF 범주형 변수)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode('utf-8')
    
    return df


def analyze_loss_distribution(df: pd.DataFrame, loss_col: str = "loss") -> Dict:
    """
    손실 분포 분석 (heavy-tailed 여부 확인)
    
    Args:
        df: 데이터프레임
        loss_col: 손실 변수 컬럼명
        
    Returns:
        Dict: 분포 통계량
    """
    loss = df[loss_col]
    
    stats_dict = {
        "n": len(loss),
        "mean": loss.mean(),
        "median": loss.median(),
        "std": loss.std(),
        "min": loss.min(),
        "max": loss.max(),
        "skewness": stats.skew(loss),
        "kurtosis": stats.kurtosis(loss),
        "quantiles": {
            "q50": loss.quantile(0.50),
            "q75": loss.quantile(0.75),
            "q90": loss.quantile(0.90),
            "q95": loss.quantile(0.95),
            "q99": loss.quantile(0.99),
            "q999": loss.quantile(0.999),
        }
    }
    
    # Heavy-tailed 지표: 평균 > 중앙값, 높은 skewness, 높은 kurtosis
    stats_dict["is_heavy_tailed"] = (
        stats_dict["skewness"] > 1.0 and 
        stats_dict["kurtosis"] > 3.0
    )
    
    return stats_dict


def select_condition_variable(df: pd.DataFrame, 
                               candidate_cols: List[str],
                               loss_col: str = "loss",
                               min_group_size: int = 1000) -> Dict:
    """
    조건 변수 후보들을 분석하여 적합한 변수 선택
    
    Args:
        df: 데이터프레임
        candidate_cols: 후보 조건 변수 리스트
        loss_col: 손실 변수 컬럼명
        min_group_size: 최소 그룹 크기
        
    Returns:
        Dict: 각 후보의 분석 결과
    """
    results = {}
    
    for col in candidate_cols:
        if col not in df.columns:
            continue
            
        group_stats = df.groupby(col)[loss_col].agg(['count', 'mean', 'std'])
        n_groups = len(group_stats)
        min_count = group_stats['count'].min()
        
        # 그룹 간 평균 차이 (조건부 분포 차이 존재 여부)
        group_means = group_stats['mean'].values
        mean_range = group_means.max() - group_means.min()
        
        results[col] = {
            "n_groups": n_groups,
            "min_group_count": min_count,
            "group_counts": group_stats['count'].to_dict(),
            "group_means": group_stats['mean'].to_dict(),
            "mean_range": mean_range,
            "suitable": (n_groups >= 2 and n_groups <= 10 and min_count >= min_group_size)
        }
    
    return results


# =============================================================================
# 3. Metric Functions
# =============================================================================

def compute_mks(Y_real: np.ndarray, Y_synth: np.ndarray) -> float:
    """
    Marginal KS Distance: 전체 Y 분포 비교
    
    Args:
        Y_real: 실제 데이터의 손실값
        Y_synth: 합성(왜곡) 데이터의 손실값
        
    Returns:
        float: KS 통계량
    """
    ks_stat, _ = stats.ks_2samp(Y_real, Y_synth)
    return ks_stat


def compute_cks(df_real: pd.DataFrame, 
                df_synth: pd.DataFrame,
                loss_col: str = "loss",
                condition_col: str = "cat79",
                aggregation: str = "weighted") -> Tuple[float, Dict]:
    """
    Conditional KS Distance: 조건별 전체 분포 비교 후 가중 평균
    
    Args:
        df_real: 실제 데이터
        df_synth: 합성(왜곡) 데이터
        loss_col: 손실 변수 컬럼명
        condition_col: 조건 변수 컬럼명
        aggregation: 집계 방법 ("weighted" or "max")
        
    Returns:
        Tuple[float, Dict]: 집계된 cKS 값과 조건별 상세 결과
    """
    conditions = df_real[condition_col].unique()
    ks_values = []
    weights = []
    details = {}
    
    for z in conditions:
        Y_real_z = df_real[df_real[condition_col] == z][loss_col].values
        Y_synth_z = df_synth[df_synth[condition_col] == z][loss_col].values
        
        if len(Y_real_z) == 0 or len(Y_synth_z) == 0:
            continue
            
        ks_stat, _ = stats.ks_2samp(Y_real_z, Y_synth_z)
        ks_values.append(ks_stat)
        weights.append(len(Y_real_z))
        
        details[z] = {
            "ks": ks_stat,
            "n_real": len(Y_real_z),
            "n_synth": len(Y_synth_z)
        }
    
    if not ks_values:
        return np.nan, details
    
    if aggregation == "weighted":
        cks = np.average(ks_values, weights=weights)
    else:  # max
        cks = max(ks_values)
    
    return cks, details


def compute_tcks(df_real: pd.DataFrame,
                 df_synth: pd.DataFrame,
                 tau_q: float,
                 loss_col: str = "loss",
                 condition_col: str = "cat79",
                 min_tail_n: int = 30,
                 aggregation: str = "weighted") -> Tuple[float, Dict]:
    """
    Tail-Conditional KS Distance: 조건별 tail 분포만 비교
    
    ★ 핵심 제안 지표 ★
    
    Args:
        df_real: 실제 데이터
        df_synth: 합성(왜곡) 데이터
        tau_q: tail threshold (실제 데이터 기준으로 계산된 값)
        loss_col: 손실 변수 컬럼명
        condition_col: 조건 변수 컬럼명
        min_tail_n: 최소 tail 샘플 수
        aggregation: 집계 방법 ("weighted" or "max")
        
    Returns:
        Tuple[float, Dict]: 집계된 T-cKS 값과 조건별 상세 결과
    """
    conditions = df_real[condition_col].unique()
    ks_values = []
    weights = []
    details = {}
    skipped_conditions = []
    
    for z in conditions:
        # Tail subset 필터링 (동일한 tau_q 사용!)
        Y_real_tail = df_real[
            (df_real[condition_col] == z) & (df_real[loss_col] > tau_q)
        ][loss_col].values
        
        Y_synth_tail = df_synth[
            (df_synth[condition_col] == z) & (df_synth[loss_col] > tau_q)
        ][loss_col].values
        
        # 최소 샘플 수 체크
        if len(Y_real_tail) < min_tail_n or len(Y_synth_tail) < min_tail_n:
            skipped_conditions.append(z)
            details[z] = {
                "ks": np.nan,
                "n_real_tail": len(Y_real_tail),
                "n_synth_tail": len(Y_synth_tail),
                "skipped": True,
                "reason": "insufficient_tail_samples"
            }
            continue
        
        ks_stat, _ = stats.ks_2samp(Y_real_tail, Y_synth_tail)
        ks_values.append(ks_stat)
        weights.append(len(Y_real_tail))  # tail sample size weight
        
        details[z] = {
            "ks": ks_stat,
            "n_real_tail": len(Y_real_tail),
            "n_synth_tail": len(Y_synth_tail),
            "skipped": False
        }
    
    if not ks_values:
        return np.nan, details
    
    if aggregation == "weighted":
        tcks = np.average(ks_values, weights=weights)
    else:  # max
        tcks = max(ks_values)
    
    return tcks, details


# =============================================================================
# 4. Failure Mode Injection
# =============================================================================

def inject_tail_distortion(base_df: pd.DataFrame,
                           target_condition: str,
                           tau_q: float,
                           method: str,
                           strength: float,
                           loss_col: str = "loss",
                           condition_col: str = "cat79",
                           random_seed: int = 42) -> pd.DataFrame:
    """
    특정 조건의 tail 분포만 왜곡
    
    ★ 핵심: base_df는 절대 수정하지 않음 ★
    
    Args:
        base_df: 원본 데이터 (수정 안 함)
        target_condition: 왜곡 대상 조건 값 (예: "A")
        tau_q: tail threshold
        method: 왜곡 방법 ("winsorization" or "thinning")
        strength: 왜곡 강도 (0.0 ~ 1.0)
        loss_col: 손실 변수 컬럼명
        condition_col: 조건 변수 컬럼명
        random_seed: 랜덤 시드
        
    Returns:
        pd.DataFrame: 왜곡된 데이터 (copy)
    """
    np.random.seed(random_seed)
    distorted_df = base_df.copy()
    
    # 대상 마스크: 특정 조건 AND tail 영역
    mask = (distorted_df[condition_col] == target_condition) & \
           (distorted_df[loss_col] > tau_q)
    
    n_target = mask.sum()
    
    if n_target == 0 or strength == 0.0:
        return distorted_df
    
    target_indices = distorted_df[mask].index.tolist()
    n_to_distort = int(n_target * strength)
    
    if n_to_distort == 0:
        return distorted_df
    
    # 왜곡할 샘플 선택
    distort_indices = np.random.choice(target_indices, size=n_to_distort, replace=False)
    
    if method == "winsorization":
        # Tail 값을 tau_q로 절단 (상한 적용)
        distorted_df.loc[distort_indices, loss_col] = tau_q
        
    elif method == "thinning":
        # Tail 샘플을 같은 조건의 non-tail에서 재샘플링으로 대체
        non_tail_pool = distorted_df[
            (distorted_df[condition_col] == target_condition) & 
            (distorted_df[loss_col] <= tau_q)
        ]
        
        if len(non_tail_pool) > 0:
            replacement_values = non_tail_pool[loss_col].sample(
                n=n_to_distort, replace=True, random_state=random_seed
            ).values
            distorted_df.loc[distort_indices, loss_col] = replacement_values
    
    return distorted_df


# =============================================================================
# 5. Experiment Runner
# =============================================================================

def run_single_experiment(base_df: pd.DataFrame,
                          config: ExperimentConfig,
                          target_condition: str,
                          q: float,
                          method: str,
                          strength: float,
                          seed: int) -> Dict:
    """
    단일 실험 실행
    
    Args:
        base_df: 원본 데이터
        config: 실험 설정
        target_condition: 왜곡 대상 조건
        q: quantile for tail threshold
        method: 왜곡 방법
        strength: 왜곡 강도
        seed: 랜덤 시드
        
    Returns:
        Dict: 실험 결과
    """
    loss_col = config.loss_col
    condition_col = config.condition_col
    
    # Tail threshold 계산 (base 데이터 기준!)
    tau_q = base_df[loss_col].quantile(q)
    
    # 왜곡 데이터 생성
    distorted_df = inject_tail_distortion(
        base_df=base_df,
        target_condition=target_condition,
        tau_q=tau_q,
        method=method,
        strength=strength,
        loss_col=loss_col,
        condition_col=condition_col,
        random_seed=seed
    )
    
    # 메트릭 계산
    mks = compute_mks(base_df[loss_col].values, distorted_df[loss_col].values)
    cks, cks_details = compute_cks(base_df, distorted_df, loss_col, condition_col)
    tcks, tcks_details = compute_tcks(
        base_df, distorted_df, tau_q, loss_col, condition_col, config.min_tail_n
    )
    
    # 샘플 수 정보
    n_total = len(base_df)
    n_tail_total = (base_df[loss_col] > tau_q).sum()
    n_tail_target = (
        (base_df[condition_col] == target_condition) & 
        (base_df[loss_col] > tau_q)
    ).sum()
    
    return {
        "seed": seed,
        "method": method,
        "strength": strength,
        "q": q,
        "tau_q": tau_q,
        "target_condition": target_condition,
        "mKS": mks,
        "cKS": cks,
        "T-cKS": tcks,
        "n_total": n_total,
        "n_tail_total": n_tail_total,
        "n_tail_target": n_tail_target,
        "cks_details": cks_details,
        "tcks_details": tcks_details
    }


def run_experiment_1(base_df: pd.DataFrame, 
                     config: ExperimentConfig,
                     target_condition: str) -> pd.DataFrame:
    """
    Experiment 1: Controlled Failure Mode Simulation
    
    전체 실험 그리드 실행
    
    Args:
        base_df: 원본 데이터
        config: 실험 설정
        target_condition: 왜곡 대상 조건
        
    Returns:
        pd.DataFrame: 실험 결과 (long format)
    """
    results = []
    run_id = 0
    
    total_runs = (
        len(config.distortion_methods) * 
        len(config.distortion_strengths) * 
        len(config.q_values) * 
        len(config.random_seeds)
    )
    
    print(f"Total experiments to run: {total_runs}")
    print("-" * 50)
    
    for method in config.distortion_methods:
        for strength in config.distortion_strengths:
            for q in config.q_values:
                for seed in config.random_seeds:
                    run_id += 1
                    
                    result = run_single_experiment(
                        base_df=base_df,
                        config=config,
                        target_condition=target_condition,
                        q=q,
                        method=method,
                        strength=strength,
                        seed=seed
                    )
                    
                    result["run_id"] = run_id
                    
                    # 상세 정보 제외하고 저장
                    result_clean = {k: v for k, v in result.items() 
                                   if k not in ["cks_details", "tcks_details"]}
                    results.append(result_clean)
                    
                    if run_id % 20 == 0:
                        print(f"Progress: {run_id}/{total_runs} runs completed")
    
    print("-" * 50)
    print(f"Experiment 1 completed: {total_runs} runs")
    
    return pd.DataFrame(results)


# =============================================================================
# 6. Main Execution
# =============================================================================

def main():
    """메인 실행 함수"""
    
    print("=" * 60)
    print("T-cKS Experiment Pipeline")
    print("=" * 60)
    
    # 설정 초기화
    config = ExperimentConfig()
    
    # 1. 데이터 로드
    print("\n[1] Loading data...")
    df = load_arff_data(config.data_path)
    print(f"    Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # 2. 손실 분포 분석
    print("\n[2] Analyzing loss distribution...")
    loss_stats = analyze_loss_distribution(df, config.loss_col)
    
    print(f"    Samples: {loss_stats['n']:,}")
    print(f"    Mean: {loss_stats['mean']:.2f}")
    print(f"    Median: {loss_stats['median']:.2f}")
    print(f"    Std: {loss_stats['std']:.2f}")
    print(f"    Min: {loss_stats['min']:.2f}")
    print(f"    Max: {loss_stats['max']:.2f}")
    print(f"    Skewness: {loss_stats['skewness']:.2f}")
    print(f"    Kurtosis: {loss_stats['kurtosis']:.2f}")
    print(f"    Heavy-tailed: {loss_stats['is_heavy_tailed']}")
    print(f"    Quantiles:")
    for q_name, q_val in loss_stats['quantiles'].items():
        print(f"      {q_name}: {q_val:.2f}")
    
    # 3. 조건 변수 분석
    print("\n[3] Analyzing condition variables...")
    candidate_cols = ["cat79", "cat80", "cat81", "cat82", "cat83"]
    cond_analysis = select_condition_variable(df, candidate_cols, config.loss_col)
    
    print("\n    Candidate Analysis:")
    for col, info in cond_analysis.items():
        status = "✓ Suitable" if info['suitable'] else "✗ Not suitable"
        print(f"    {col}: {info['n_groups']} groups, "
              f"min_count={info['min_group_count']}, "
              f"mean_range={info['mean_range']:.2f} [{status}]")
    
    # 적합한 조건 변수 선택
    suitable_cols = [col for col, info in cond_analysis.items() if info['suitable']]
    if suitable_cols:
        config.condition_col = suitable_cols[0]
        print(f"\n    Selected condition variable: {config.condition_col}")
    
    # 타겟 조건 선택 (가장 샘플 수가 적은 그룹 선택 - 효과가 더 잘 보임)
    selected_cond_info = cond_analysis[config.condition_col]
    target_condition = min(selected_cond_info['group_counts'], 
                          key=selected_cond_info['group_counts'].get)
    print(f"    Target condition for distortion: {target_condition}")
    
    # 4. Experiment 1 실행
    print("\n[4] Running Experiment 1: Controlled Failure Mode Simulation")
    print("-" * 60)
    
    results_df = run_experiment_1(df, config, target_condition)
    
    # 5. 결과 저장
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    results_path = output_dir / "experiment_1_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n[5] Results saved to: {results_path}")
    
    # 6. 결과 요약
    print("\n[6] Results Summary")
    print("=" * 60)
    
    # 왜곡 강도별 평균
    summary = results_df.groupby(['method', 'strength', 'q'])[['mKS', 'cKS', 'T-cKS']].mean()
    print("\nMean metrics by method, strength, and q:")
    print(summary.to_string())
    
    # 설정 저장
    config_path = output_dir / "experiment_config.json"
    config_dict = {
        "loss_col": config.loss_col,
        "condition_col": config.condition_col,
        "target_condition": target_condition,
        "q_values": list(config.q_values),
        "min_tail_n": config.min_tail_n,
        "distortion_methods": list(config.distortion_methods),
        "distortion_strengths": list(config.distortion_strengths),
        "random_seeds": list(config.random_seeds),
        "loss_stats": {k: v for k, v in loss_stats.items() if k != 'quantiles'},
        "loss_quantiles": loss_stats['quantiles']
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"\nConfig saved to: {config_path}")
    
    print("\n" + "=" * 60)
    print("Experiment completed successfully!")
    print("=" * 60)
    
    return results_df, config


if __name__ == "__main__":
    results, config = main()

