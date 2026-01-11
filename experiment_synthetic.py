#!/usr/bin/env python3
"""
T-cKS 실험: 실제 합성 데이터 생성 모델 평가
=============================================
CTGAN, TVAE, GaussianCopula 등 실제 합성 데이터 생성 모델을 사용하여
T-cKS 지표의 유용성을 검증합니다.

Author: 문덕룡 (Deok Lyong Moon)
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
import time

warnings.filterwarnings('ignore')

# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass
class SyntheticExperimentConfig:
    """실험 설정"""
    # Data settings
    data_path: str = "dataset.arff"
    loss_col: str = "loss"
    condition_col: str = "cat79"
    
    # Columns to use for synthesis (loss + condition + some categorical)
    synthesis_cols: Tuple[str, ...] = ("loss", "cat79", "cat80", "cat81")
    
    # Tail settings
    q_values: Tuple[float, ...] = (0.90, 0.95, 0.99)
    min_tail_n: int = 30
    
    # Synthetic data generation settings
    synthesizers: Tuple[str, ...] = ("GaussianCopula", "CTGAN", "TVAE")
    n_synthetic_samples: int = 50000  # 생성할 합성 데이터 샘플 수
    
    # CTGAN/TVAE settings
    epochs: int = 30
    batch_size: int = 500
    
    # Experiment settings
    random_seeds: Tuple[int, ...] = (42, 123, 456)
    
    # Output settings
    output_dir: str = "results"


# =============================================================================
# 2. Data Loading
# =============================================================================

def load_arff_data(file_path: str) -> pd.DataFrame:
    """ARFF 파일을 DataFrame으로 로드"""
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode('utf-8')
    
    return df


def prepare_data_for_synthesis(df: pd.DataFrame, 
                                cols: Tuple[str, ...]) -> pd.DataFrame:
    """합성 데이터 생성을 위한 데이터 준비"""
    subset = df[list(cols)].copy()
    return subset


# =============================================================================
# 3. Metric Functions (기존과 동일)
# =============================================================================

def compute_mks(Y_real: np.ndarray, Y_synth: np.ndarray) -> float:
    """Marginal KS Distance"""
    ks_stat, _ = stats.ks_2samp(Y_real, Y_synth)
    return ks_stat


def compute_cks(df_real: pd.DataFrame, 
                df_synth: pd.DataFrame,
                loss_col: str = "loss",
                condition_col: str = "cat79") -> Tuple[float, Dict]:
    """Conditional KS Distance"""
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
    
    cks = np.average(ks_values, weights=weights)
    return cks, details


def compute_tcks(df_real: pd.DataFrame,
                 df_synth: pd.DataFrame,
                 tau_q: float,
                 loss_col: str = "loss",
                 condition_col: str = "cat79",
                 min_tail_n: int = 30) -> Tuple[float, Dict]:
    """Tail-Conditional KS Distance"""
    conditions = df_real[condition_col].unique()
    ks_values = []
    weights = []
    details = {}
    
    for z in conditions:
        Y_real_tail = df_real[
            (df_real[condition_col] == z) & (df_real[loss_col] > tau_q)
        ][loss_col].values
        
        Y_synth_tail = df_synth[
            (df_synth[condition_col] == z) & (df_synth[loss_col] > tau_q)
        ][loss_col].values
        
        if len(Y_real_tail) < min_tail_n or len(Y_synth_tail) < min_tail_n:
            details[z] = {
                "ks": np.nan,
                "n_real_tail": len(Y_real_tail),
                "n_synth_tail": len(Y_synth_tail),
                "skipped": True
            }
            continue
        
        ks_stat, _ = stats.ks_2samp(Y_real_tail, Y_synth_tail)
        ks_values.append(ks_stat)
        weights.append(len(Y_real_tail))
        
        details[z] = {
            "ks": ks_stat,
            "n_real_tail": len(Y_real_tail),
            "n_synth_tail": len(Y_synth_tail),
            "skipped": False
        }
    
    if not ks_values:
        return np.nan, details
    
    tcks = np.average(ks_values, weights=weights)
    return tcks, details


# =============================================================================
# 4. Synthetic Data Generation
# =============================================================================

def create_synthesizer(method: str, metadata, config: SyntheticExperimentConfig):
    """합성 데이터 생성 모델 생성"""
    from sdv.single_table import (
        GaussianCopulaSynthesizer,
        CTGANSynthesizer,
        TVAESynthesizer
    )
    
    if method == "GaussianCopula":
        return GaussianCopulaSynthesizer(metadata)
    elif method == "CTGAN":
        return CTGANSynthesizer(
            metadata,
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=False
        )
    elif method == "TVAE":
        return TVAESynthesizer(
            metadata,
            epochs=config.epochs,
            batch_size=config.batch_size
        )
    else:
        raise ValueError(f"Unknown synthesizer: {method}")


def generate_synthetic_data(real_data: pd.DataFrame,
                            method: str,
                            config: SyntheticExperimentConfig,
                            seed: int) -> pd.DataFrame:
    """합성 데이터 생성"""
    from sdv.metadata import SingleTableMetadata
    
    # Metadata 생성
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)
    
    # Synthesizer 생성 및 학습
    synthesizer = create_synthesizer(method, metadata, config)
    
    # 시드 설정
    np.random.seed(seed)
    
    # 학습
    synthesizer.fit(real_data)
    
    # 합성 데이터 생성
    synthetic_data = synthesizer.sample(num_rows=config.n_synthetic_samples)
    
    return synthetic_data


# =============================================================================
# 5. Experiment Runner
# =============================================================================

def run_single_synthetic_experiment(real_df: pd.DataFrame,
                                     synth_df: pd.DataFrame,
                                     config: SyntheticExperimentConfig,
                                     method: str,
                                     q: float,
                                     seed: int) -> Dict:
    """단일 합성 데이터 평가 실험"""
    loss_col = config.loss_col
    condition_col = config.condition_col
    
    # Tail threshold (실제 데이터 기준)
    tau_q = real_df[loss_col].quantile(q)
    
    # 메트릭 계산
    mks = compute_mks(real_df[loss_col].values, synth_df[loss_col].values)
    cks, cks_details = compute_cks(real_df, synth_df, loss_col, condition_col)
    tcks, tcks_details = compute_tcks(real_df, synth_df, tau_q, loss_col, condition_col, config.min_tail_n)
    
    # 추가 통계
    real_tail_mean = real_df[real_df[loss_col] > tau_q][loss_col].mean()
    synth_tail_mean = synth_df[synth_df[loss_col] > tau_q][loss_col].mean() if len(synth_df[synth_df[loss_col] > tau_q]) > 0 else np.nan
    
    real_tail_count = len(real_df[real_df[loss_col] > tau_q])
    synth_tail_count = len(synth_df[synth_df[loss_col] > tau_q])
    
    return {
        "method": method,
        "q": q,
        "seed": seed,
        "tau_q": tau_q,
        "mKS": mks,
        "cKS": cks,
        "T-cKS": tcks,
        "real_tail_mean": real_tail_mean,
        "synth_tail_mean": synth_tail_mean,
        "real_tail_count": real_tail_count,
        "synth_tail_count": synth_tail_count,
        "tail_mean_ratio": synth_tail_mean / real_tail_mean if real_tail_mean > 0 and not np.isnan(synth_tail_mean) else np.nan,
        "tail_count_ratio": synth_tail_count / real_tail_count if real_tail_count > 0 else np.nan,
        "cks_details": cks_details,
        "tcks_details": tcks_details
    }


def run_synthetic_experiments(config: SyntheticExperimentConfig) -> pd.DataFrame:
    """전체 합성 데이터 실험 실행"""
    
    print("=" * 70)
    print("  T-cKS Synthetic Data Experiment")
    print("  실제 합성 데이터 생성 모델을 사용한 평가")
    print("=" * 70)
    
    # 1. 데이터 로드
    print("\n[1] Loading data...")
    full_df = load_arff_data(config.data_path)
    real_df = prepare_data_for_synthesis(full_df, config.synthesis_cols)
    print(f"    Loaded {len(real_df)} samples with columns: {list(real_df.columns)}")
    
    # 손실 분포 통계
    loss = real_df[config.loss_col]
    print(f"\n    Loss distribution:")
    print(f"      Mean: {loss.mean():.2f}")
    print(f"      Median: {loss.median():.2f}")
    print(f"      Std: {loss.std():.2f}")
    print(f"      Skewness: {stats.skew(loss):.2f}")
    print(f"      95th percentile: {loss.quantile(0.95):.2f}")
    
    results = []
    total_runs = len(config.synthesizers) * len(config.q_values) * len(config.random_seeds)
    run_id = 0
    
    print(f"\n[2] Running experiments ({total_runs} total runs)...")
    print("-" * 70)
    
    for method in config.synthesizers:
        print(f"\n  📊 Synthesizer: {method}")
        
        for seed in config.random_seeds:
            print(f"    Seed {seed}: ", end="", flush=True)
            
            try:
                # 합성 데이터 생성
                start_time = time.time()
                synth_df = generate_synthetic_data(real_df, method, config, seed)
                gen_time = time.time() - start_time
                print(f"Generated {len(synth_df)} samples in {gen_time:.1f}s", end="")
                
                # 각 q에 대해 평가
                for q in config.q_values:
                    run_id += 1
                    
                    result = run_single_synthetic_experiment(
                        real_df, synth_df, config, method, q, seed
                    )
                    result["run_id"] = run_id
                    result["generation_time"] = gen_time
                    
                    # 상세 정보 제외
                    result_clean = {k: v for k, v in result.items() 
                                   if k not in ["cks_details", "tcks_details"]}
                    results.append(result_clean)
                
                print(" ✓")
                
            except Exception as e:
                print(f" ✗ Error: {e}")
                continue
    
    print("-" * 70)
    
    results_df = pd.DataFrame(results)
    return results_df, real_df


def main():
    """메인 실행"""
    config = SyntheticExperimentConfig()
    
    # 실험 실행
    results_df, real_df = run_synthetic_experiments(config)
    
    # 결과 저장
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    results_path = output_dir / "synthetic_experiment_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n[3] Results saved to: {results_path}")
    
    # 결과 요약
    print("\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY")
    print("=" * 70)
    
    summary = results_df.groupby(['method', 'q']).agg({
        'mKS': ['mean', 'std'],
        'cKS': ['mean', 'std'],
        'T-cKS': ['mean', 'std'],
        'tail_mean_ratio': 'mean',
        'tail_count_ratio': 'mean'
    }).round(4)
    
    print("\n📊 Results by Synthesizer and Quantile:")
    print(summary.to_string())
    
    # 핵심 발견
    print("\n\n📈 Key Findings (q=0.95):")
    q95_results = results_df[results_df['q'] == 0.95].groupby('method')[['mKS', 'cKS', 'T-cKS']].mean()
    print(q95_results.to_string())
    
    # T-cKS가 cKS 대비 얼마나 더 민감한지
    print("\n\n🔍 Detection Sensitivity (T-cKS / cKS ratio at q=0.95):")
    for method in config.synthesizers:
        method_data = results_df[(results_df['method'] == method) & (results_df['q'] == 0.95)]
        if len(method_data) > 0:
            cks_mean = method_data['cKS'].mean()
            tcks_mean = method_data['T-cKS'].mean()
            if cks_mean > 0:
                ratio = tcks_mean / cks_mean
                print(f"  {method}: {ratio:.2f}x")
    
    print("\n" + "=" * 70)
    print("  Experiment completed!")
    print("=" * 70)
    
    return results_df, config


if __name__ == "__main__":
    results, config = main()

