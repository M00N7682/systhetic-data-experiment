#!/usr/bin/env python3
"""
T-cKS 실험 실행 스크립트
=========================
전체 실험 파이프라인을 실행하고 결과를 시각화합니다.

사용법:
    python run_experiment.py
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent))

from experiment_pipeline import main as run_pipeline
from visualization import generate_all_figures


def run_all():
    """전체 실험 및 시각화 실행"""
    
    print("=" * 70)
    print("  T-cKS (Tail-Conditional KS Distance) Experiment")
    print("  보험 합성 데이터 평가 지표 검증 실험")
    print("=" * 70)
    
    # Step 1: 실험 파이프라인 실행
    print("\n[STEP 1] Running experiment pipeline...")
    results_df, config = run_pipeline()
    
    # Step 2: 시각화 생성
    print("\n[STEP 2] Generating visualizations...")
    try:
        generate_all_figures()
    except Exception as e:
        print(f"Warning: Visualization failed - {e}")
        print("You can run visualization separately after installing matplotlib/seaborn")
    
    # Step 3: 결과 요약
    print("\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY")
    print("=" * 70)
    
    # 핵심 결과 출력
    print("\n📊 Key Findings (q=0.95, strength=0.8):")
    
    key_results = results_df[
        (results_df['q'] == 0.95) & 
        (results_df['strength'] == 0.8)
    ].groupby('method')[['mKS', 'cKS', 'T-cKS']].mean()
    
    print(key_results.to_string())
    
    # 비교 분석
    print("\n📈 Detection Improvement (T-cKS vs cKS):")
    for method in ['winsorization', 'thinning']:
        baseline = results_df[
            (results_df['method'] == method) & 
            (results_df['strength'] == 0.0) & 
            (results_df['q'] == 0.95)
        ][['cKS', 'T-cKS']].mean()
        
        distorted = results_df[
            (results_df['method'] == method) & 
            (results_df['strength'] == 0.8) & 
            (results_df['q'] == 0.95)
        ][['cKS', 'T-cKS']].mean()
        
        cks_change = distorted['cKS'] - baseline['cKS']
        tcks_change = distorted['T-cKS'] - baseline['T-cKS']
        
        print(f"\n  {method.capitalize()}:")
        print(f"    cKS change:   {baseline['cKS']:.4f} → {distorted['cKS']:.4f} (Δ = {cks_change:.4f})")
        print(f"    T-cKS change: {baseline['T-cKS']:.4f} → {distorted['T-cKS']:.4f} (Δ = {tcks_change:.4f})")
        
        if tcks_change > cks_change:
            improvement = (tcks_change / cks_change - 1) * 100 if cks_change > 0 else float('inf')
            print(f"    → T-cKS detected {improvement:.1f}% more distortion than cKS")
    
    print("\n" + "=" * 70)
    print("  Experiment completed successfully!")
    print("  Results saved to: results/")
    print("=" * 70)


if __name__ == "__main__":
    run_all()

