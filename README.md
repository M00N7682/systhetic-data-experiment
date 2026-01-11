# T-cKS: Tail-Conditional KS Distance

> **Tail-Conditional KS Distance for Evaluating Conditional Extreme Loss Preservation in Insurance Synthetic Data**

보험 합성 데이터의 조건부 극단 손실 보존을 평가하기 위한 새로운 통계적 지표 **T-cKS (Tail-Conditional KS Distance)**를 제안합니다.

## 📌 연구 요약

기존 합성 데이터 품질 평가 지표(mKS, cKS)는 전체 분포의 평균적 유사성에 초점을 맞추고 있어, **보험 리스크 관점에서 핵심적인 조건부 극단 손실(tail) 분포의 왜곡**을 충분히 탐지하지 못합니다.

### 주요 기여
1. **T-cKS 지표 제안**: 조건부 분포의 tail 영역에 특화된 평가 지표
2. **이론적 기반**: T-cKS의 통계적 일관성(consistency) 증명
3. **실험적 검증**: 
   - 통제된 왜곡 실험에서 **20배 높은 민감도**
   - 실제 합성 모델(TVAE)에서 **3.5배 높은 tail 왜곡 탐지**

## 📊 지표 비교

| 특성 | mKS | cKS | **T-cKS** |
|------|-----|-----|-----------|
| 평가 범위 | 전체 분포 | 조건별 분포 | 조건별 Tail |
| 조건 변수 고려 | ✗ | ✓ | ✓ |
| Tail 영역 집중 | ✗ | ✗ | **✓** |
| 80% Tail 축소 탐지 | 0.017 | 0.017 | **0.344** |

## 🗂 프로젝트 구조

```
.
├── paper_final.tex              # 최종 논문 (LaTeX)
├── paper_overleaf_v2.zip        # Overleaf 업로드용
├── requirements.txt             # Python 의존성
├── dataset.arff                 # Allstate Claims Severity (OpenML)
│
├── experiment_synthetic.py      # 실제 합성 모델 실험 (CTGAN, TVAE, GaussianCopula)
├── generate_missing_figures.py  # 통제된 왜곡 실험 그래프 생성
├── visualize_synthetic_results.py # 합성 모델 실험 시각화
│
└── results/
    ├── figures/                 # 실험 결과 그래프
    └── *.csv                    # 실험 결과 데이터
```

## 🚀 실행 방법

### 환경 설정
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 실험 실행
```bash
# 실제 합성 데이터 생성 모델 실험 (CTGAN, TVAE, GaussianCopula)
python experiment_synthetic.py

# 그래프 생성
python visualize_synthetic_results.py
python generate_missing_figures.py
```

## 📈 실험 결과

### 1. 통제된 왜곡 실험
- Tail Scaling (80% 축소) 시 T-cKS가 기존 지표 대비 **~20배** 민감도

### 2. 실제 합성 모델 평가

| 모델 | mKS | cKS | T-cKS | Tail 재현율 |
|------|-----|-----|-------|-------------|
| TVAE | 0.014 | 0.021 | **0.075** | 25.4% |
| CTGAN | 0.086 | 0.072 | 0.125 | 43.2% |
| GaussianCopula | 0.068 | 0.141 | 0.142 | 17.7% |

**핵심 발견**: TVAE는 기존 지표(cKS=0.021)로는 우수해 보이지만, T-cKS(0.075)가 **3.5배 높은 tail 왜곡**을 탐지

## 📚 데이터셋

- **Allstate Claims Severity Dataset** (OpenML)
- URL: https://www.openml.org/search?type=data&id=42571
- 188,318 샘플, heavy-tailed 손실 분포

## 📄 인용

```bibtex
@article{moon2026tcks,
  title={Tail-Conditional KS Distance: A Consistent Statistical Metric for Evaluating Conditional Extreme Loss Preservation in Insurance Synthetic Data},
  author={Moon, Deok Lyong},
  year={2026}
}
```

## 📧 연락처

- **문덕룡 (Deok Lyong Moon)**
- 경희대학교 경영학과
- Email: dfjk71@khu.ac.kr

## 📜 라이선스

MIT License

