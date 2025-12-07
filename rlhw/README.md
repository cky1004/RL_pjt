# Dynamic Sigma Band Control (DDQN) — Shortage 불량 검출 (20분 단위)

현업 요청사항(요약):  
- **3차 회귀(트렌드) 기반 예측값(yhat)** 주변에서, **σ(표준편차)** 를 기준으로 상/하한 밴드를 만들고  
- 밴드 안이면 **정상**, 밴드 밖이면 **불량(알람)** 으로 처리합니다.  
- 기존은 **고정 k(예: k=6)** 로 운영(고정 밴드)했는데, 그룹(20분)마다 변동성이 달라서 **동적으로 k를 선택**하면  
  - 불량을 더 잘 잡으면서(**posF1/TPR↑**)  
  - 알람 부담은 과도하게 늘리지 않는(**alarm_rate↓**)  
  trade-off를 만들 수 있습니다.

이 프로젝트는 위를 **강화학습(DDQN)** 관점으로 정식화합니다.

---

## 1) 파일 구성

- `env_dynamic_band.py`  
  - 20분 그룹 단위 환경(gymnasium)  
  - **Action = 대칭 밴드 k 선택**  
  - **Reward = TPR - α·FPR - β·alarm_rate - switch_cost**  
  - (옵션) 알람 스파이크 억제: `alarm_cap`, `spike_coef`

- `train_ddqn_v4.py`  
  - DDQN 학습 + 평가 + baseline 비교 + 그래프/로그 저장
  - baseline:
    - **fixed-k(op)**: `--fixed_k` (기본 6.0)
    - **best-fixed**: 같은 k 후보 중 (posF1 - λ·alarm_rate) 최대를 sweep

---

## 2) 설치

```bash
pip install -r requirements.txt
```


## 3) 데이터 포맷

CSV 컬럼(대소문자 무관):

- `EQUIPMENTID` : 설비 ID
- `GROUPNUM`    : 20분 그룹 번호
- `VALUE`       : 시계열 값
- `PRED`        : 라벨(불량=1, 정상=0)
- (옵션) `ROWNUMS` : 그룹 내 시간 인덱스

---

## 4) 실행 (학습 + 결과 파일 생성)

```bash
python train_ddqn_v4.py   --train_csv data_skab/train.csv   --test_csv data_skab/test.csv   --meta_json split_meta.json  --episodes 1500   --eval_every 20   --device cpu   --sigma_mode value --k_min 2 --k_max 7 --k_step 0.5   --fixed_k 3  --baseline_metric utility
   --lam_alarm 0.2   --fp_pen 0.10 --fn_pen 5.00 --alarm_pen 0.00 --switch_cost 0.02   --spike_coef 200 --cap_source best_fixed   --gamma 0.95   --lr 5e-4   --buffer 100000   --batch 256   --warmup_steps 3000   --target_update 200   --eps_start 1.0 --eps_end 0.01 --eps_decay_steps 20000
```

## 5) 결과물

실행 폴더에 아래가 생성됩니다:

- `ddqn_model.pt` : 마지막 모델
- `ddqn_best.pt`  : **utility(Uλ = posF1 - λ·alarm_rate)** 기준 best 체크포인트
- `eval_log.csv` : 에피소드별 평가 로그
- `learning_curve_v3.png` : return + posF1/alarm + baseline 라인
- `utility_curve_v3.png` : utility 비교 
- `results_summary.json` : baseline/RL 핵심 수치 요약

---
