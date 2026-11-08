# Step 1: 물리 동역학 함수공간의 neural operator — 설계

목표(프로젝트 주제): flow matching(pg-flow, W2·Sobolev 구조)이 만든 앙상블 시나리오를, 물리 제약과 동역학을 이해하는
neural operator가 **함수공간에서** 미래로 사상해 예측 구조를 만든다. Step 1은 그 operator, 즉 **dynamics field**
`G_θ : x ↦ ∂x/∂t`를 설계·구현하고, ERA5를 pg-flow의 입출력 형식으로 통과시켜 WeatherBench2로 climate-ODE 기준선과
비교하는 것이다.

## 1. 가져온 것 (새로 짜지 않은 것)

| 부분 | 어디서 | 어떻게 쓰는가 |
| --- | --- | --- |
| ERA5 자료 계약 | pg-flow `pgcfm/contract.py` | 67채널 KAI 순서·단위·정규화, 66 동적 + 1 정적(orog) 분리, 위도 등록 |
| ERA5 변환 | pg-flow `scripts/wb2_to_kai.py` | WeatherBench2 공개 zarr(64×32, 6 h) → KAI 연도 파일 (`scripts/fetch_era5.sh`) |
| 로더·학습 루프 | pg-flow `pgcfm/runs.py` | 메모리맵 정규화 연도, 경향 std, 재시작 가능한 `fit`, 검증·조기종료 |
| 원시방정식 잔차 | pg-flow `pgcfm/physics/{primitive,grid}.py` | 운동량·열역학·연속·정역학 강형식 잔차(SI), 극 필터, 지하 mask |
| 잔차 ↔ 경향 결합 | pg-flow `pgcfm/dyncore.py` `StrongPhysics`, `estimate_strong_scales` | 망의 경향 f를 ∂_s X에 넣는 잔차, ERA5 수준 b_j, hinge |
| climate-ODE 기준선 | pg-flow `pgcfm/core.py` `Net` = ClimODE `Climate_ResNet_2D` (step 12/16) | 같은 solver·손실·자료에서 backbone만 ResNet |
| 양수성·질량 | pg-flow `pgcfm/balance.py` `PositiveMass` (step 14) | step마다 q, tcwv ≥ 0, 층별 전역 질량 보존 |
| 구면 neural operator | neuraloperator `SFNO` (`SphericalConv`, torch-harmonics SHT) | 접벡터장 G_θ의 backbone |
| 평면 FNO (대조) | neuraloperator `FNO` | `--backbone fno` |
| ODE 적분 | torchdiffeq `odeint` (고정 격자 `midpoint` / `rk4`) | 6 h step 안의 적분 |
| 평가 | pg-flow `pgcfm/eval/{wb2,scores}.py` (WeatherBench2 metrics 벤더) | RMSE·ACC·CRPS·spread-skill, headline 변수 |
| pg-flow 앙상블 형식 | pg-flow `scripts/infer_pgcfm.py` / `pgcfm/eval/wb2.py` | `forecast.nc (initial_condition, lead_time, member, channel, lat, lon)` |

pg-flow와 neuraloperator는 `third_party/`의 서브모듈로 커밋에 고정했다(pg-flow는 `main`이 비어 있어 `claude/blend-sr-s1`
브랜치의 fc37fdc).

## 2. 설계

### 2.1 함수공간과 operator

상태 x(t)는 구면 S² 위의 66채널 함수(정규화된 KAI 동적 채널). dynamics field는

    dx/dt = G_θ(x, static, time),      G_θ = SFNO (lift → 4 × [구면조화 spectral conv + channel MLP] → projection, 0 초기화)

구면조화 기저 Y_lm은 Laplace–Beltrami의 고유함수라서 (i) spectral conv는 격자 해상도와 무관한 함수공간 사상이고,
(ii) 모든 Sobolev 노름이 이 기저에서 대각이다. 그래서 손실과 operator가 같은 기저를 쓴다. 출력 head는 0으로 초기화해
field가 persistence에서 출발한다(pg-flow step 12).

입력은 x(66) + 위치·지형 특징(7: cos/sin lat, cos/sin lon, sin lat·cos/sin lon, 표준화 지형) + 시간 특징(4: 하루·계절
위상). 출력은 채널별 6 h 경향 std(t6) 단위이고 tisr(알려진 강제력)은 예측하지 않고 자료에서 넣는다.

### 2.2 적분과 rollout

    x_{k+1} = odeint(G_θ, x_k, s ∈ [0, 1]; midpoint, step 1)   (s는 6 h 단위; --dt-hours로 세분)
    tisr_{k+1} := 자료,  x_{k+1} := PositiveMass(x_{k+1})

앙상블 멤버는 배치의 행이다(`rollout_members`): 같은 operator가 모든 시나리오에 적용된다.

### 2.3 손실

    L = L2 + λ_sob · Sobolev + λ_p · Physics

* **L2** (pg-flow step 16): 위도 가중, 6 h 경향 분산 단위, 커리큘럼 K = 1 → 2 → 4 → 8.
* **Sobolev** (`ffc/sobolev.py`): 오차 e_k = x_k^θ − x_k의 `‖e‖²_{H^s} = Σ_{l,m} (1 + l(l+1))^s |ê_lm|²`, torch-harmonics
  `RealSHT`(orthonormal)로 계산하고 4π로 나눠 s = 0이면 면적 평균 e²와 같게, 가중치는 백색 스펙트럼의 평균이 1이 되게
  정규화해 λ_sob = 1이 L2와 같은 자릿수가 되게 했다. s = 1은 경사(전선·제트)를, **s = −1은 W2 기하의 접공간 노름**(Otto
  calculus: 밀도 공간의 W2 metric은 접벡터의 가중 H^{−1} 노름)을 본다. 두 arm을 모두 돌린다.
* **Physics (PINN)** (`ffc/physics.py`): 무작위 rollout 상태(IC 포함) 2개에서 망의 경향 f = G_θ(x)를 원시방정식의
  ∂_s X에 넣은 강형식 잔차 R_j(운동량 u,v · 열역학 T · 연속 · 정역학)를 SI로 계산, b_j = ERA5에 12 h 중심차분 경향을 넣은
  같은 잔차의 RMS. 손실은 `Σ_j γ_j max(m_j − 1, 0)`, m_j = 가중평균 (R_j/b_j)². **목표를 0이 아니라 ERA5 수준(1)에 둔 이유**는
  pg-flow step 3/16의 결과다: 이 격자(5.625°, 12층)에서 이산 연산자의 절단오차가 경향만큼 커서 잔차 0은 정지 대기를
  뜻한다(KE 비 0.63까지 감쇠). 수집점은 같은 lead의 관측 상태(지표기압)로 고정해 field가 sp를 바꿔 점을 없애지 못하게 했다.

### 2.4 앙상블 I/O와 평가

* 입력: pg-flow `forecast.nc`의 member 차원(`read_members`, lead 선택) → `(N, M, 66, H, W)`. 이 환경에는 pg-flow 체크포인트가
  없어 `perturbed_members`(x0 + α × 무작위 6 h 분석 증분, tisr 제외)가 **flow-matching 시나리오의 대역**이다. 표에는 그렇게
  표기한다.
* 출력: 같은 `forecast.nc` 형식(lead 0 = IC, 물리 단위, member 차원). pg-flow의 `scripts/evaluate.py`도 그대로 읽는다.
* 평가: `ffc/evaluate.py`가 KAI 배열을 WB2 xarray로 바꾸고(6 h lead, 극 없는 위도) pg-flow의 `score_forecast`(WB2 metrics)로
  headline(z500, t850, q700, u850, v850, t2m, msl)의 RMSE·bias·ACC(학습연도 dayofyear/hour 기후값)·CRPS·spread/skill을 낸다.
  persistence는 평가 스크립트가 항상 덧붙인다.

### 2.5 arm (configs/step1_64x32.yaml)

| arm | backbone | 손실 | 뜻 |
| --- | --- | --- | --- |
| climate_ode | ClimODE ResNet (w48, 블록 2+2) | L2 | climate-ODE 기준선: pg-flow step 16 DynCore와 같은 구조 |
| sfno_l2 | SFNO (w64, 4층, lmax 16) | L2 | 함수공간 operator만 바꾼 것 |
| sfno_h1_pinn | SFNO | L2 + H¹(λ 1) + PINN(λ_p 0.05, 목표 1) | 제안 설계 |
| sfno_w2 | SFNO | L2 + H^{−1}(λ 1) | W2 접노름이 예측·안정성에 주는 영향 |

공통: 학습 2010–2015, 검증 2016(8개 고정 창, K = 8), 시험 2017(00 UTC, 5일 간격 73 IC, 6–144 h), 3000 반복, batch 4,
AdamW 3e-4 cosine, CPU 1 thread/arm(4 arm 동시).

## 3. 환경 제약 (명시)

이 클라우드 스레드는 CPU 4코어·RAM 15 GB·디스크 30 GB, GPU 없음. 전체 ERA5(0.25°)나 KAI 1.5°(240×121, 연 2.8 GB)는
불가능해 WeatherBench2 공개 ERA5 **64×32 (5.625°, 6 h)** 를 썼다(ClimODE·pg-flow step 11–16과 같은 격자). 결과는 구조
비교용이고, 1.5° KAI 자료에서는 `--kai <KAI dir> --device cuda --amp --width 128 --modes-lat 48`로 같은 코드가 돈다
(operator는 해상도 무관; `SphericalSobolev`는 lmax만 바뀐다).

알려진 근사: WB2 64×32는 셀 중심 격자(극 없음)인데 torch-harmonics의 `equiangular` 구적은 극을 포함한 노드를 가정한다.
`SphericalSobolev.parseval_error`로 측정한 차이는 매끄러운 장에서 1–2 %, 증분 장에서 수 %다. SFNO의 spectral conv도 같은
구적을 쓰므로 operator와 손실이 같은 근사를 공유한다.
