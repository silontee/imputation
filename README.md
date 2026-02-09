# Impute platform - Missing Value Imputation Platform

> 결측치(Missing Value) 자동 보정을 위한 풀스택 웹 애플리케이션

[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=next.js)](https://nextjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch)](https://pytorch.org/)

---

## 목차

- [개요](#개요)
- [시스템 아키텍처](#시스템-아키텍처)
- [데이터 처리 흐름](#데이터-처리-흐름)
- [Imputation 알고리즘](#imputation-알고리즘)
- [데이터 증강 (SMOTENC)](#데이터-증강-smotenc)
- [컬럼 역할 시스템](#컬럼-역할-시스템)
- [GPU 지원](#gpu-지원)
- [API 엔드포인트](#api-엔드포인트)
- [프로젝트 구조](#프로젝트-구조)
- [빠른 시작](#빠른-시작)
- [DB 마이그레이션](#db-마이그레이션)
- [환경 변수](#환경-변수)

---

## 개요

**ImputeX**는 CSV 파일의 결측치를 자동으로 탐지하고, 통계 기법부터 딥러닝(NAOMI)까지 다양한 알고리즘으로 보정하는 웹 기반 플랫폼입니다.

### 주요 기능

- **자동 열 타입 감지** — Numeric, Categorical, DateTime, ID 자동 분류
- **결측치 분석** — 열별 결측치 비율, 패턴, 경고 표시
- **6가지 알고리즘** — MICE, KNN, Mean/Mode, Regression, NAOMI, TOTEM
- **컬럼 역할 지정** — TARGET(보정 대상) / FEATURE(입력 전용) / IGNORE(제외)
- **모델별 하이퍼파라미터** — KNN의 K값, MICE의 estimator, NAOMI의 epochs 등 세부 조정
- **실시간 진행률** — 1초 단위 폴링으로 학습/보정 과정 실시간 표시
- **GPU 가속** — CUDA, MPS(Apple Silicon) 자동 감지
- **비동기 처리** — Celery 기반 백그라운드 작업
- **결과 다운로드** — 보정된 CSV 파일 즉시 다운로드

---

## 시스템 아키텍처

### 전체 서비스 구성

```mermaid
graph TB
    subgraph "Client"
        Browser["Browser"]
    end

    subgraph "Docker Network"
        subgraph "Entry Point"
            Nginx["Nginx<br/>Reverse Proxy<br/>:80"]
        end

        subgraph "Frontend"
            NextJS["Next.js 16<br/>React UI<br/>:3000"]
        end

        subgraph "Backend"
            FastAPI["FastAPI<br/>REST API<br/>:8000"]
            Celery["Celery Worker<br/>Imputation Engine"]
        end

        subgraph "Data Layer"
            PostgreSQL["PostgreSQL 16<br/>Job Storage<br/>:5432"]
            Redis["Redis 7<br/>Task Queue<br/>:6379"]
        end

        subgraph "Storage"
            Uploads["/uploads"]
            Results["/results"]
        end
    end

    Browser --> Nginx
    Nginx -->|"/"| NextJS
    Nginx -->|"/api/*"| FastAPI
    NextJS -.->|"API Calls"| Nginx
    FastAPI --> PostgreSQL
    FastAPI --> Redis
    Celery --> Redis
    Celery --> PostgreSQL
    FastAPI --> Uploads
    Celery --> Uploads
    Celery --> Results
```

### 서비스별 역할

| 서비스 | 기술 스택 | 역할 |
|--------|-----------|------|
| **nginx** | Nginx 1.25 | 리버스 프록시, `/` → Frontend, `/api/*` → Backend |
| **frontend** | Next.js 16, pnpm, shadcn/ui | 파일 업로드, 스키마 리뷰, 진행률 표시, 결과 다운로드 |
| **api** | FastAPI + Uvicorn | REST API, 파일 분석, 작업 관리, Health Check |
| **worker** | Celery + PyTorch | 백그라운드 Imputation 처리, GPU 가속 지원 |
| **postgres** | PostgreSQL 16 | 작업 상태, 메타데이터, 로그 저장 |
| **redis** | Redis 7 | Celery 메시지 브로커 |

---

## 데이터 처리 흐름

사용자가 파일을 업로드하면 아래 3단계를 거쳐 보정된 CSV를 받습니다.

### Step 1. 파일 업로드 및 분석

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend
    participant API as FastAPI
    participant AN as Analyzer
    participant DB as PostgreSQL

    U->>FE: CSV 파일 드래그 & 드롭
    FE->>API: POST /api/v1/analyze (multipart)
    API->>API: 파일 크기 검증 (500MB 이하)
    API->>DB: Job 레코드 생성 (UPLOADED)
    API->>API: 파일 저장 → /uploads/{job_id}_input.csv
    API->>AN: analyze_dataframe(df)
    AN->>AN: 열 타입 추론 (Numeric/Categorical/DateTime/ID)
    AN->>AN: 결측치 비율 계산, 경고 생성
    AN-->>API: List[ColumnProfile]
    API-->>FE: AnalyzeResponse (job_id, columns, null_ratio)
    FE-->>U: Schema Review 화면 표시
```

### Step 2. 스키마 리뷰 및 설정

사용자가 아래 항목을 설정한 뒤 "Start Imputation" 버튼을 클릭합니다.

| 설정 항목 | 설명 |
|-----------|------|
| **컬럼 역할** | 각 컬럼을 TARGET / FEATURE / IGNORE로 지정 |
| **컬럼 타입** | 자동 감지된 타입을 수동 변경 가능 |
| **모델 선택** | MICE, KNN, Mean/Mode, Regression, NAOMI, TOTEM 중 선택 |
| **하이퍼파라미터** | 모델별 세부 파라미터 조정 (선택 사항) |

### Step 3. Imputation 실행 및 결과

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend
    participant API as FastAPI
    participant RD as Redis
    participant WK as Celery Worker
    participant DB as PostgreSQL

    U->>FE: 설정 완료 후 "Start Imputation" 클릭
    FE->>API: POST /api/v1/jobs/{job_id}/start
    API->>API: 하이퍼파라미터 검증 (Pydantic)
    API->>DB: Job 상태 → QUEUED
    API->>RD: Celery Task 등록
    API-->>FE: 202 Accepted

    loop 1초 폴링
        FE->>API: GET /api/v1/jobs/{job_id}
        API->>DB: 상태 및 진행률 조회
        API-->>FE: JobStatusResponse (progress, stage, logs)
        FE->>FE: 진행률 바 및 로그 업데이트
    end

    RD-->>WK: Task 수신
    WK->>DB: 상태 → PROCESSING (5%)
    WK->>WK: CSV 로드, 전처리 (15%)
    WK->>WK: 인코딩 (25%)
    WK->>WK: Imputation 실행 (35%→70%)
    Note over WK: NAOMI인 경우 epoch별 진행률 보고
    WK->>WK: Preview 생성 (80%)
    WK->>WK: 결과 CSV 저장 → /results/{job_id}_output.csv
    WK->>DB: 상태 → COMPLETED (100%)

    FE-->>U: 완료 표시 + 다운로드 버튼
    U->>FE: 다운로드 클릭
    FE->>API: GET /api/v1/jobs/{job_id}/download
    API-->>U: 보정된 CSV 파일
```

### Step 4. (선택) Time Series 증강(SMOTENC)

Imputation 완료 후, **Time Series 증강**을 선택적으로 수행할 수 있습니다.  
증강은 **슬라이딩 윈도우 기반 벡터화 → SMOTENC 적용 → 합성 윈도우 생성** 흐름으로 진행되며,  
웹 UI에서는 **원본 + 증강 결과를 오버레이 차트로 동적 시각화**합니다.

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend
    participant API as FastAPI
    participant DB as PostgreSQL
    participant WK as Celery Worker

    U->>FE: 증강 설정 입력 (label, categorical 등)
    FE->>API: POST /api/v1/jobs/{job_id}/augment
    API->>DB: 증강 상태/파라미터 기록
    API-->>FE: 202 Accepted
    WK->>DB: 증강 진행률 업데이트
    WK->>WK: 슬라이딩 윈도우 → SMOTENC → 합성 윈도우
    WK->>DB: 증강 결과 저장 (CSV/preview)
    FE->>API: GET /api/v1/jobs/{job_id}/augment/status
    API-->>FE: 진행률/preview
    FE-->>U: 오버레이 차트 표시 + 다운로드 버튼
```

---

## Imputation 알고리즘

### 지원 알고리즘 비교

| 알고리즘 | 방식 | 설명 | 적합한 데이터 | 속도 |
|----------|------|------|---------------|------|
| **MICE** | 다변량 통계 | 반복적 연쇄 방정식으로 결측치 추정 | 변수 간 상관관계가 있는 데이터 | 보통 |
| **KNN** | 다변량 통계 | K개 유사 데이터포인트 기반 추정 | 패턴이 유사한 데이터 | 보통 |
| **Mean/Mode** | 단변량 통계 | 평균(수치형) / 최빈값(범주형) 대체 | 단순 결측, 빠른 처리 | 빠름 |
| **Regression** | 다변량 통계 | 회귀 모델 기반 예측 (MICE 엔진 사용) | 선형 관계가 강한 데이터 | 보통 |
| **NAOMI** | 딥러닝 | 양방향 LSTM + 다중해상도 디코더 | 시계열, 순차 데이터 | 느림 (GPU 권장) |
| **TOTEM** | 딥러닝 | VQVAE 기반 window 추론 보간 | 시계열, 결측 패턴 복원 | 빠름~보통 |

### 모델별 하이퍼파라미터

| 모델 | 파라미터 | 기본값 | 범위 | 설명 |
|------|----------|--------|------|------|
| **MICE / Regression** | `max_iter` | 10 | 1-100 | 반복 횟수 |
| | `random_state` | 42 | 정수 | 재현성을 위한 시드 |
| | `estimator` | bayesian_ridge | bayesian_ridge, random_forest, extra_trees | 내부 추정 모델 |
| **KNN** | `n_neighbors` | 5 | 1-50 | 참조 이웃 수 (K) |
| | `weights` | uniform | uniform, distance | 이웃 가중치 방식 |
| | `metric` | nan_euclidean | nan_euclidean, euclidean | 거리 측정 방식 |
| **NAOMI** | `epochs` | 50 | 1-500 | 학습 반복 횟수 |
| | `hidden_dim` | 64 | 16-512 | LSTM 은닉 차원 수 |
| | `lr` | 0.001 | 1e-6 ~ 1.0 | 학습률 |
| | `num_resolutions` | auto | 1-16 또는 null | 다중해상도 레벨 수 |
| **TOTEM** | `window_size` | 96 | 16-512 (`%4==0`) | 추론 window 길이 |
| | `normalization` | zscore | zscore, minmax | 정규화 방식 |
| | `merge_mode` | non_overlap | non_overlap, overlap | 윈도우 병합 모드 |
| | `stride` | window_size | overlap 시 `1 <= stride < window_size` | 윈도우 이동 간격 |

### NAOMI 알고리즘 구조 (NeurIPS 2019)

NAOMI는 시퀀스 데이터에 특화된 딥러닝 보간 모델입니다.

```mermaid
flowchart TD
    A["입력 시퀀스 X (T x D) + 마스크 M"] --> B["양방향 LSTM 인코더"]
    B --> B1["Forward LSTM → h_f"]
    B --> B2["Backward LSTM → h_b"]

    B1 --> C["다중해상도 디코더 (Algorithm 1)"]
    B2 --> C

    C --> D["Resolution 1 (가장 넓은 간격)"]
    D --> E["Resolution 2"]
    E --> F["Resolution R (가장 좁은 간격)"]

    F --> G["보간된 시퀀스 X_imputed"]

    subgraph "학습 (Self-supervised)"
        H["관측된 값 중 30% 랜덤 마스킹"] --> I["마스킹된 위치에서 MSE Loss 계산"]
        I --> J["역전파 및 가중치 업데이트"]
    end
```

**동작 원리:**
1. **인코더**: 관측된 값과 마스크를 양방향 LSTM으로 인코딩하여 문맥 벡터 생성
2. **디코더**: 결측 구간의 좌우 관측점(pivot) 문맥 벡터를 입력으로, 중간 지점부터 재귀적으로 보간 (coarse → fine)
3. **학습**: 관측된 값 일부를 추가로 마스킹하여 자기지도 학습 (MSE loss)

### 열 타입 자동 추론 로직

```mermaid
flowchart TD
    A["Series 입력"] --> B{"모든 값이 null?"}
    B -->|Yes| C["CATEGORICAL"]
    B -->|No| D{"DateTime 파싱 >= 90%?"}
    D -->|Yes| E["DATETIME"]
    D -->|No| F{"숫자 변환 >= 95%?"}
    F -->|Yes| G{"고유값 <= 20개?"}
    G -->|Yes| H["CATEGORICAL<br/>(경고: Low cardinality)"]
    G -->|No| I["NUMERIC"]
    F -->|No| J{"모든 값이 고유?"}
    J -->|Yes| K["ID"]
    J -->|No| L["CATEGORICAL"]
```

---

## 데이터 증강 (SMOTENC)

Time Series 데이터를 대상으로 **SMOTENC** 기반 증강을 지원합니다.  
특성(feature)에 범주형이 포함될 수 있어 **SMOTENC**를 사용하며, 라벨은 분류형이어야 합니다.

### 핵심 개념
- **슬라이딩 윈도우**: 시계열을 `window_size`와 `stride`로 구간 분할
- **벡터화 + SMOTENC**: 윈도우를 특성 벡터로 변환 후 증강
- **시각화**: 원본(파란선) + 증강(주황선) **오버레이 차트**

### 제약
- 시계열 연속성을 직접 보장하지 않음 → **윈도우 기반 합성**으로 접근
- 라벨 컬럼과 범주형 특성 컬럼은 **명시적으로 지정** 필요

---

## 컬럼 역할 시스템

각 컬럼에 역할을 지정하여 보간 대상과 입력 변수를 분리할 수 있습니다.

| 역할 | 의미 | 결측치 보정 | 모델 입력 |
|------|------|:-----------:|:---------:|
| **TARGET** | 보간 대상 컬럼 | O | O |
| **FEATURE** | 모델 입력 전용 (보간 안 함) | X | O |
| **IGNORE** | 완전히 제외 | X | X |

**예시:** 센서 데이터에서 `temperature`와 `humidity`를 보간하고 싶은데, `timestamp`는 입력으로만 사용하고 `device_id`는 제외하려면:

| 컬럼 | 역할 |
|------|------|
| `temperature` | TARGET |
| `humidity` | TARGET |
| `timestamp` | FEATURE |
| `device_id` | IGNORE |

Quick Actions 버튼으로 일괄 지정도 가능합니다:
- **Missing > 0 → Target** : 결측치가 있는 컬럼을 모두 TARGET으로
- **All → Feature** : 모든 컬럼을 FEATURE로
- **ID columns → Ignore** : ID 타입 컬럼을 모두 IGNORE로

---

## GPU 지원

NAOMI 모델은 PyTorch 기반으로, GPU를 자동 감지하여 학습을 가속합니다.

### 디바이스 감지 우선순위

```
CUDA (NVIDIA GPU) → MPS (Apple Silicon) → CPU
```

### 환경별 사용 가능 디바이스

| 환경 | CUDA | MPS | CPU |
|------|:----:|:---:|:---:|
| Docker (Mac/일반 서버) | - | - | O |
| Docker + NVIDIA GPU 서버 | O | - | fallback |
| 로컬 실행 (Mac Apple Silicon) | - | O | fallback |
| 로컬 실행 (NVIDIA GPU 머신) | O | - | fallback |

> **MPS는 Docker에서 사용 불가합니다.** Docker Desktop은 내부적으로 Linux VM을 실행하며, MPS는 macOS Metal API 전용이므로 Linux VM에서 접근할 수 없습니다.

### NVIDIA GPU Docker 사용법

NVIDIA GPU가 있는 서버에서 Docker로 GPU를 사용하려면:

```bash
# 1. nvidia-container-toolkit 설치 (호스트에서)
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# 2. GPU 오버라이드 파일과 함께 실행
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

---

## API 엔드포인트

### 데이터 분석

| Method | Endpoint | 설명 |
|--------|----------|------|
| `POST` | `/api/v1/analyze` | CSV 파일 업로드 및 스키마 분석 |

```bash
curl -X POST "http://localhost/api/v1/analyze" \
  -F "file=@data.csv"
```

### 작업 관리

| Method | Endpoint | 설명 |
|--------|----------|------|
| `POST` | `/api/v1/jobs/{job_id}/start` | Imputation 작업 시작 |
| `GET` | `/api/v1/jobs/{job_id}` | 작업 상태 조회 (진행률, 로그, 미리보기) |
| `GET` | `/api/v1/jobs/{job_id}/download` | 결과 파일 다운로드 |
| `DELETE` | `/api/v1/jobs/{job_id}` | 작업 취소 |

### 증강 (SMOTENC) - 설계 반영

| Method | Endpoint | 설명 |
|--------|----------|------|
| `POST` | `/api/v1/jobs/{job_id}/augment` | 증강 작업 시작 |
| `GET` | `/api/v1/jobs/{job_id}/augment/status` | 증강 상태 조회 |
| `GET` | `/api/v1/jobs/{job_id}/augment/download` | 증강 결과 다운로드 |

### 작업 시작 요청 예시

```json
{
  "model_type": "NAOMI",
  "hyperparameters": {
    "hidden_dim": 64,
    "epochs": 50,
    "lr": 0.001
  },
  "column_config": [
    { "name": "temperature", "type": "NUMERIC", "role": "TARGET" },
    { "name": "humidity",    "type": "NUMERIC", "role": "TARGET" },
    { "name": "timestamp",   "type": "DATETIME","role": "FEATURE" },
    { "name": "device_id",   "type": "ID",      "role": "IGNORE" }
  ]
}
```

### 작업 상태 값

| Status | 설명 |
|--------|------|
| `UPLOADED` | 파일 업로드 완료 |
| `REVIEWED` | 사용자 설정 완료 |
| `QUEUED` | 작업 대기열에 추가됨 |
| `PROCESSING` | Imputation 진행 중 (진행률 0~100%) |
| `COMPLETED` | 완료, 다운로드 가능 |
| `FAILED` | 오류 발생 |
| `CANCELED` | 사용자 취소 |

---

## 프로젝트 구조

```
imputation/
├── docker-compose.yml              # 프로덕션 설정
├── docker-compose.dev.yml          # 개발 환경 오버라이드
├── docker-compose.gpu.yml          # NVIDIA GPU 오버라이드
│
├── frontend/                       # Next.js 프론트엔드
│   ├── Dockerfile
│   ├── app/
│   │   └── page.tsx                # 메인 페이지 (상태 머신: upload→schema→processing→complete)
│   ├── components/
│   │   ├── file-upload.tsx         # 파일 업로드 UI
│   │   ├── schema-review.tsx       # 스키마 리뷰 (컬럼 역할, 모델 선택, 하이퍼파라미터)
│   │   ├── processing-view.tsx     # 진행률 표시 및 결과
│   │   └── imputation-chart.tsx    # 보간 결과 차트
│   ├── lib/
│   │   └── api.ts                  # API 클라이언트
│   └── types/
│       └── schema.ts               # TypeScript 타입 정의
│
├── backend/                        # FastAPI 백엔드
│   ├── docker/
│   │   ├── Dockerfile.api          # API 서버 이미지
│   │   ├── Dockerfile.worker       # Worker 이미지 (PyTorch 포함)
│   │   └── entrypoint.sh
│   ├── requirements.txt            # Python 의존성 (PyTorch 포함)
│   ├── app/
│   │   ├── main.py                 # FastAPI 앱 진입점
│   │   ├── api/v1/
│   │   │   ├── analyze.py          # POST /analyze
│   │   │   └── jobs.py             # 작업 CRUD API
│   │   ├── services/
│   │   │   ├── analyzer.py         # 열 타입 추론 엔진
│   │   │   └── imputation/
│   │   │       ├── base.py         # BaseImputer 추상 클래스
│   │   │       ├── mice.py         # MICE (IterativeImputer)
│   │   │       ├── knn.py          # KNN Imputer
│   │   │       ├── statistical.py  # Mean/Mode Imputer
│   │   │       ├── naomi/          # NAOMI 딥러닝 모듈
│   │   │           ├── model.py    # 양방향 LSTM 인코더 + 다중해상도 디코더
│   │   │           ├── trainer.py  # 학습 루프 (self-supervised)
│   │   │           └── imputer.py  # BaseImputer 인터페이스
│   │   │       └── totem/          # TOTEM 추론 모듈
│   │   │           ├── imputer.py
│   │   │           ├── model_loader.py
│   │   │           └── inference.py
│   │   ├── db/
│   │   │   ├── models.py           # SQLAlchemy ORM 모델
│   │   │   └── repository.py       # 데이터 접근 계층
│   │   └── schemas/
│   │       ├── analyze.py          # 분석 스키마 (ColumnType, ColumnRole)
│   │       └── jobs.py             # 작업 스키마 (ModelType, 하이퍼파라미터 검증)
│   ├── worker/
│   │   ├── celery_app.py           # Celery 설정
│   │   └── tasks.py                # run_imputation 태스크 (모델 디스패치, 진행률 콜백)
│   └── alembic/                    # DB 마이그레이션
│
└── nginx/
    ├── nginx.conf
    └── conf.d/
        └── imputex.conf            # / → frontend:3000, /api → api:8000
```

### Backend 모듈 구조

```mermaid
graph LR
    subgraph "API Layer"
        A1["analyze.py<br/>파일 분석"]
        A2["jobs.py<br/>작업 관리"]
    end

    subgraph "Service Layer"
        S1["analyzer.py<br/>열 타입 추론"]
        subgraph "imputation/"
            I1["mice.py"]
            I2["knn.py"]
            I3["statistical.py"]
            subgraph "naomi/"
                N1["model.py<br/>LSTM + Decoder"]
                N2["trainer.py<br/>학습 루프"]
                N3["imputer.py<br/>BaseImputer"]
            end
        end
    end

    subgraph "Data Layer"
        D1["repository.py"]
        D2["models.py"]
    end

    subgraph "Worker"
        W1["tasks.py<br/>Celery Task"]
    end

    A1 --> S1
    A2 --> D1
    W1 --> I1
    W1 --> I2
    W1 --> I3
    W1 --> N3
    N3 --> N2
    N2 --> N1
    W1 --> D1
```

---

## 빠른 시작

### 요구사항

- Docker Desktop 4.0+
- Docker Compose v2

### 실행

```bash
# 1. 저장소 클론
git clone https://github.com/your-repo/imputation.git
cd imputation

# 2. 환경 변수 설정
cp backend/.env.example backend/.env

# 3. 전체 서비스 빌드 및 실행
docker compose up -d --build

# 4. 상태 확인
docker compose ps

# 5. 로그 확인
docker compose logs -f
```

### GPU가 있는 서버에서 실행

```bash
# NVIDIA GPU 사용 (nvidia-container-toolkit 필요)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

### 접속

| 서비스 | URL |
|--------|-----|
| **웹 UI** | http://localhost |
| **API 문서 (Swagger)** | http://localhost/api/docs |
| **Health Check** | http://localhost/health |

### 개발 환경

```bash
# 개발 모드 (소스 코드 핫리로드)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
```

---

## DB 마이그레이션

`ImputationJob` 테이블에 **증강 관련 컬럼 7개**가 추가되어, 기존 DB 사용 시 마이그레이션이 필요합니다.  
운영/데이터 유지가 필요하면 **Alembic 마이그레이션**을 권장합니다.

### Alembic 실행 (예시)

```bash
cd backend
python -m alembic upgrade head
```

### 개발 환경 대안
- 데이터 보존이 필요 없으면 DB를 재생성해도 됩니다.

---

## 환경 변수

### Backend (`backend/.env`)

```ini
# Database
DATABASE_URL=postgresql+psycopg2://imputex:imputex_password@postgres:5432/imputex

# Redis
REDIS_URL=redis://redis:6379/0

# File Storage
UPLOAD_DIR=/app/uploads
RESULT_DIR=/app/results
MAX_UPLOAD_MB=500

# Analysis
SAMPLE_ROWS=1000

# TOTEM
TOTEM_MODEL_PATH=/app/models/totem
TOTEM_CONFIG_PATH=
TOTEM_CODE_PATH=/app/TOTEM/imputation
```

---

## 출처 및 라이선스 고지

- 본 프로젝트는 외부 연구/코드(NAOMI, TOTEM)를 참조해 통합합니다.
- TOTEM 원본 저장소/배포물의 라이선스 표기가 불명확할 수 있으므로 배포/상용 사용 전 별도 확인이 필요합니다.
- 저장소 자체 코드 라이선스와 외부 모델/가중치 라이선스는 분리해 관리해야 합니다.
