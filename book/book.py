from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

app = FastAPI()

# CORS 설정 (React와 FastAPI 간의 통신 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 로드 (CSV 데이터를 Pandas로 읽기)
df = pd.read_csv('./national_heritage_full_data.csv')

# NaN 및 inf 처리 (NaN을 None으로 대체하지 않고 빈 문자열로 처리)
df_clean = df.replace([np.inf, -np.inf], np.nan).fillna('')  # NaN 값을 빈 문자열로 대체

@app.get("/heritage")
def get_heritage(limit: int = 10, offset: int = 0):
    # 페이징 처리
    paginated_df = df_clean.iloc[offset:offset+limit]
    
    return paginated_df.to_dict(orient="records")

# 필터 데이터 API 추가 (종목, 지역, 시대 필터)
@app.get("/heritage/filters")
def get_filters():
    categories = df_clean['ccmaName'].dropna().unique().tolist()  # 종목
    regions = df_clean['ccbaCtcdNm'].dropna().unique().tolist()  # 지역
    periods = df_clean['ccceName'].dropna().unique().tolist()  # 시대
    return {
        "categories": categories,
        "regions": regions,
        "periods": periods
    }

# 필터된 데이터를 반환
@app.get("/heritage/filter")
def filter_heritage(category: str = None, region: str = None, period: str = None, limit: int = 10, offset: int = 0):
    filtered_df = df.copy()

    if category:
        filtered_df = filtered_df[filtered_df['ccmaName'] == category]
    if region:
        filtered_df = filtered_df[filtered_df['ccbaCtcdNm'] == region]
    if period:
        filtered_df = filtered_df[filtered_df['ccceName'] == period]

    # NaN 처리 (이미 적용하셨으니 유지)
    filtered_df_clean = filtered_df.replace([np.inf, -np.inf], np.nan).dropna()

    # 페이징 처리
    paginated_df = filtered_df_clean.iloc[offset:offset + limit]

    return paginated_df.to_dict(orient="records")

@app.get("/heritage/{heritage_id}")
def get_heritage_by_id(heritage_id: int):
    # 주어진 ID에 해당하는 유산 데이터 반환
    heritage = df_clean[df_clean['ccbaAsno'] == heritage_id].to_dict(orient="records")
    if heritage:
        return heritage[0]
    return {"error": "Heritage not found"}
