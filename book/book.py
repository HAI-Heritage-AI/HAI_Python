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


@app.get("/heritage")
def get_heritage(limit: int = 10, offset: int = 0):
    # NaN, inf, -inf 값을 처리하고 None으로 변환
    df_clean = df.replace([np.inf, -np.inf], np.nan).where(pd.notnull(df), None)

    # 페이징 처리
    paginated_df = df_clean.iloc[offset:offset+limit]
    
    return paginated_df.to_dict(orient="records")

@app.get("/heritage/filter")
def filter_heritage(category: str = None, region: str = None, period: str = None):
    filtered_df = df
    
    # 필터 적용
    if category:
        filtered_df = filtered_df[filtered_df['ccmaName'] == category]
    if region:
        filtered_df = filtered_df[filtered_df['ccbaCtcdNm'] == region]
    if period:
        filtered_df = filtered_df[filtered_df['ccceName'] == period]
    
    # NaN, inf, -inf 값을 None으로 변환
    filtered_df_clean = filtered_df.replace([np.inf, -np.inf], np.nan).fillna(None)
    
    return filtered_df_clean.to_dict(orient="records")

@app.get("/heritage/{heritage_id}")
def get_heritage_by_id(heritage_id: int):
    # 주어진 ID에 해당하는 유산 데이터 반환
    heritage = df[df['ccbaAsno'] == heritage_id].to_dict(orient="records")
    if heritage:
        return heritage[0]
    return {"error": "Heritage not found"}

@app.get("/heritage/filter")
def filter_heritage(category: str = None, region: str = None, period: str = None):
    # 카테고리, 지역, 시대에 따른 필터링 처리
    filtered_df = df
    if category:
        filtered_df = filtered_df[filtered_df['ccmaName'] == category]
    if region:
        filtered_df = filtered_df[filtered_df['ccbaCtcdNm'] == region]
    if period:
        filtered_df = filtered_df[filtered_df['ccceName'] == period]
    
    return filtered_df.to_dict(orient="records")
