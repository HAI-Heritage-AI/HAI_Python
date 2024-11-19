from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
import json
from datetime import datetime
import pandas as pd

# 축제 데이터 파일 로드
file_path = 'app/travel/data/festival/festival.csv'
try:
    festival_data = pd.read_csv(file_path, encoding='cp949')
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="축제 데이터 파일을 찾을 수 없습니다.")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"데이터를 로드하는 중 오류가 발생했습니다: {e}")

# FastAPI Router 생성
festival_router = APIRouter()

# 축제 데이터에서 start_date를 datetime으로 변환할 때 수정
@festival_router.get("/")
async def get_festivals(
    destination: str = Query(..., description="목적지(제공기관명)"),
    start_date: str = Query(..., description="조회 시작 날짜 (YYYY-MM-DD 형식)"),
):
    """
    특정 목적지와 날짜 이후에 열리는 축제 목록을 반환합니다.
    """
    try:
        # 날짜를 datetime 형식으로 변환
        start_date = datetime.strptime(start_date, "%Y-%m-%d")  # str -> datetime 변환
    except ValueError:
        raise HTTPException(status_code=400, detail="잘못된 날짜 형식입니다. YYYY-MM-DD 형식을 사용하세요.")

    # 목적지와 날짜로 필터링
    try:
        filtered_data = festival_data[
            (festival_data['제공기관명'].str.startswith(destination)) &  # 목적지 조건
            (pd.to_datetime(festival_data['축제시작일자'], errors='coerce') >= start_date)  # 날짜 조건
        ]
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"필수 컬럼이 누락되었습니다: {e}")

    # 필터링 결과 반환
    result = filtered_data[['축제명', '개최장소', '축제내용', '전화번호',
                            '홈페이지주소', '소재지도로명주소', '축제시작일자', '축제종료일자']]

    # JSONResponse를 사용하여 ensure_ascii=False 설정
    return JSONResponse(
        content=json.loads(result.to_json(orient="records", force_ascii=False)),
        media_type="application/json"
    )
