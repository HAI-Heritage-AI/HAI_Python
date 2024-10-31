from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 테스트를 위해 모든 도메인 허용으로 변경
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PostgreSQL 데이터베이스 연결 설정
DATABASE_URL = "postgresql+psycopg2://postgres:iam%40123@localhost/heritage_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 모든 유산 데이터 조회
@app.get("/heritage")
def get_heritage():
    session = SessionLocal()
    try:
        query = text("SELECT * FROM national_heritage")
        result = session.execute(query).fetchall()
        heritage_data = [dict(row._mapping) for row in result]  # SQLAlchemy 2.0 이상에서는 _mapping 사용
        return heritage_data
    except SQLAlchemyError as e:
        print("Database Error:", e)  # 에러 로그 추가
        return {"error": str(e)}
    finally:
        session.close()


# 필터 데이터 API 추가 (종목, 지역, 시대 필터)
@app.get("/heritage/filter")
def filter_heritage(category: str = None, region: str = None, period: str = None, offset: int = 0, limit: int = 10):
    session = SessionLocal()
    try:
        # 기본 SQL 쿼리
        query = "SELECT * FROM national_heritage WHERE 1=1"
        
        # 필터 조건 추가
        if category and category != "전체":
            query += " AND ccmaName = :category"
        if region and region != "전체":
            query += " AND ccbaCtcdNm = :region"
        if period and period != "전체":
            query += " AND ccceName = :period"
        
        # 페이징 처리
        query += " OFFSET :offset LIMIT :limit"
        
        # 쿼리 실행
        result = session.execute(
            text(query), {
                "category": category,
                "region": region,
                "period": period,
                "offset": offset,
                "limit": limit
            }
        ).fetchall()
        
        # `_mapping`을 사용하여 각 row를 dict로 변환
        heritage_data = [dict(row._mapping) for row in result]
        return heritage_data
    except SQLAlchemyError as e:
        return {"error": str(e)}
    finally:
        session.close()



# 특정 유산 데이터 조회
@app.get("/heritage/{heritage_id}")
def get_heritage_by_id(heritage_id: int):
    session = SessionLocal()
    try:
        query = text("SELECT * FROM national_heritage WHERE ccbaAsno = :heritage_id")
        result = session.execute(query, {"heritage_id": heritage_id}).fetchone()
        if result:
            return dict(result)
        else:
            return {"error": "Heritage not found"}
    except SQLAlchemyError as e:
        return {"error": str(e)}
    finally:
        session.close()
