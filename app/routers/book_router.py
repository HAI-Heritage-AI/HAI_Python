from fastapi import APIRouter
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# PostgreSQL 데이터베이스 연결 설정
DATABASE_URL = "postgresql+psycopg2://postgres:iam%40123@localhost/heritage_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 라우터 생성
book_router = APIRouter()

# 모든 유산 데이터 조회
@book_router.get("/heritage")
def get_heritage():
    session = SessionLocal()
    try:
        result = session.execute(text("SELECT * FROM national_heritage")).fetchall()
        heritage_data = [dict(row._mapping) for row in result]
        return heritage_data
    except SQLAlchemyError as e:
        print("Database Error:", e)
        return {"error": str(e)}
    finally:
        session.close()

# 필터 데이터 API 추가 (종목, 지역, 시대 필터)
@book_router.get("/heritage/filter")
def filter_heritage(category: str = None, region: str = None, period: str = None, offset: int = 0, limit: int = 10):
    session = SessionLocal()
    try:
        query = "SELECT * FROM national_heritage WHERE 1=1"
        if category and category != "전체":
            query += " AND ccmaName = :category"
        if region and region != "전체":
            query += " AND ccbaCtcdNm = :region"
        if period and period != "전체":
            query += " AND ccceName = :period"
        query += " OFFSET :offset LIMIT :limit"
        result = session.execute(
            text(query), {
                "category": category,
                "region": region,
                "period": period,
                "offset": offset,
                "limit": limit
            }
        ).fetchall()
        heritage_data = [dict(row._mapping) for row in result]
        return heritage_data
    except SQLAlchemyError as e:
        return {"error": str(e)}
    finally:
        session.close()

# 특정 유산 데이터 조회
@book_router.get("/heritage/{heritage_id}")
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
