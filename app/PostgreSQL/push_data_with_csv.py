# 필요한 라이브러리 불러오기
import psycopg2
import pandas as pd
from psycopg2 import sql

# 1. PostgreSQL 연결 설정
try:
    connection = psycopg2.connect(
        host="localhost",  # PostgreSQL 서버 주소
        database="heritage_db",  # 사용할 데이터베이스 이름
        user="postgres",  # 사용자명
        password="iam@123"  # 설정한 비밀번호
    )
    connection.autocommit = True  # 자동 커밋 설정
    cursor = connection.cursor()

    print("PostgreSQL에 성공적으로 연결되었습니다.")

    # 2. 테이블 존재 여부 확인 후 생성 (기존에 테이블이 없다면 생성)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS national_heritage (
            id SERIAL PRIMARY KEY,  -- 기본 키로 사용할 id 컬럼 추가
            ccbaAsno NUMERIC,
            ccbaKdcd INTEGER,
            ccbaCtcd VARCHAR(10),
            ccbaCpno VARCHAR(20),
            ccbaMnm1 VARCHAR(255),
            ccbaMnm2 VARCHAR(255),
            ccmaName VARCHAR(100),
            ccbaCtcdNm VARCHAR(100),
            ccsiName VARCHAR(100),
            ccceName VARCHAR(255),
            imageUrl VARCHAR(500),
            content TEXT,
            ccbaLcad VARCHAR(500)
        );
    ''')
    print("테이블이 성공적으로 생성되었거나 이미 존재합니다.")

    # 4. CSV 파일 불러오기
    csv_file_path = 'updated01_national_heritage_full_data.csv'
    df = pd.read_csv(csv_file_path, encoding='utf-8') 
    df.to_csv('updated01_national_heritage_utf8.csv', index=False, encoding='utf-8')

    # 5. 데이터 삽입 함수 정의
    def insert_data(row):
        try:
            # 관리번호를 문자열로 변환하여 삽입
            ccbaAsno = str(row['관리번호(ccbaAsno)']) if not pd.isna(row['관리번호(ccbaAsno)']) else None

            insert_query = sql.SQL('''
                INSERT INTO national_heritage (ccbaAsno, ccbaKdcd, ccbaCtcd, ccbaCpno, ccbaMnm1, ccbaMnm2, 
                                              ccmaName, ccbaCtcdNm, ccsiName, ccceName, imageUrl, content, ccbaLcad)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''')
            cursor.execute(insert_query, (
                ccbaAsno, row['중목코드(ccbaKdcd)'], row['시도코드(ccbaCtcd)'], row['국가유산연계번호(ccbaCpno)'],
                row['국가유산명_국문(ccbaMnm1)'], row['국가유산명_한자(ccbaMnm2)'], row['국가유산종목(ccmaName)'],
                row['시도명(ccbaCtcdNm)'], row['시군구명(ccsiName)'], row['시대(ccceName)'],
                row['메인이미지URL(imageUrl)'], row['내용(content)'], row['소재지상세(ccbaLcad)']
            ))
        except Exception as e:
            print(f"데이터 삽입 오류: {e}")

    # 6. 데이터 삽입 루프 (각 행을 데이터베이스에 삽입)
    for _, row in df.iterrows():
        insert_data(row)

    print("모든 데이터가 성공적으로 삽입되었습니다.")

except Exception as error:
    print(f"PostgreSQL 연결 또는 데이터 삽입 중 오류가 발생했습니다: {error}")

finally:
    # 7. 연결 닫기
    if cursor:
        cursor.close()
    if connection:
        connection.close()
    print("PostgreSQL 연결이 닫혔습니다.")
