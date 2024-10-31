import psycopg2
import pandas as pd
from psycopg2 import sql

# 1. CSV 파일 불러오기
csv_file_path = 'national_heritage_full_data.csv'  # CSV 파일 경로
df = pd.read_csv(csv_file_path)

# 2. PostgreSQL 연결 설정
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

    # 3. CSV 데이터를 PostgreSQL에 삽입
    def insert_data(row):
        insert_query = sql.SQL('''
            INSERT INTO national_heritage (ccbaAsno, ccbaKdcd, ccbaCtcd, ccbaCpno, ccbaMnm1, ccbaMnm2, 
                                          ccmaName, ccbaCtcdNm, ccsiName, ccceName, imageUrl, content, ccbaLcad)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''')
        cursor.execute(insert_query, tuple(row))

    # 4. 데이터 삽입 루프 (각 행을 데이터베이스에 삽입)
    for _, row in df.iterrows():
        try:
            insert_data(row)
        except Exception as e:
            print(f"데이터 삽입 오류: {e}")

    print("모든 데이터가 성공적으로 삽입되었습니다.")

except Exception as error:
    print(f"PostgreSQL 연결 또는 데이터 삽입 중 오류가 발생했습니다: {error}")

finally:
    # cursor와 connection이 정의된 경우에만 닫기
    if 'cursor' in locals() and cursor:
        cursor.close()
    if 'connection' in locals() and connection:
        connection.close()
    print("PostgreSQL 연결이 닫혔습니다.")
