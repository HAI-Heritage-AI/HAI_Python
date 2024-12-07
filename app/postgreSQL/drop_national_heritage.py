import psycopg2

# PostgreSQL 연결 설정
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

    # 기존 테이블 삭제
    cursor.execute("DROP TABLE IF EXISTS national_heritage;")
    print("테이블이 성공적으로 삭제되었습니다.")

except Exception as error:
    print(f"PostgreSQL 연결 또는 테이블 삭제 중 오류가 발생했습니다: {error}")

finally:
    # 연결 닫기
    if cursor:
        cursor.close()
    if connection:
        connection.close()
    print("PostgreSQL 연결이 닫혔습니다.")
