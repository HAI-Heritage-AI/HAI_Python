# 처음 시작

## 콘다 가상환경 설정
conda create -n HAI_Python python=3.8

## 콘다 가상환경 실행
conda activate HAI_Python

## 의존성 설치
pip install -r requirements.txt

## 현재 설치된 패키지 목록 확인 및 출력
pip freeze > requirements.txt

## Uvicorn 실행
uvicorn main:app --reload

# 구조 설명

## main.py
- FastAPI 애플리케이션을 생성하고 엔드포인트에 대한 라우터를 등록
- 프로그램의 진입점으로 uvicorn을 사용해 서버를 실행

## chatbot_router.py
- /api/chatbot 경로데 대한 엔드포인트를 정의
- /api/chatbot 경로로 POST 요청이 들어오면 결과를 반환

## proce