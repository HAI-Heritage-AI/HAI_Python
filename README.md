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
