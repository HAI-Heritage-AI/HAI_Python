# temp_ner.py
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# 모델과 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained("Leo97/KoELECTRA-small-v3-modu-ner")
model = AutoModelForTokenClassification.from_pretrained("Leo97/KoELECTRA-small-v3-modu-ner")

# NER 파이프라인 설정
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# 메타데이터 파일 읽기
def read_metadata(file_path):
    """
    메타데이터를 포함하는 피클 파일을 읽어오는 함수

    Args:
    - file_path (str): 메타데이터 피클 파일의 경로

    Returns:
    - metadata_df (pd.DataFrame): 피클 파일에서 추출한 메타데이터가 포함된 DataFrame
    """
    try:
        # 피클 파일을 로드합니다.
        data = pd.read_pickle(file_path)

        # 데이터가 리스트 형식인지 확인하고 DataFrame으로 변환합니다.
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            metadata_df = pd.DataFrame(data)
            return metadata_df
        else:
            raise ValueError("데이터 형식이 메타데이터 추출에 적합하지 않습니다.")
    
    except Exception as e:
        print(f"메타데이터를 읽는 도중 오류가 발생했습니다: {e}")
        return None

# NER 태깅 수행 함수
def perform_ner_on_first_text(metadata_df):
    """
    첫 번째 데이터의 text_segment에 NER 태깅을 수행하는 함수

    Args:
    - metadata_df (pd.DataFrame): 메타데이터 DataFrame
    """
    if metadata_df is not None and not metadata_df.empty:
        first_text = metadata_df.iloc[0]['text_segment']
        ner_results = ner_pipeline(first_text)
        print(f"첫 번째 텍스트 : \n{first_text}")
        # 분리된 토큰들을 하나의 단어로 병합하여 보기 쉽게 처리
        merged_results = []
        for entity in ner_results:
            if merged_results and entity['entity_group'] == merged_results[-1]['entity_group'] and entity['start'] == merged_results[-1]['end']:
                merged_results[-1]['word'] += entity['word'].replace('##', '')
                merged_results[-1]['end'] = entity['end']
            else:
                merged_results.append(entity)
        
        # 결과 출력
        print("첫 번째 데이터의 NER 결과:")
        for entity in merged_results:
            print(f"Entity: {entity['word']}, Label: {entity['entity_group']}, Score: {entity['score']:.2f}")
    else:
        print("메타데이터가 비어 있거나 유효하지 않습니다.")

# 예시 사용법
if __name__ == "__main__":
    file_path = 'jhgan_metadata.pkl'
    metadata_df = read_metadata(file_path)
    
    if metadata_df is not None:
        perform_ner_on_first_text(metadata_df)
