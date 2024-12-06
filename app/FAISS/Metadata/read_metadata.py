# read_metadate.py
import pandas as pd

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

def find_text_segment_nan(metadata_df):
    # 여러 가지 조건을 추가하여 text_segment가 NaN인 경우를 찾습니다.
    nan_rows = metadata_df[
        metadata_df['내용'].isna() |                # NaN으로 판별되는 경우
        (metadata_df['내용'] == '') |               # 빈 문자열인 경우
        (metadata_df['내용'].str.lower() == 'nan')  # 'NaN'이 문자열로 저장된 경우
    ]
    print("text_segment의 내용이 NaN인 행들:")
    print(nan_rows)

# 예시 사용법
if __name__ == "__main__":
    file_path = 'jhgan_metadata.pkl'
    metadata_df = read_metadata(file_path)
    
    if metadata_df is not None:
        print(metadata_df.head())
        print(metadata_df.loc[0, '내용'])
        find_text_segment_nan(metadata_df)
        non_zero_segments = metadata_df[metadata_df['segment_id'] != 0]
        print(non_zero_segments)
