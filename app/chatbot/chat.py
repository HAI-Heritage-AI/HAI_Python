from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import torch

# XLM-R 모델과 토크나이저 로드
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base")

# 챗봇 함수 정의
def process_chat(input_text):
    # 입력 텍스트를 토큰화
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # 모델을 통해 답변 예측
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 예측 결과를 해석 (여기서는 간단하게 logits을 출력으로 사용)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    
    # 간단한 응답 처리 (여기서는 0 또는 1로 분류된 결과를 가정)
    if predicted_class == 0:
        return "챗봇 응답: 긍정적인 대답입니다!"
    else:
        return "챗봇 응답: 부정적인 대답입니다!"
