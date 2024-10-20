#app/chatbot/chat.py
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import torch

# KoGPT 모델과 토크나이저 로드 (전역 변수로 설정하여 재사용)
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                    pad_token='<pad>', mask_token='<mask>')
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

# 챗봇 함수 정의
def process_chat(input_text: str) -> str:
    # 입력 텍스트를 KoGPT 형식에 맞게 토큰화
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # 모델을 통해 예측 수행 (대화 응답 생성)
    with torch.no_grad():
        generated_ids = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)
    
    # 생성된 텍스트를 디코딩하여 응답으로 변환
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return response

