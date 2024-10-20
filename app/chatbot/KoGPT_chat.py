#app/chatbot/chat.py
from transformers import AutoTokenizer, GPT2LMHeadModel, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline  # 변경된 import 경로

# GPT2 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

# HuggingFace 파이프라인 생성
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=150, no_repeat_ngram_size=3, temperature=0.3, top_p=0.95, num_return_sequences=1, truncation=True)

# langchain-huggingface를 이용한 HuggingFace LLM 래퍼 생성
hf_llm = HuggingFacePipeline(pipeline=pipe)

# LangChain의 PromptTemplate 수정 - 간결한 응답 제공 유도
prompt_template = """사용자의 질문에 대해 간단하고 정확한 답변을 제공해줘.

질문: {input}
답변:"""

prompt = PromptTemplate(input_variables=["input"], template=prompt_template)

# 챗봇 함수 정의
def process_chat(input_text: str) -> str:
    # PromptTemplate를 사용하여 실제 프롬프트 문자열 생성
    prompt_text = prompt.format(input=input_text)
    
    # hf_llm에 문자열 프롬프트를 전달하여 결과 생성
    response = hf_llm.invoke(prompt_text)

    # 응답 형식 확인을 위해 디버깅 출력 추가
    print("Input from User:", input_text)
    print("Response from LLM:", response)
    
    # 응답이 문자열로 반환된 경우 처리
    if isinstance(response, str):
        response_text = response.strip()
    elif isinstance(response, list) and len(response) > 0:
        response_text = response[0]["generated_text"].strip()
    else:
        # 빈 응답일 경우 기본 응답 설정
        response_text = "죄송합니다, 적절한 응답을 생성하지 못했습니다. 다시 한번 말씀해 주세요."
    
    # 응답에서 의미 없는 반복 제거 (예: 중복된 단어나 무의미한 문장)
    if len(set(response_text.split())) < len(response_text.split()) * 0.5:
        response_text = "죄송합니다, 이해할 수 있는 대답을 하지 못했어요. 다시 한번 질문해 주세요."

    # 응답의 길이 제약 추가
    if len(response_text) > 150:
        response_text = response_text[:150] + "..."

    return response_text
