from typing import Annotated, TypedDict, Sequence
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# 환경 변수에서 API 키 가져오기
tavily_api_key = os.getenv("TAVILY_API_KEY")

# State 정의
class State(TypedDict):
   messages: Annotated[list, add_messages]

# 검색 도구 설정
travel_tool = TavilySearchResults(    
   max_results=3,
   search_depth="advanced",
   include_raw_content=True,
#    include_domains=["instagram.com", "naver.com", "tistory.com"],
   k=5,
)

food_tool = TavilySearchResults(    
   max_results=3,
   search_depth="advanced",
   include_raw_content=True,
#    include_domains=["instagram.com", "naver.com", "tistory.com", "mango.com"],
   k=5,
)

# LLM 모델 설정
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 프롬프트 템플릿 설정
travel_prompt = ChatPromptTemplate.from_messages([
   ("system", """한국의 최신 트렌드와 여행 정보를 제공하는 AI 어시스턴트입니다. 
   관광지, 전시회, 포토스팟 등 SNS에서 인기있는 여행지 정보만 제공합니다.
   맛집이나 카페는 제외하고 순수 관광지/포토스팟만 추천해주세요.
   
   1. 장소는 반드시 정확한 상호명과 주소를 포함할 것
   2. 실제 존재하는 구체적인 장소만 추천할 것
   3. 인스타그램에서 인기있는 포토스팟 위주로 추천할 것
   4. 입장료나 이용료가 있다면 반드시 포함할 것"""),
   MessagesPlaceholder(variable_name="messages"),
   MessagesPlaceholder(variable_name="agent_scratchpad")
])

food_prompt = ChatPromptTemplate.from_messages([
   ("system", """한국의 최신 트렌드와 맛집 정보를 제공하는 AI 어시스턴트입니다. 
   SNS에서 인기있는 맛집과 카페 정보만 제공합니다.
   관광지나 포토스팟은 제외하고 순수 맛집/카페만 추천해주세요.
   
   1. 장소는 반드시 정확한 상호명과 주소를 포함할 것
   2. 실제 존재하는 구체적인 맛집만 추천할 것
   3. 대표 메뉴와 가격대를 반드시 포함할 것
   4. 영업시간과 휴무일 정보도 포함할 것"""),
   MessagesPlaceholder(variable_name="messages"),
   MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Agent 설정
from langchain.agents import create_openai_tools_agent
travel_agent = create_openai_tools_agent(llm, [travel_tool], travel_prompt)
food_agent = create_openai_tools_agent(llm, [food_tool], food_prompt)

# Agent executor 설정
from langchain.agents import AgentExecutor
travel_executor = AgentExecutor(agent=travel_agent, tools=[travel_tool])
food_executor = AgentExecutor(agent=food_agent, tools=[food_tool])

# Chatbot 노드 함수
def travel_chatbot(state: State):
    messages = state["messages"]
    # 여행지 검색을 위한 구체적인 쿼리 생성
    travel_query = "서울 관광지 포토스팟 2024 인스타그램 핫플레이스 전시회 "
    result = travel_executor.invoke({
        "messages": messages + [HumanMessage(content=travel_query)]
    })
    return {"messages": [AIMessage(content=result["output"])]}

def food_chatbot(state: State):
    messages = state["messages"]
    # 맛집 검색을 위한 구체적인 쿼리 생성
    food_query = "서울 20대 여성 맛집 카페 2024 인스타그램 핫플레이스 "
    result = food_executor.invoke({
        "messages": messages + [HumanMessage(content=food_query)]
    })
    return {"messages": [AIMessage(content=result["output"])]}

# 결과 합치는 함수
def combine_results(state: State):
   messages = state["messages"]
   combined_content = """### 🎯 추천 여행 정보\n\n""" + messages[-2].content + """\n\n### 🍽 추천 맛집 정보\n\n""" + messages[-1].content
   return {"messages": [AIMessage(content=combined_content)]}

# should_continue 함수
def should_continue(state: State) -> Sequence[str]:
   messages = state["messages"]
   last_message = messages[-1]
   
   if not hasattr(should_continue, 'count'):
       should_continue.count = 0
   
   if should_continue.count < 1:  
       should_continue.count += 1
       return ["tools"]
   return ["__end__"]

# 그래프 구성
graph_builder = StateGraph(State)

# 노드 추가
graph_builder.add_node("travel_chatbot", travel_chatbot)
graph_builder.add_node("food_chatbot", food_chatbot)
graph_builder.add_node("combine_results", combine_results)

# 엣지 추가
graph_builder.add_edge("travel_chatbot", "food_chatbot")
graph_builder.add_edge("food_chatbot", "combine_results")
graph_builder.add_edge("combine_results", END)

# 시작점 설정
graph_builder.set_entry_point("travel_chatbot")
graph = graph_builder.compile()

# 사용 예시
messages = [HumanMessage(content="20대 여성이 서울을 여행하는데 SNS/감성 위주의 핫플 소개해줘")]
result = graph.invoke({"messages": messages}, {"recursion_limit": 10})
print(result["messages"])