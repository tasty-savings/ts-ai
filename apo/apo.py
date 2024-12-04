# 초기 프롬프트가 담긴 변수 P
# 실제로 llm에 돌려보고, 
# 이 결과의 문제점을 정리해서 문제점을 고칠 수 있도록 P를 변경하고,
# 이렇게 변경한 프롬프트를 paraphrase해서 여러 후보군을 만듦
# 변경한 것과 paraphrase한 P들 중에서 bandit selection을 해서 최적의 프롬프트를 생성

from config import OPENAI_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from recipe_change import langfuse_tracking

langfuse_handler = langfuse_tracking()

prompt = ChatPromptTemplate.from_template("")

llm = ChatOpenAI(model="gpt-4o-mini")

chain1 = prompt1 | llm | StrOutputParser()

print(chain2.invoke({"korean_word":"미래"}, config={"callbacks": [langfuse_handler]}))