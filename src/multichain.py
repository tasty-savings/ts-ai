from config import OPENAI_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from recipe_change import langfuse_tracking

langfuse_handler = langfuse_tracking()

prompt1 = ChatPromptTemplate.from_template("translates {korean_word} to English.")
prompt2 = ChatPromptTemplate.from_template(
    "explain {english_word} using oxford dictionary to me in Korean."
)

llm = ChatOpenAI(model="gpt-4o-mini")

chain1 = prompt1 | llm | StrOutputParser()

#print(chain1.invoke({"korean_word":"미래"}, config={"callbacks": [langfuse_handler]}))

chain2 = (
    {"english_word": chain1}
    | prompt2
    | llm
    | StrOutputParser()
)

print(chain2.invoke({"korean_word":"미래"}, config={"callbacks": [langfuse_handler]}))