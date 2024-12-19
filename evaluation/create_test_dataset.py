from recipe_change_origin import langfuse_tracking, get_system_prompt, choose_feature, ChangeRecipe, RecipeChangeBalanceNutrition 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from logger import logger_eval
import pandas as pd

langfuse_handler = langfuse_tracking()

llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=1000,
        timeout=20
    )
logger_eval.info("LLM 초기화 완료.")

async def generate_test_dataset(prompt, df):
    prompt = get_system_prompt(prompt)
    llm_chain = prompt | llm | StrOutputParser()

    logger_eval.info("LLM 레시피 생성 중...")
    for i in range(len(df)):
        result = await llm_chain.ainvoke(input={"user_info":df.user_info[i], "recipe_info":df.recipe_info[i]}, config={"callbacks": [langfuse_handler]})
        df["recipe_keyIngredients_tasty"] = result
    logger_eval.info("LLM 레시피 생성 완료")
    return df

# user_info, recipe_info가 있는 dataset
df = pd.read_json('data/combined_dataset.json')
new_df = generate_test_dataset("find_keyIngredients_tasty", df)
print(new_df.describe())