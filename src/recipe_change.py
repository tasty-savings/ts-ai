from typing import TypedDict, Annotated, Sequence, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import Graph, StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from db import MongoDB
from bson import ObjectId
from config import OPENAI_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
from logger import logger_recipe

# Pydantic 모델 정의
class ChangeRecipe(BaseModel):
    main_changes_from_original_recipe: str = Field(description="기본 레시피와 새로운 레시피 사이의 주요 변경점")
    reason_for_changes: str = Field(description="레시피가 바뀐 이유")
    recipe_cooking_order: list = Field(description="조리 순서")
    recipe_cooking_time: str = Field(description="조리 시간")
    recipe_difficulty: str = Field(description="조리 난이도")
    recipe_ingredients: list = Field(description="조리에 사용되는 재료(양)")
    recipe_menu_name: str = Field(description="새로운 레시피의 이름")
    recipe_tips: str = Field(description="조리팁")
    recipe_type: str = Field(description="조리 타입")
    unchanged_parts_and_reasons: str = Field(description="기존 레시피에서 바뀌지 않은 부분과 바뀌지 않은 이유")

class RecipeChangeBalanceNutrition(BaseModel):
    original_recipe_food_group_composition: str = Field(description="기본 레시피의 식품군 구성")
    user_meal_food_group_requirements: str = Field(description="사용자가 끼니당 필요로 하는 식품군 구성")
    new_recipe_food_group_composition: str = Field(description="새로운 레시피의 식품군 구성")
    main_changes_from_original_recipe: str = Field(description="기본 레시피와 새로운 레시피 사이의 주요 변경점")
    reason_for_changes: str = Field(description="레시피가 바뀐 이유")
    recipe_cooking_order: list = Field(description="조리 순서")
    recipe_cooking_time: str = Field(description="조리 시간")
    recipe_difficulty: str = Field(description="조리 난이도")
    recipe_ingredients: list = Field(description="조리에 사용되는 재료(양)")
    recipe_menu_name: str = Field(description="새로운 레시피의 이름")
    recipe_tips: str = Field(description="조리팁")
    recipe_type: str = Field(description="조리 타입")
    unchanged_parts_and_reasons: str = Field(description="기존 레시피에서 바뀌지 않은 부분과 바뀌지 않은 이유")

# 상태 타입 정의
class RecipeState(TypedDict):
    recipe_info: dict
    user_info: dict
    recipe_change_type: int
    key_ingredients_tasty: str | None
    food_group_ratio: str | None
    transformed_recipe: dict | None

# LangGraph 노드 함수들
class RecipeTransformationGraph:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4-mini",
            temperature=0.0,
            max_tokens=1000,
            timeout=20,
            api_key=OPENAI_API_KEY
        )
        self.langfuse_handler = CallbackHandler(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST
        )
        self.langfuse = Langfuse()

    def get_system_prompt(self, langfuse_prompt_name: str) -> PromptTemplate:
        """Langfuse에서 프롬프트 템플릿 가져오기"""
        langfuse_text_prompt = self.langfuse.get_prompt(langfuse_prompt_name)
        return PromptTemplate.from_template(
            langfuse_text_prompt.get_langchain_prompt(),
            metadata={"langfuse_prompt": langfuse_text_prompt}
        )

    async def analyze_key_ingredients(self, state: RecipeState) -> RecipeState:
        """핵심 재료와 맛 분석"""
        prompt = self.get_system_prompt("find_keyIngredients_tasty")
        chain = prompt | self.llm | StrOutputParser()
        state["key_ingredients_tasty"] = await chain.ainvoke(
            {"recipe_info": state["recipe_info"]},
            config={"callbacks": [self.langfuse_handler]}
        )
        return state

    async def analyze_food_group_ratio(self, state: RecipeState) -> RecipeState:
        """식품군 비율 분석 (영양 맞춤형 레시피인 경우)"""
        if state["recipe_change_type"] == 3:
            prompt = self.get_system_prompt("generate_food_group_ratio")
            chain = prompt | self.llm | StrOutputParser()
            state["food_group_ratio"] = await chain.ainvoke(
                {"recipe_info": state["recipe_info"]},
                config={"callbacks": [self.langfuse_handler]}
            )
        return state

    async def transform_recipe(self, state: RecipeState) -> RecipeState:
        """레시피 변환"""
        feature_type = {
            1: "fridge_recipe_transform",
            2: "simple_recipe_transform",
            3: "balance_nutrition"
        }.get(state["recipe_change_type"])
        
        if not feature_type:
            raise ValueError(f"지원하지 않는 recipe_change_type: {state['recipe_change_type']}")

        prompt = self.get_system_prompt(feature_type)
        output_parser = JsonOutputParser(
            pydantic_object=RecipeChangeBalanceNutrition if state["recipe_change_type"] == 3 else ChangeRecipe
        )

        # 프롬프트 준비
        input_data = {
            "recipe_keyIngredients_tasty": state["key_ingredients_tasty"],
            "user_info": state["user_info"],
            "recipe_info": state["recipe_info"]
        }
        
        if state["recipe_change_type"] == 3:
            input_data["original_recipe_food_group_composition"] = state["food_group_ratio"]

        # 변환 체인 실행
        chain = prompt | self.llm | output_parser
        state["transformed_recipe"] = await chain.ainvoke(
            input_data,
            config={"callbacks": [self.langfuse_handler]}
        )
        return state

def create_recipe_graph() -> Graph:
    """레시피 변환 그래프 생성"""
    workflow = RecipeTransformationGraph()
    
    # 그래프 정의
    graph = StateGraph(RecipeState)
    
    # 노드 추가
    graph.add_node("analyze_key_ingredients", workflow.analyze_key_ingredients)
    graph.add_node("analyze_food_group_ratio", workflow.analyze_food_group_ratio)
    graph.add_node("transform_recipe", workflow.transform_recipe)
    
    # 엣지 연결
    graph.set_entry_point("analyze_key_ingredients")
    graph.add_edge("analyze_key_ingredients", "analyze_food_group_ratio")
    graph.add_edge("analyze_food_group_ratio", "transform_recipe")
    
    # 최종 상태를 반환하는 조건 설정
    graph.set_finish_point("transform_recipe")
    
    return graph.compile()

async def generate_recipe(recipe_info: dict, user_info: dict, recipe_change_type: int) -> dict:
    """레시피 생성 메인 함수"""
    try:
        logger_recipe.info("레시피 변환 시작")
        
        # 초기 상태 설정
        initial_state = RecipeState(
            recipe_info=recipe_info,
            user_info=user_info,
            recipe_change_type=recipe_change_type,
            key_ingredients_tasty=None,
            food_group_ratio=None,
            transformed_recipe=None
        )
        
        # 그래프 생성 및 실행
        graph = create_recipe_graph()
        final_state = await graph.ainvoke(initial_state)
        
        logger_recipe.info("레시피 변환 완료")
        return final_state["transformed_recipe"]
    
    except Exception as e:
        logger_recipe.error(f"레시피 생성 중 오류 발생: {str(e)}")
        raise