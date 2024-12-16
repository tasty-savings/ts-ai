from typing import TypedDict, Annotated, Sequence
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from typing import Dict, List, Any
import json

# 상태 정의
class RecipeState(TypedDict):
    user_info: dict
    recipe_info: dict
    recipe_change_type: int
    key_ingredients_tasty: str
    food_group_ratio: str
    generated_recipe: dict
    errors: List[str]
    status: str

# 노드 함수들
async def initialize_state(state: RecipeState) -> RecipeState:
    """초기 상태 설정"""
    state["errors"] = []
    state["status"] = "initialized"
    return state

async def analyze_key_ingredients(
    state: RecipeState,
    llm: ChatOpenAI,
    key_ingredients_prompt: PromptTemplate
) -> RecipeState:
    """핵심 재료와 맛 분석"""
    try:
        result = await llm.ainvoke(
            key_ingredients_prompt.format(recipe_info=state["recipe_info"])
        )
        state["key_ingredients_tasty"] = result.content
        state["status"] = "key_ingredients_analyzed"
    except Exception as e:
        state["errors"].append(f"Key ingredients analysis failed: {str(e)}")
        state["status"] = "error"
    return state

async def analyze_food_group(
    state: RecipeState,
    llm: ChatOpenAI,
    food_group_prompt: PromptTemplate
) -> RecipeState:
    """식품군 분석 (영양 맞춤형 레시피인 경우)"""
    if state["recipe_change_type"] == 3:
        try:
            result = await llm.ainvoke(
                food_group_prompt.format(recipe_info=state["recipe_info"])
            )
            state["food_group_ratio"] = result.content
            state["status"] = "food_group_analyzed"
        except Exception as e:
            state["errors"].append(f"Food group analysis failed: {str(e)}")
            state["status"] = "error"
    return state

async def generate_transformed_recipe(
    state: RecipeState,
    llm: ChatOpenAI,
    transform_prompts: Dict[int, PromptTemplate],
    output_parsers: Dict[int, JsonOutputParser]
) -> RecipeState:
    """레시피 변환"""
    try:
        prompt = transform_prompts[state["recipe_change_type"]]
        parser = output_parsers[state["recipe_change_type"]]
        
        # 프롬프트 입력 준비
        prompt_input = {
            "recipe_keyIngredients_tasty": state["key_ingredients_tasty"],
            "user_info": state["user_info"],
            "recipe_info": state["recipe_info"]
        }
        if state["recipe_change_type"] == 3:
            prompt_input["original_recipe_food_group_composition"] = state["food_group_ratio"]
        
        result = await llm.ainvoke(prompt.format(**prompt_input))
        parsed_result = parser.parse(result.content)
        
        state["generated_recipe"] = parsed_result
        state["status"] = "recipe_generated"
    except Exception as e:
        state["errors"].append(f"Recipe transformation failed: {str(e)}")
        state["status"] = "error"
    return state

# 그래프 생성 함수
def create_recipe_graph(
    llm: ChatOpenAI,
    prompts: Dict[str, PromptTemplate],
    output_parsers: Dict[int, JsonOutputParser]
) -> Graph:
    """레시피 변환 그래프 생성"""
    
    workflow = StateGraph(RecipeState)
    
    # 노드 추가
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("analyze_key_ingredients", 
                     lambda state: analyze_key_ingredients(state, llm, prompts["key_ingredients"]))
    workflow.add_node("analyze_food_group",
                     lambda state: analyze_food_group(state, llm, prompts["food_group"]))
    workflow.add_node("generate_recipe",
                     lambda state: generate_transformed_recipe(state, llm, prompts, output_parsers))
    
    # 엣지 연결
    workflow.add_edge("initialize", "analyze_key_ingredients")
    workflow.add_conditional_edges(
        "analyze_key_ingredients",
        lambda state: "analyze_food_group" if state["recipe_change_type"] == 3 else "generate_recipe"
    )
    workflow.add_edge("analyze_food_group", "generate_recipe")
    
    # 종료 조건 설정
    workflow.set_finish_criterion(lambda state: state["status"] == "recipe_generated" or state["status"] == "error")
    
    return workflow.compile()

# 메인 실행 함수
async def generate_recipe(
    recipe_info: dict,
    user_info: dict,
    recipe_change_type: int,
    config: dict = None
) -> dict:
    """레시피 생성 실행"""
    
    # LLM 초기화
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=1000,
        timeout=20
    )
    
    # 프롬프트 및 파서 초기화
    prompts = {
        "key_ingredients": get_system_prompt("find_keyIngredients_tasty"),
        "food_group": get_system_prompt("generate_food_group_ratio"),
        1: get_system_prompt("fridge_recipe_transform"),
        2: get_system_prompt("simple_recipe_transform"),
        3: get_system_prompt("balance_nutrition")
    }
    
    output_parsers = {
        1: JsonOutputParser(pydantic_object=ChangeRecipe),
        2: JsonOutputParser(pydantic_object=ChangeRecipe),
        3: JsonOutputParser(pydantic_object=RecipeChangeBalanceNutrition)
    }
    
    # 그래프 생성
    graph = create_recipe_graph(llm, prompts, output_parsers)
    
    # 초기 상태 설정
    initial_state: RecipeState = {
        "user_info": user_info,
        "recipe_info": recipe_info,
        "recipe_change_type": recipe_change_type,
        "key_ingredients_tasty": "",
        "food_group_ratio": "",
        "generated_recipe": {},
        "errors": [],
        "status": "starting"
    }
    
    # 그래프 실행
    final_state = await graph.ainvoke(initial_state)
    
    if final_state["status"] == "error":
        raise Exception("\n".join(final_state["errors"]))
    
    return final_state["generated_recipe"]