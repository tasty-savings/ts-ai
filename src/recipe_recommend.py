import os
import time
import asyncio
from functools import lru_cache

import numpy as np
from db import MongoDB
from dotenv import load_dotenv
from collections import Counter
from statistics import mean, median
from dataclasses import dataclass
from langfuse import Langfuse
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Set, Tuple

load_dotenv()
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class ScoringResult:
    doc_info: Dict
    type_score: float
    vector_score: float
    has_exact_match: bool

@dataclass
class RecipeDocument:
    id: str
    title: str
    recipe_type: List[str]

    def to_langchain_document(self) -> Document:
        """레시피 데이터를 Langchain Document 형식으로 변환"""
        sorted_types = sorted(self.recipe_type)
        content_parts = [f"이 레시피는 {tag} 요리입니다." for tag in sorted_types]

        return Document(
            page_content=" ".join(content_parts),
            metadata={
                "id": self.id,
                "title": self.title,
                "recipe_type": self.recipe_type
            }
        )

class AsyncRateLimiter:
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.calls = []
        self._lock = asyncio.Lock()

    async def wait_if_needed(self):
        current_time = time.time()
        async with self._lock:
            self.calls = [call_time for call_time in self.calls
                          if current_time - call_time < 60]

            if len(self.calls) >= self.calls_per_minute:
                sleep_time = 60 - (current_time - self.calls[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                self.calls = self.calls[1:]

            self.calls.append(current_time)

class RecipeSearchAnalytics:
    """레시피 검색 결과 분석을 위한 클래스"""
    @staticmethod
    def calculate_metrics(exact_matches: List[Dict],
                          similar_matches: List[Dict],
                          search_types: List[str]) -> Dict:

        all_results = exact_matches + similar_matches

        if not all_results:
            return {
                "total_results": 0,
                "metrics": "No results found"
            }

        # 스코어 분포 분석
        type_scores = [r['type_score'] for r in all_results]
        vector_scores = [r['vector_score'] for r in all_results]

        # 태그 매칭 분석
        all_recipe_types = [t for r in all_results for t in r['recipe_type']]
        tag_frequency = Counter(all_recipe_types)
        search_tag_coverage = {
            tag: tag_frequency[tag] / len(all_results) * 100
            for tag in search_types
        }

        return {
            "total_results": len(all_results),
            "exact_match_ratio": len(exact_matches) / len(all_results) if all_results else 0,
            "score_distribution": {
                "type_score": {
                    "mean": mean(type_scores),
                    "median": median(type_scores),
                    "min": min(type_scores),
                    "max": max(type_scores)
                },
                "vector_score": {
                    "mean": mean(vector_scores),
                    "median": median(vector_scores),
                    "min": min(vector_scores),
                    "max": max(vector_scores)
                }
            },
            "search_tag_coverage": search_tag_coverage,
            "most_common_tags": dict(tag_frequency.most_common(5))
        }

class AsyncRecipeSearch:
    def __init__(self, rate_limit: int = 3000):
        self._type_embeddings = {}
        self._similarity_cache = {}
        self._index = None
        self._index_cache = None
        self._initialized = False
        self._initialization_lock = asyncio.Lock()
        self.data_dir = os.path.join(ROOT_DIR, "data")
        self.rate_limiter = AsyncRateLimiter(rate_limit)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self._langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host="https://cloud.langfuse.com"
        )

    async def initialize(self):
        if self._initialized:
            return

        async with self._initialization_lock:
            if self._initialized:
                return

        await self._load_embeddings()
        await self._load_index()
        self._initialized = True

    async def _load_embeddings(self):
        """레시피 타입 임베딩 로드 또는 계산"""
        cache_path = os.path.join(self.data_dir, "type_embeddings.npy")
        self._type_embeddings = await asyncio.to_thread(np.load, cache_path, allow_pickle=True)
        self._type_embeddings = self._type_embeddings.item()

    async def _load_index(self):
        """FAISS 인덱스 로드"""
        index_path = os.path.join(self.data_dir, "recipe_index")
        if self._index_cache is None:
            self._index_cache = await asyncio.to_thread(
                FAISS.load_local,
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        self._index = self._index_cache

    @staticmethod
    def _get_recipe_types() -> Set[str]:
        """전체 레시피 타입 목록"""
        return {
            "건강식", "고단백", "한식", "주식", "채식", "저칼로리", "반찬", "해산물", "간편식",
            "구이", "전통식", "고섬유", "새콤달콤한맛", "볶음", "생식", "저당식", "무침",
            "양식", "매운맛1단계", "매운맛2단계", "매운맛3단계", "매운맛4단계", "매운맛5단계",
            "저염식", "퓨전", "담백한맛", "국", "찜", "디저트", "샐러드", "저탄수화물", "가벼운",
            "튀김", "비타민", "조림", "소화촉진", "음료", "장건강", "체중조절", "중식", "부침",
            "무기질", "삶기", "특별식", "죽", "글루텐프리", "전", "일식", "스프"
        }

    async def _calculate_type_similarity(self, type1: str, type2: str) -> float:
        """두 레시피 타입 간의 유사도 계산"""
        cache_key = (type1, type2)
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        emb1 = self._type_embeddings[type1]
        emb2 = self._type_embeddings[type2]
        similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

        self._similarity_cache[cache_key] = similarity
        return similarity

    async def _calculate_recipe_score(self, doc_data: Tuple[Document, float], search_types: List[str]) -> ScoringResult:
        """레시피 스코어 계산"""
        doc, vector_score = doc_data
        recipe_types = set(doc.metadata.get('recipe_type', []))

        if not recipe_types:
            return ScoringResult(
                doc_info={
                    'id': str(doc.metadata['id']),
                    'title': doc.metadata['title'],
                    'recipe_type': doc.metadata.get('recipe_type', [])
                },
                type_score=0.0,
                vector_score=float(vector_score),
                has_exact_match=False
            )

        # 정확한 태그 매칭 점수
        matches = set(search_types) & recipe_types
        match_ratio = len(matches) / len(search_types)

        similarity_tasks = []
        for search_type in search_types:
            if search_type in recipe_types:
                similarity_tasks.append(1.0)
            else:
                similarities = await asyncio.gather(*[self._calculate_type_similarity(search_type, recipe_type)
                    for recipe_type in recipe_types])
                similarity_tasks.append(max(similarities) if similarities else 0.0)

        # 최종 스코어 계산
        similarity_score = float(np.mean(similarity_tasks))
        EXACT_MATCH_WEIGHT = 0.6
        SIMILARITY_WEIGHT = 0.2
        VECTOR_SCORE_WEIGHT = 0.2

        final_score = float(
            EXACT_MATCH_WEIGHT * match_ratio
            + SIMILARITY_WEIGHT * similarity_score
            + VECTOR_SCORE_WEIGHT * (1 - vector_score)
        )

        return ScoringResult(
            doc_info={
                'id': str(doc.metadata['id']),
                'title': doc.metadata['title'],
                'recipe_type': doc.metadata.get('recipe_type', [])
            },
            type_score=final_score,
            vector_score=float(vector_score),
            has_exact_match=len(matches) > 0
        )

    async def search_recipes(self, query_types: str, k: int = 100) -> List[str]:
        """레시피 검색"""
        search_types = query_types.split(',')
        query = " ".join(f"이 레시피는 {tag} 요리입니다." for tag in search_types)

        # Langfuse 트레이스 시작
        trace = self._langfuse.trace(
            name="recipe_search",
            metadata={"query_key": query_types, "k": k}
        )

        # FAISS 검색
        await self.rate_limiter.wait_if_needed()
        initial_results = await asyncio.to_thread(
            self._index.similarity_search_with_score,
            query,
            k * 2
        )

        # 스코어 계산
        scored_results = await asyncio.gather(*[
            self._calculate_recipe_score(doc_data, search_types)
            for doc_data in initial_results
        ])

        # 결과 정렬 및 반환
        final_results = sorted(
            [
                {
                    **result.doc_info,
                    'type_score': result.type_score,
                    'vector_score': result.vector_score,
                    'total_score': result.type_score * (1 - result.vector_score)
                }
                for result in scored_results
            ],
            key=lambda x: (-x['total_score'], -x['type_score'])
        )[:k]

        # 분석 메트릭 계산
        analytics = RecipeSearchAnalytics.calculate_metrics(
            [r for r in final_results if any(t in r['recipe_type'] for t in search_types)],
            [r for r in final_results if not any(t in r['recipe_type'] for t in search_types)],
            search_types
        )

        trace.update(
            input={
                "search_types": search_types,
                "query_content": query
            },
            output=[{
                'id': str(doc.metadata['id']),
                'title': doc.metadata['title'],
                'recipe_type': str(doc.metadata.get('recipe_type', [])),
                'vector_score': float(score)
            } for doc, score in initial_results],
            metadata={
                "status": "success",
                "analytics": analytics
            }
        )

        return [str(result['id']) for result in final_results]