import os
import time
import pickle
from dataclasses import dataclass
from pymongo import MongoClient
from typing import List, Dict, Tuple, Set
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict


@dataclass
class RecipeDocument:
    id: str
    title: str
    method_key: str
    ingredients: List[str]
    cooking_steps: List[str]
    recipe_type: List[str]

    def to_langchain_document(self) -> Document:
        """레시피 데이터를 Langchain Document 형식으로 변환"""
        content = f"""
        제목: {self.title}
        레시피 타입: {', '.join(self.recipe_type)}
        조리방식: {self.method_key}
        재료: {', '.join(self.ingredients)}
        조리순서: {', '.join(self.cooking_steps)}
        """

        return Document(
            page_content=content,
            metadata={
                "id": self.id,
                "title": self.title,
                "recipe_type": self.recipe_type,
                "method_key": self.method_key,
                "ingredients": self.ingredients,
                "cooking_steps": self.cooking_steps
            }
        )

class TagSimilarityEngine:
    RECIPE_TYPES = frozenset([
        "건강식", "고단백", "한식", "주식", "채식", "저칼로리", "반찬", "해산물", "간편식",
        "구이", "전통식", "고섬유", "새콤달콤한맛", "볶음", "생식", "저당식", "무침",
        "양식", "매운맛1단계", "매운맛2단계", "매운맛3단계", "매운맛4단계", "매운맛5단계",
        "저염식", "퓨전", "담백한맛", "국", "찜", "디저트", "샐러드", "저탄수화물", "가벼운",
        "튀김", "비타민", "조림", "소화촉진", "음료", "장건강", "체중조절", "중식", "부침",
        "무기질", "삶기", "특별식", "죽", "글루텐프리", "전", "일식", "스프"
    ])

    # 태그 간의 부정적 관계 정의
    TAG_CONFLICTS = {
        '체중조절': {'튀김': 0.8, '디저트': 0.6, '부침': 0.4, '전': 0.4},
        '저칼로리': {'튀김': 0.9, '디저트': 0.7, '부침': 0.5, '전': 0.5},
        '건강식': {'튀김': 0.7, '디저트': 0.5},
        '저탄수화물': {'디저트': 0.8, '튀김': 0.6, '부침': 0.5, '전': 0.5, '주식': 0.7, '죽': 0.6},
        '저당식': {'디저트': 0.9, '새콤달콤한맛': 0.6},
        '소화촉진': {'튀김': 0.7, '매운맛4단계': 0.6, '매운맛5단계': 0.8},
        '간편식': {'전통식': 0.5, '특별식': 0.4},
        '글루텐프리': {'전': 0.8, '부침': 0.8, '튀김': 0.7},
        '채식': {'해산물': 0.9},
        '고단백': {'디저트': 0.5},
        '가벼운': {'튀김': 0.8, '디저트': 0.4, '매운맛4단계': 0.5, '매운맛5단계': 0.6}
    }

    SIMILARITY_CACHE_FILE = "./data/tag_similarities.pkl"
    EMBEDDINGS_CACHE_FILE = "./data/tag_embeddings.pkl"

    def __init__(self, embeddings: HuggingFaceEmbeddings):
        self.embeddings = embeddings
        self._tag_embeddings: Dict[str, np.ndarray] = {}
        self._similarity_cache: Dict[Tuple[str, str], float] = {}
        self._load_or_initialize_cache()

    def _load_or_initialize_cache(self) -> None:
        """캐시된 태그 유사도와 임베딩을 로드하거나 새로 생성"""
        cache_exists = os.path.exists(self.SIMILARITY_CACHE_FILE)
        embeddings_exists = os.path.exists(self.EMBEDDINGS_CACHE_FILE)

        if cache_exists and embeddings_exists:
            # 캐시된 데이터 로드
            with open(self.SIMILARITY_CACHE_FILE, 'rb') as f:
                self._similarity_cache = pickle.load(f)
            with open(self.EMBEDDINGS_CACHE_FILE, 'rb') as f:
                self._tag_embeddings = pickle.load(f)
            print("Loaded cached tag similarities and embeddings")
        else:
            # 새로 계산하고 캐시 저장
            print("Computing tag similarities for the first time...")
            self._initialize_embeddings()
            self._precompute_similarities()

            with open(self.SIMILARITY_CACHE_FILE, 'wb') as f:
                pickle.dump(self._similarity_cache, f)
            with open(self.EMBEDDINGS_CACHE_FILE, 'wb') as f:
                pickle.dump(self._tag_embeddings, f)
            print("Saved tag similarities and embeddings to cache")

    def _initialize_embeddings(self) -> None:
        """태그 임베딩 초기화"""
        embeddings = self.embeddings.embed_documents(list(self.RECIPE_TYPES))
        self._tag_embeddings = {tag: emb for tag, emb in zip(self.RECIPE_TYPES, embeddings)}

    def _precompute_similarities(self) -> None:
        """모든 가능한 태그 조합의 유사도를 계산"""
        for tag1 in self.RECIPE_TYPES:
            for tag2 in self.RECIPE_TYPES:
                if (tag1, tag2) not in self._similarity_cache:
                    similarity = self._compute_similarity(tag1, tag2)
                    self._similarity_cache[(tag1, tag2)] = similarity
                    self._similarity_cache[(tag2, tag1)] = similarity

    def _compute_similarity(self, tag1: str, tag2: str) -> float:
        """태그 간 유사도 계산"""
        if tag1 == tag2:
            return 1.0

        if "매운맛" in tag1 and "매운맛" in tag2:
            level1 = int(tag1[3:-2])
            level2 = int(tag2[3:-2])
            level_diff = abs(level1 - level2)
            return max(0, 1 - (level_diff / 5))

        embedding1 = self._tag_embeddings[tag1]
        embedding2 = self._tag_embeddings[tag2]
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return max(0, float(similarity))

    def calculate_tag_similarity(self, tag1: str, tag2: str) -> float:
        """캐시된 유사도 반환"""
        return self._similarity_cache.get((tag1, tag2), 0.0)


class RecipeSearchEngine:
    def __init__(self, index: FAISS, similarity_engine: TagSimilarityEngine):
        self.index = index
        self.similarity_engine = similarity_engine
        self._recipe_cache: Dict[str, List[str]] = {}
        self.max_workers = min(32, os.cpu_count() * 4)

    @classmethod
    def create(cls, embeddings: HuggingFaceEmbeddings) -> 'RecipeSearchEngine':
        """RecipeSearchEngine 인스턴스 생성을 위한 팩토리 메서드"""
        # if os.path.exists("../data/recipe_index"):
        recipe_index = FAISS.load_local("./data/recipe_index", embeddings, allow_dangerous_deserialization=True)
        # else:
        #     documents = cls.load_recipe_data()
        #     langchain_docs = [doc.to_langchain_document() for doc in documents]
        #     recipe_index = FAISS.from_documents(langchain_docs, embeddings)
        #     recipe_index.save_local("../data/recipe_index")

        similarity_engine = TagSimilarityEngine(embeddings)
        return cls(recipe_index, similarity_engine)

    @staticmethod
    def load_recipe_data() -> List[RecipeDocument]:
        """MongoDB에서 레시피 데이터를 로드"""
        client = MongoClient('mongodb://localhost:27017/')
        db = client['tasty-saving']
        collection = db['recipe']
        recipes = []

        for doc in collection.find():
            try:
                recipe = RecipeDocument(
                    id=doc['_id'],
                    title=doc['title'],
                    method_key=doc['method_key'],
                    recipe_type=doc['recipe_type'],
                    ingredients=doc['ingredients'],
                    cooking_steps=doc['cooking_steps']
                )
                recipes.append(recipe)
            except KeyError as e:
                print(f"Error processing recipe {doc.get('title', 'Unknown')}: {e}")
                continue

        client.close()
        return recipes

    def _calculate_recipe_score(
            self,
            recipe_data: Tuple[Document, float],
            target_types: Set[str]
    ) -> Tuple[Document, float, float]:
        """레시피 점수 계산"""
        doc, vector_score = recipe_data
        recipe_types = set(doc.metadata['recipe_type'])

        similarities = defaultdict(float)
        penalties = defaultdict(float)

        for target_tag in target_types:
            max_similarity = max(
                self.similarity_engine.calculate_tag_similarity(target_tag, recipe_tag)
                for recipe_tag in recipe_types
            )
            similarities[target_tag] = max_similarity

            if target_tag in self.similarity_engine.TAG_CONFLICTS:
                conflicts = self.similarity_engine.TAG_CONFLICTS[target_tag]
                penalty = sum(
                    conflicts.get(recipe_tag, 0)
                    for recipe_tag in recipe_types
                )
                penalties[target_tag] = penalty

        tag_scores = [
            max(0, similarities[tag] - penalties[tag])
            for tag in target_types
        ]

        avg_tag_score = sum(tag_scores) / len(tag_scores) if tag_scores else 0
        return doc, vector_score, avg_tag_score

    def search(self, target_recipe_types: List[str], k: int = 10) -> List[Document]:
        """레시피 검색"""
        target_types = set(target_recipe_types)
        cache_key = ','.join(sorted(target_types))

        if cache_key in self._recipe_cache:
            return self._recipe_cache[cache_key][:k]

        query = f"""
        레시피 특징: {', '.join(target_recipe_types)}
        조리 특징: {', '.join([f'{t}와 관련된 요리' for t in target_recipe_types])}
        """

        initial_results = self.index.similarity_search_with_score(query, len(self.index.docstore._dict))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._calculate_recipe_score, result, target_types)
                for result in initial_results
            ]
            scored_results = [future.result() for future in futures]

        scored_results.sort(key=lambda x: (-x[2], x[1]))
        results = [item[0] for item in scored_results[:k]]

        self._recipe_cache[cache_key] = results
        return results


def recommend_recipes(search_types: List[str], embeddings: HuggingFaceEmbeddings) -> List[str]:
    """레시피 추천"""
    engine_load_time = time.time()
    recipe_search_engine = RecipeSearchEngine.create(embeddings)
    print(f"Engine load time: {time.time() - engine_load_time} seconds")
    
    search_time = time.time()
    results = recipe_search_engine.search(search_types)
    print(f"Search time: {time.time() - search_time} seconds")

    return [str(doc.metadata['id']) for doc in results if hasattr(doc, 'metadata')]