from pymongo import MongoClient
from config import MONGO_DB_URI, MONGO_DB_NAME
from logger import logger_db
import time
from contextlib import contextmanager


class MongoDB:
    def __init__(self):
        self._connect_to_db()

    def _connect_to_db(self):
        """MongoDB 연결 초기화"""
        try:
            logger_db.info(f"MongoDB 연결 시도 중... (URI: {MONGO_DB_URI})")
            self.client = MongoClient(MONGO_DB_URI)
            self.db = self.client[MONGO_DB_NAME]
            # 연결 테스트
            self.client.server_info()
            logger_db.info(f"MongoDB '{MONGO_DB_NAME}' 데이터베이스 연결 성공")
        except Exception as e:
            logger_db.error(f"MongoDB 연결 실패: {str(e)}", exc_info=True)
            raise


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @contextmanager
    def _measure_time(self, operation_name):
        """작업 실행 시간을 측정"""
        start_time = time.time()
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            logger_db.debug(f"{operation_name} 실행 시간: {execution_time:.4f}초")

    def _log_and_get_collection(self, collection_name):
        """컬렉션 접근 시 로깅 및 반환"""
        logger_db.debug(f"컬렉션 '{collection_name}' 접근")
        return self.db[collection_name]

    def find_one(self, collection_name, query=None, projection=None):
        """단일 문서 조회"""
        query = query or {}
        projection = projection or {}
        try:
            collection = self._log_and_get_collection(collection_name)
            logger_db.info(f"단일 문서 조회 - 컬렉션: {collection_name}, 쿼리: {query}")

            with self._measure_time(f"find_one 쿼리 ({collection_name})"):
                result = collection.find_one(query, projection)
            if result:
                logger_db.info(f"문서 조회 성공 - ID: {result.get('_id')}")
            else:
                logger_db.warning(f"문서를 찾을 수 없음 - 컬렉션: {collection_name}, 쿼리: {query}")
            return result
        except Exception as e:
            logger_db.error(f"단일 문서 조회 실패 - 컬렉션: {collection_name}, 쿼리: {query}", exc_info=True)
            raise

    def find_many(self, collection_name, query=None, projection=None):
        """다중 문서 조회"""
        query = query or {}
        projection = projection or {}
        try:
            collection = self._log_and_get_collection(collection_name)
            logger_db.info(f"다중 문서 조회 - 컬렉션: {collection_name}, 쿼리: {query}")

            with self._measure_time(f"find_many 쿼리 ({collection_name})"):
                results = list(collection.find(query, projection))

            logger_db.info(f"문서 {len(results)}개 조회 성공 - 컬렉션: {collection_name}")
            return results
        except Exception as e:
            logger_db.error(f"다중 문서 조회 실패 - 컬렉션: {collection_name}, 쿼리: {query}", exc_info=True)
            raise

    def close(self):
        """MongoDB 연결 종료"""
        try:
            self.client.close()
            logger_db.info("MongoDB 연결 종료")
        except Exception as e:
            logger_db.error("MongoDB 연결 종료 중 오류 발생", exc_info=True)

