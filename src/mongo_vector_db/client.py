from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote_plus

import certifi
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.operations import SearchIndexModel
from pymongo.server_api import ServerApi

from core.settings import load_environment_variables

logger = logging.getLogger(__name__)

settings = load_environment_variables()

_mongo_vector_db: MongoVectorDB | None = None


def get_mongodb_uri() -> str:
    username = quote_plus(settings.mongo.user.get_secret_value())
    password = quote_plus(settings.mongo.password.get_secret_value())
    cluster_id = settings.mongo.cluster_id
    return f"mongodb+srv://{username}:{password}@cluster0.jsbdmm9.mongodb.net/?appName={cluster_id}"


class MongoVectorDB:
    def __init__(self, uri: str, db_name: str, collection_name: str = "pdf_embeddings"):
        self.client: MongoClient = MongoClient(
            uri,
            server_api=ServerApi("1"),
            tlsCAFile=certifi.where(),
        )
        self.db: Database = self.client[db_name]
        self.collection_name: str = collection_name

    @property
    def collection(self):
        return self.db[self.collection_name]

    def test_database_connection(self):
        self.client["admin"].command("ping")
        logger.info("vector.database.connection.completed")

    def get_all_documents_from_collection(self, collection_name: str):
        return self.db[collection_name].find(limit=10)

    def get_all_search_indexes_from_collection(self, collection_name: str) -> list[Any]:
        list_of_all_search_indexes = self.db[collection_name].list_search_indexes()
        return [index["name"] for index in list_of_all_search_indexes]

    def create_search_index_model(self, index_name: str):
        return SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": 1536,
                        "similarity": "cosine",
                    }
                ]
            },
            name=index_name,
            type="vectorSearch",
        )

    def create_search_index(self, collection_name: str, index_name: str):
        search_index_model = self.create_search_index_model(index_name)
        self.db[collection_name].create_search_index(search_index_model)
        logger.info("vector.search_index.create.completed")

    def query_search_index(self, collection_name: str, index_name: str, query: str):
        return self.db[collection_name].aggregate


def get_mongo_vector_db() -> MongoVectorDB:
    global _mongo_vector_db
    if _mongo_vector_db is None:
        _mongo_vector_db = MongoVectorDB(
            uri=get_mongodb_uri(),
            db_name=settings.mongo.db_name,
        )
    return _mongo_vector_db
