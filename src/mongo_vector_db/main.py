from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.server_api import ServerApi
from pymongo.operations import SearchIndexModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class MongoVectorDB:
    def __init__(self, uri: str, db_name: str, collection_name: str = "pdf_embeddings"):
        self.client: MongoClient = MongoClient(uri, server_api=ServerApi("1"))
        self.db: Database = self.client[db_name]
        self.collection_name: str = collection_name
    
    @property
    def collection(self):
        return self.db[self.collection_name]

    def test_database_connection(self):
        self.client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!")

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
              "path": "movie_plot_embedding",
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
      
    def query_search_index(self, collection_name: str, index_name: str, query: str):
      return self.db[collection_name].aggregate

# if __name__ == "__main__":
    
#     load_dotenv(PROJECT_ROOT / ".env", override=True)
    
#     username = os.environ.get("username")
#     password = os.environ.get("password")
#     cluster_id = os.environ.get("cluster_id")
    
    
#     MONGO_URI = f'mongodb+srv://{username}:{password}@cluster0.jsbdmm9.mongodb.net/?appName={cluster_id}'
#     db_name = os.environ.get("MONGO_DB", "sample_mflix")
#     if not MONGO_URI:
#         raise SystemExit("Missing MONGO_URI in .env or environment.")

#     # MongoVectorDB(uri=MONGO_URI, db_name=db_name).test_connection()
#     mongodb_instance = MongoVectorDB(uri=MONGO_URI, db_name=db_name)
#     # all_data = mongodb_instance
#     # for doc in all_data:
#     #     print(doc)
#     # try:
#     #   mongodb_instance.create_search_index(collection_name="movies", index_name="movie_plot_embedding_index")
#     # except Exception as e:
#     #   raise e
#     all_search_indexes = mongodb_instance.get_all_search_indexes_from_collection(collection_name="movies")
#     print(all_search_indexes)