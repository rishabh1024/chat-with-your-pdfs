import datetime
import logging
import os
import re
from pathlib import Path
from typing import Any, Literal

import pymupdf
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain_openrouter import ChatOpenRouter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openrouter.errors.toomanyrequestsresponse_error import TooManyRequestsResponseError
from pydantic import SecretStr, ValidationError
from pymongo.errors import ServerSelectionTimeoutError
from tenacity import (
    TryAgain,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
    wait_random,
    wait_random_exponential,
)

from mongo_vector_db.main import MongoVectorDB

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(override=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

logger = logging.getLogger(__name__)

class DocumentIndexer:

    def __init__(
        self,
        structured_llm_instance: ChatOpenRouter,
        file_path: str,
        document_id: str | None = None
    ) -> None:
        self.file_path = file_path
        self.document_id = document_id
        self.cleaned_documents: list[str]
        self.chunked_documents: list[Document]
        self.structured_llm_instance = structured_llm_instance
        self.vector_store: MongoDBAtlasVectorSearch | None = None


    @staticmethod
    def get_mongodb_uri() -> str:

        username = os.environ.get("MONGODB_USER")
        password = os.environ.get("PASSWORD")
        cluster_id = os.environ.get("CLUSTER_ID")

        MONGO_URI = (
            f"mongodb+srv://{username}:{password}@cluster0.jsbdmm9.mongodb.net/?appName={cluster_id}"
        )

        return MONGO_URI

    @staticmethod
    def get_document_index_status_collection():
        mongo_db = MongoVectorDB(
            uri=DocumentIndexer.get_mongodb_uri(),
            db_name=os.environ.get("MONGO_DB", "sample_mflix"),
        )
        return mongo_db.db["document_index_status"]

    @staticmethod
    def get_status_for_document_indexing(
        document_id
    ) -> dict[str, Any] | None:
        if not document_id:
            raise ValueError("document_id is required to store indexing status.")
        status_collection = DocumentIndexer.get_document_index_status_collection()
        return status_collection.find_one({'document_id': document_id})




    @staticmethod
    def set_status_for_document_indexing(document_id,
                                         status: Literal["pending", "success", "failed"],
                                         message: str,
                                        ) -> None:
        if not document_id:
            raise ValueError("document_id is required to store indexing status.")
        now = datetime.datetime.now(datetime.UTC)
        status_collection = DocumentIndexer.get_document_index_status_collection()
        status_collection.update_one(
            {"document_id": document_id},
            {
                "$set": {
                    "status": status,
                    "message": message,
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "created_at": now,
                },
            },
            upsert=True,
        )



    @staticmethod
    def get_mongodb_collection():
        MONGODB_URI = DocumentIndexer.get_mongodb_uri()

        return MongoVectorDB(
            uri=MONGODB_URI, db_name=os.environ.get("MONGO_DB", "sample_mflix")
        ).collection

    @retry(
        stop=stop_after_attempt(2),
        retry=retry_if_exception_type(pymupdf.FileDataError)
    )
    def _load_document(self, page_mode:Literal['page', 'single']='page') -> list[Document]:

        if not Path(self.file_path).is_file():
            raise FileNotFoundError("The PDF file could not be found. Please try uploading again.")

        try:
            loader = PyMuPDFLoader(self.file_path, mode=page_mode)
            documents = loader.load()
            return documents
        except ValueError:
            loader = PyMuPDFLoader(self.file_path, mode="page")
            documents = loader.load()
            return documents
        except pymupdf.EmptyFileError as empty_file_error:
            raise pymupdf.EmptyFileError(
                "The file uploaded is empty. ", """Try uploading a new file."""
            ) from empty_file_error
        except pymupdf.FileDataError as file_data_error:
            raise pymupdf.FileDataError(
                """The file uploaded is either corrupted or not valid.Check for your file"""
                """before uploading and try again."""
            ) from file_data_error
        except Exception as e:
            raise RuntimeError(f"Error Loading Document: {e}") from e


    def preview_data_from_pdf(self) -> None:
        pages = self._load_document()
        for page in pages:
            print(page.page_content[:100])


    def clean_document_data(self) -> None:
        pages: list[Document] = self._load_document()
        self.cleaned_documents = []
        for page in pages:
            # print("Page content: ", page.page_content)
            if len(page.page_content.split()) > 10:
                self.cleaned_documents.append(page.page_content)


    def chunk_and_split_document(self) -> None:
        if not self.cleaned_documents:
            raise ValueError("No cleaned documents to chunk. Call clean_document_data() first.")

        document_to_chunk_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=75,
        )

        self.chunked_document = []

        self.chunked_documents = document_to_chunk_splitter.split_documents(
            [Document(page_content=document) for document in self.cleaned_documents]
        )


    def strip_thinking(self, text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @retry(
    retry=retry_if_exception_type(ValueError),
    stop=stop_after_attempt(3),
    wait=wait_fixed(2)+wait_random(0, 3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
    )
    async def _generate_metadata(self, text: str) -> dict:
        metadata = await self.structured_llm_instance.ainvoke(text)
        if not isinstance(metadata, dict):
            raise ValueError("Metadata generated is not in the expected format.")
        return metadata


    async def add_metadata_to_document(self):

        if not self.chunked_documents:
            raise ValueError(
                "No document chunks are available for metadata generation. "
                "Call chunk_and_split_document() first."
            )

        try:
            for document_chunk in self.chunked_documents:
                if document_chunk.metadata.get("document_id"):
                    continue
                document_metadata = await self._generate_metadata(document_chunk.page_content)
                if document_metadata:

                    if isinstance(document_metadata, str):
                        document_metadata = self.strip_thinking(document_metadata)
                    else:
                        document_chunk.metadata = dict(document_metadata)
                else:
                    print("No metadata found for document")
                if self.document_id:
                    document_chunk.metadata["document_id"] = self.document_id
        except TypeError as type_error:
            raise type_error
        except TooManyRequestsResponseError as too_many_requests_error:
            raise too_many_requests_error
        except Exception as e:
            raise RuntimeError(f"Failed to add metadata to document. Error Reason: {e}") from e


    @staticmethod
    def create_vector_store_instance() -> MongoDBAtlasVectorSearch:

        MONGODB_URI = DocumentIndexer.get_mongodb_uri()
        embedding_model = OpenAIEmbeddings(
            base_url="https://openrouter.ai/api/v1",
            model="perplexity/pplx-embed-v1-0.6b",
            api_key=SecretStr(OPENROUTER_API_KEY or ""),
            embedding_ctx_length=1024,
            check_embedding_ctx_length=False,
            model_kwargs={"encoding_format": "float"},
        )

        vector_store_instance = MongoDBAtlasVectorSearch.from_connection_string(
            connection_string=MONGODB_URI,
            namespace="sample_mflix.pdf_embeddings",
            embedding=embedding_model,
            index_name="document_embeddings",
        )

        return vector_store_instance


    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=2))
    def create_document_embeddings(self) -> dict[str, str]:

        openai_embeddings = OpenAIEmbeddings(
                base_url="https://openrouter.ai/api/v1",
                model="perplexity/pplx-embed-v1-0.6b",
                api_key=SecretStr(OPENROUTER_API_KEY or ""),
                embedding_ctx_length=1024,
                check_embedding_ctx_length=False,
                model_kwargs={"encoding_format": "float"},
            )

        try:
            self.vector_store = MongoDBAtlasVectorSearch.from_documents(
                documents=self.chunked_documents,
                embedding=openai_embeddings,
                collection=DocumentIndexer.get_mongodb_collection(),
                index_name="document_embeddings",
            )
            return  {"status": "Success", "message": "Document Embeddings Created Successfully"}
        except ServerSelectionTimeoutError as server_selection_error:
            raise server_selection_error
        except Exception as e:
            raise e


    def prepare_document_for_embedding_creation(self):

        self.clean_document_data()
        self.chunk_and_split_document()



    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(TryAgain)
        | retry_if_exception_type(TooManyRequestsResponseError)
        | retry_if_exception_type(ServerSelectionTimeoutError)
    )
    async def convert_document_to_vector(self) -> dict[str, str]:
        print("Creating Embeddings for the uploaded Document and adding to the Vector DB")

        if not getattr(self, "chunked_douments", None):
            self.prepare_document_for_embedding_creation()

        try:
            await self.add_metadata_to_document()
            embeddings_status = dict()
            embeddings_status = self.create_document_embeddings()
            print(f"Status for creation of document embedding: {embeddings_status["status"]}")
            if embeddings_status["status"] == "Success":
                return embeddings_status
            else:
                raise TryAgain
        except (ValidationError, TypeError) as e:
            raise e
        except TooManyRequestsResponseError as e:
            raise e
        except ServerSelectionTimeoutError as e:
            raise e
        except Exception as e:
            raise e


    @staticmethod
    def get_similar_documents_from_database(user_query: str) -> list[Document] | dict:
        try:
            vector_store_instance = DocumentIndexer.create_vector_store_instance()
            vector_store_retriever = vector_store_instance.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3, "score_threshold": 0.50},
            )

            documents = vector_store_retriever.invoke(user_query)
            return documents
        except ServerSelectionTimeoutError as e:
            return {
                "status": "Failed",
                "message": "Failed to Create Document Embeddings. ",
                "error": str(e),
            }
        except Exception as e:
            return {"status": "Failed",
                    "message": "Failed to create Document Embeddings", "error": str(e)}


# async def main():
#     load_dotenv(PROJECT_ROOT / ".env", override=True)

#     schema = {
#         "title": "DocumentMetadata",
#         "properties": {
#             "title": {"type": "string"},
#             "keywords": {"type": "array", "items": {"type": "string"}},
#             "hasCode": {"type": "boolean"},
#         },
#         "required": ["title", "keywords",
#                      "hasCode"],
#     }

#     structured_llm_client = ChatOpenRouter(
#         model="cohere/command-r7b-12-2024",
#         api_key=SecretStr(OPENROUTER_API_KEY or ""),
#         model_kwargs={
#             "models": [
#                 "qwen/qwen-2.5-7b-instruct",
#                 "openai/gpt-oss-20b:free",
#                 "meta-llama/llama-3.2-3b-instruct:free",
#             ]
#         },
#         verbose=True,
#         temperature=0.4,
#     ).with_structured_output(schema, method="json_schema")

#     document_indexer = DocumentIndexer(
#         file_path="Linkedin-Profile.pdf",
#         structured_llm_instance=structured_llm_client
#     )

#     document_indexer.clean_document_data()
#     print("Cleaned document data....")
#     document_indexer.chunk_and_split_document()
#     print("Chunked and split document data....")
#     # await document_indexer.add_metadata_to_document()
#     # print("Added metadata to document data....")

#     for document in document_indexer.chunked_documents:
#         print("Document metadata: ", document.metadata.keys())
#     print("Metadata: ", document_indexer.chunked_documents)
#     document_indexer.create_document_embeddings()
#     print("Created document embeddings....")

#     vector_store_instance: MongoDBAtlasVectorSearch = (
#         DocumentIndexer.create_vector_store_instance()
#     )
#     retriever = vector_store_instance.as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": 3, "score_threshold": 0.01},
#     )

#     documents = retriever.invoke("Who is Rishabh?")
#     for document in documents:
#         print(document)

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())