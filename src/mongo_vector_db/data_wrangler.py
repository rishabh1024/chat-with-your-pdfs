import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain_openrouter import ChatOpenRouter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from openrouter.errors.toomanyrequestsresponse_error import TooManyRequestsResponseError
from pydantic import SecretStr, ValidationError
from pymongo.errors import ServerSelectionTimeoutError
from tenacity import retry, stop_after_attempt, wait_exponential

from mongo_vector_db.main import MongoVectorDB

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(override=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

schema = {
    "title": "DocumentMetadata",
    "properties": {
        "title": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "hasCode": {"type": "boolean"},
    },
    "required": ["title", "keywords", "hasCode"],
}

class DocumentIndexer:
    def __init__(
        self,
        file_path: str,
        mongo_vector_db: MongoVectorDB,
        document_id: str | None = None,
    ) -> None:
        self.file_path = file_path
        self.document_id = document_id
        self.document_to_chunk_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=75,
        )
        self.cleaned_documents: list[str] = []
        self.chunked_documents: list[Document] = []
        self.openai_client: OpenAI = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        self.collection = mongo_vector_db.collection
        self.llm_with_structured_output = ChatOpenRouter(
            model="cohere/command-r7b-12-2024",
            api_key=SecretStr(OPENROUTER_API_KEY or ""),
            model_kwargs={
                "models": [
                    "qwen/qwen-2.5-7b-instruct",
                    "openai/gpt-oss-20b:free",
                    "meta-llama/llama-3.2-3b-instruct:free",
                ]
            },
            verbose=True,
            temperature=0.4,
        ).with_structured_output(schema, method="json_schema")
        self.vector_store: MongoDBAtlasVectorSearch | None = None
    
    @staticmethod
    def get_mongodb_client() -> MongoVectorDB:
        username = os.environ.get("username")
        password = os.environ.get("password")
        cluster_id = os.environ.get("cluster_id")

        MONGO_URI = (
            f"mongodb+srv://{username}:{password}@cluster0.jsbdmm9.mongodb.net/?appName={cluster_id}"
        )

        mongo_vector_db = MongoVectorDB(
            uri=MONGO_URI, db_name=os.environ.get("MONGO_DB", "sample_mflix")
        )
        
        # print("MongoDB client created successfully: ", MONGO_URI)
        return mongo_vector_db
    def _load_document(self) -> list[Document]:
        try:
            print("Loading document: ", self.file_path)
            loader = PyMuPDFLoader(self.file_path)
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"Error loading document: {e}")
            return []
    
    def preview_data_from_pdf(self) -> None:
        pages = self._load_document()
        for page in pages:
            print(page.page_content[:100])

    def clean_document_data(self) -> None:
        pages = self._load_document()
        for page in pages:
            # print("Page content: ", page.page_content)
            if len(page.page_content.split(" ")) > 10:
                print("Page content is greater than 10 words")
                self.cleaned_documents.append(page.page_content)

    def chunk_and_split_document(self) -> None:
        for document in self.cleaned_documents:
            chunks = self.document_to_chunk_splitter.split_documents(
                documents=[Document(page_content=document)]
            )
            for chunk in chunks:
                self.chunked_documents.append(chunk)
    
    def strip_thinking(self, text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    async def add_metadata_to_document(self):

        for attempt in range(3):
            try:
                for doc in self.chunked_documents:
                    # print("doc data: ", doc.page_content)
                    document_metadata = await self.llm_with_structured_output.ainvoke(doc.page_content)

                    # print("Document metadata: ", document_metadata)
                    if document_metadata:
                        if isinstance(document_metadata, str):
                            document_metadata = self.strip_thinking(document_metadata)
                        else:
                            doc.metadata = dict(document_metadata)
                    else:
                        print("No metadata found for document")
                    if self.document_id:
                        doc.metadata["document_id"] = self.document_id
                return
            except (ValidationError, TypeError) as e:
                wait = min(2**attempt * 2, 60)
                print(f"Validation error, retrying in {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
            except TooManyRequestsResponseError as e:
                retry_after = e.raw_response.headers.get("retry-after")
                wait = float(retry_after) if retry_after else min(2**attempt * 2, 60)
                print(f"Rate limited, retrying in {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
            except Exception as e:
                raise RuntimeError(f"Failed to add metadata to document. Error Reason: {e}") from e
    
    @staticmethod
    def create_vector_store() -> MongoDBAtlasVectorSearch:
        username = os.environ.get("username")
        password = os.environ.get("password")
        cluster_id = os.environ.get("cluster_id")

        MONGO_URI = f"mongodb+srv://{username}:{password}@cluster0.jsbdmm9.mongodb.net/?appName={cluster_id}"

        vector_store_ = MongoDBAtlasVectorSearch.from_connection_string(
            connection_string=MONGO_URI,
            namespace="sample_mflix.pdf_embeddings",
            embedding=OpenAIEmbeddings(
                base_url="https://openrouter.ai/api/v1",
                model="perplexity/pplx-embed-v1-0.6b",
                api_key=SecretStr(OPENROUTER_API_KEY or ""),
                embedding_ctx_length=1024,
                check_embedding_ctx_length=False,
                model_kwargs={"encoding_format": "float"},
            ),
            index_name="document_embeddings",
        )
        return vector_store_

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2))
    def create_document_embeddings(self) -> dict[str, str]:\
        
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
                collection=self.collection,
                index_name="document_embeddings",
            )
            return  {"status": "success", "message": "Document Embeddings Created Successfully"}
        except ServerSelectionTimeoutError as e:
            return {
                "status": "error",
                "message": "Failed to Create Document Embeddings. ",
                "error": str(e),
            }
        except Exception as e:
            return {"status": "error",
                    "message": "Failed to create Document Embeddings", "error": str(e)}

    async def convert_document_to_vector(self) -> dict[str, str]:
        self._load_document()
        self.clean_document_data()
        self.chunk_and_split_document()
        print("Chunked Documents:", self.chunked_documents)
        await self.add_metadata_to_document()
        embeddings_status = dict()
        for _ in range(3):
            embeddings_status = self.create_document_embeddings()
            if embeddings_status["status"] == "sucess":
                break
        return embeddings_status
    
    @classmethod
    def get_similar_documents_from_database(cls, user_query: str) -> list[Document]:
        vector_store: MongoDBAtlasVectorSearch = DocumentIndexer.create_vector_store()
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3, "score_threshold": 0.70},
        )
        
        documents = retriever.invoke(user_query)
        return documents

# if __name__ == "__main__":
#     load_dotenv(PROJECT_ROOT / ".env", override=True)

#     username = os.environ.get("username")
#     password = os.environ.get("password")
#     cluster_id = os.environ.get("cluster_id")

#     MONGO_URI = (
#         f"mongodb+srv://{username}:{password}@cluster0.jsbdmm9.mongodb.net/?appName={cluster_id}"
#     )

#     mongo_vector_db = MongoVectorDB(
#         uri=MONGO_URI, db_name=os.environ.get("MONGO_DB", "sample_mflix")
#     )
#     document_indexer = DocumentIndexer(
#         file_path="Linkedin-Profile.pdf",
#         mongo_vector_db=mongo_vector_db
#     )
#     document_indexer.clean_document_data()
#     print("Cleaned document data....")
#     # document_indexer.preview_data_from_pdf()
#     document_indexer.chunk_and_split_document()
#     print("Chunked and split document data....")
    
#     print(document_indexer.chunked_documents)
    
    # document_indexer.add_metadata_to_document()
    # print("Added metadata to document data....")
    
    # for document in document_indexer.chunked_documents:
    #     print("Document metadata: ", document.metadata.keys())
    # print("Metadata: ", document_indexer.chunked_documents)
    # # document_indexer.create_document_embeddings()
    # # print("Created document embeddings....")
    
    # vector_store: MongoDBAtlasVectorSearch = DocumentIndexer.create_vector_store()
    # retriever = vector_store.as_retriever(
    #     search_type="similarity",
    #     search_kwargs={"k": 3, "score_threshold": 0.01},
    # )
    
    # documents = retriever.invoke("Who is Rishabh?")
    # for document in documents:
    #     print(document.page_content)