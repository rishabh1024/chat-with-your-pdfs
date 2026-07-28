import asyncio
import tempfile
from contextlib import asynccontextmanager
from typing import cast
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from mongo_vector_db.chat import ChatService
from mongo_vector_db.data_wrangler import DocumentIndexer
from mongo_vector_db.file_upload import FileUpload
from mongo_vector_db.main import MongoVectorDB
from mongo_vector_db.models import ChatResponseModel, FileUploadResponse, StorageUploadResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
  app.state.file_uploader = await FileUpload.create()
  app.state.mongodb_client = DocumentIndexer.get_mongodb_client()
  app.state.chat_service = ChatService()
  yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


REQUIRED_FILE = File(...)


def get_file_uploader(request: Request) -> FileUpload:
    return request.app.state.file_uploader
  
def get_mongodb_client(request: Request) -> MongoVectorDB:
    return request.app.state.mongodb_client

def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service

@app.post("/upload_and_index", response_model=FileUploadResponse)
async def upload_file(input_file : UploadFile = REQUIRED_FILE,
                      file_uploader : FileUpload = Depends(get_file_uploader),
                      monggodb_client : MongoVectorDB = 
                      Depends(get_mongodb_client)) -> FileUploadResponse:
    
    print(f"User has uploaded a file named {input_file.filename}")
    
    file_contents_of_uploaded_file = await input_file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        temp.write(file_contents_of_uploaded_file)
        temp_file_path = temp.name
      
    file_upload_and_index_response = await upload_and_index_pdf_file(input_file=input_file,
                                                              file_contents=file_contents_of_uploaded_file,
                                                              temp_file_path=temp_file_path,
                                                              file_uploader=file_uploader,
                                                              mongodb_client=monggodb_client)
                                                              
    
    return FileUploadResponse(
      document_id=file_upload_and_index_response.document_id,
      file_hash=file_upload_and_index_response.file_hash,
      upload_status=file_upload_and_index_response.upload_status,
      upload_error=file_upload_and_index_response.upload_error,
      document_indexing_status=file_upload_and_index_response.document_indexing_status
    )


async def upload_and_index_pdf_file(input_file: UploadFile,
                                    file_contents: bytes,
                                    temp_file_path: str,
                                    file_uploader: FileUpload,
                                    mongodb_client: MongoVectorDB
                                    ) -> FileUploadResponse:
    
    hash_of_file_contents = FileUpload.calculate_file_hash(file_contents)

    original_file_name_string = input_file.filename
    
    if not original_file_name_string:
      original_file_name_string = f"{uuid4().int}.pdf"
    
    document_indexer = DocumentIndexer(file_path=temp_file_path,mongo_vector_db=mongodb_client,
                                       document_id=hash_of_file_contents)
    
    file_uploading_task = asyncio.create_task(
                                    file_uploader.upload_to_file_storage(
                                        uploaded_file_content_in_bytes=file_contents,original_filename=original_file_name_string,
                                        file_hash=hash_of_file_contents
                                        )
                                    )
    
    document_indexing_task = asyncio.create_task(document_indexer.convert_document_to_vector())
    
    file_uploading_status, document_indexing_status = await asyncio.gather(file_uploading_task, 
                                                                           document_indexing_task)
    file_uploading_status = cast(StorageUploadResponse, file_uploading_status)
    return FileUploadResponse(
      document_id=file_uploading_status.document_id,
      file_hash=file_uploading_status.file_hash,
      upload_status=file_uploading_status.upload_status,
      upload_error=file_uploading_status.upload_error,
      document_indexing_status=document_indexing_status
    )

@app.post(path="/chat/conversation/{chat_id}/messages/")
def send_message(chat_id: UUID,
                 message_query: str,
                 chat_service: ChatService = Depends(get_chat_service)) -> ChatResponseModel:
    ai_message_response, chat_history_messages = chat_service.send_message(
        chat_id=chat_id, user_message=message_query
    )

    return ChatResponseModel(
        chat_id=chat_id, 
        ai_message=ai_message_response,
        chat_history_messages=chat_history_messages
    )
  