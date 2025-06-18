import os

import json
import hashlib

from langchain_community.document_loaders import TextLoader, UnstructuredWordDocumentLoader, PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import uuid

from langchain.memory import ConversationBufferMemory
from langchain_redis import RedisChatMessageHistory

#track processed files
def _get_file_hash(filepath):
    '''
    Converts file into SHA256 hash and turns it into a hexadecimal string

    Args:
        filepath: filepath to add in the hash files
    '''
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def _load_processed_files(TRACKING_FILE):
    '''
    Loads all the files that have been turned into their own embeddings

    Args:
        TRACKING_FILE: The filepath to place in all the processed embeddings
    '''
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, 'r') as f:
            return json.load(f)
    else : return {}

def _save_processed_files(processed):
    '''
    Saves the processed embedding files

    Args:
        processed: all the hashed files that are finished processing
    '''
    with open(TRACKING_FILE, 'w') as f:
        json.dump(processed, f, indent=2)

def _get_new_or_modified_files(directory, processed_files):
    '''
    Finds new or modified files
    '''
    allowed_extensions = {'.pdf', '.docx', '.txt'}
    new_or_modified = []

    for root, _, files in os.walk(directory):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in allowed_extensions:
                continue  # Skip unsupported file types

            path = os.path.join(root, name)
            file_hash = _get_file_hash(path)

            if path not in processed_files or processed_files[path] != file_hash:
                new_or_modified.append((path, file_hash))

    return new_or_modified


def _get_file_loader(filename):
    '''
    Loads pdf, txt, and docx files to be converted into an embdedding

    Args:
        filename: pdf, txt, files to be loaded
    '''
    ext = filename.split(".")[-1]
    if ext == 'pdf':
        return UnstructuredPDFLoader(filename).load()
    if ext == 'txt':
        return TextLoader(filename).load()
    if ext == 'docx':
        return UnstructuredWordDocumentLoader(filename).load()
    else: return None

def _your_splitter(docs):
    '''
    Splits loaded files into sizable chunks

    Args:
        docs: strings loaded from get_file_loader function
    '''
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)

# Main sync function
def sync_new_files_to_faiss(data_dir, TRACKING_FILE, VECTORSTORE_DIR):
    '''
    Syncing the unprocessed files and processing them to create new embedings

    Args:
        data_dir: the directory where the files for RAG are being added
        TRACKING_FILE: .json directory where all processed file are being placed 
        VECTORESTORE_DIR: directory where the vectorestore is located
    '''
    processed_files = _load_processed_files(TRACKING_FILE)
    new_files = _get_new_or_modified_files(data_dir, processed_files)

    if not new_files:
        print("No new or modified files found.")
        return

    embedding = OpenAIEmbeddings()
    if os.path.exists(VECTORSTORE_DIR):
        vectorstore = FAISS.load_local( VECTORSTORE_DIR,
                                        embedding,
                                        allow_dangerous_deserialization=True
                                        )
    else:
        vectorstore = None


    for path, file_hash in new_files:
        try:
            docs = _get_file_loader(path)
            split_docs = _your_splitter(docs)

            if vectorstore is None:
                # First time creation with initial docs
                vectorstore = FAISS.from_documents(split_docs, embedding)
            else:
                vectorstore.add_documents(split_docs)

            #vectorstore.add_documents(split_docs)
            processed_files[path] = file_hash
            print(f"Processed and added: {path}")
        except Exception as e:
            print(f"Error processing {path}: {e}")

    # Save updated state
    _save_processed_files(processed_files)

    # Persist FAISS
    vectorstore.save_local(VECTORSTORE_DIR)

def redis_session(redis_url_use):
    '''
    Creates a new redis session. Redis is used mainly for scalibity of having multiple users use a chatbot

    Args:
        redis_url_use: the url where redis is being stored
    '''
    session_id = str(uuid.uuid4())

    history = RedisChatMessageHistory(
        session_id = session_id,
        redis_url= redis_url_use
    )
    return history

def create_chatbot_memory(history):
    '''
    Allows the chatbot to remember the old chat history

    Args:
        history: the redis output from redis_session function
    '''
    memory = ConversationBufferMemory(
        chat_history = history,
        return_messages = True,
        memory_key ='chat_history'
    )
    return memory

def chat_with_memory(chatbot, history):
    '''
    Creates the chatbot flow

    Args:
        chatbot: the LLM added with memory, and RAG
        history: the redis history
    '''
    print("Type in 'exit' or 'quit' to stop chatbot.")
    try:
        while True:     
            input_text = input('You: ')
            if input_text.lower() == 'exit' or input_text.lower() == 'quit': break
            response = chatbot.invoke({'question': input_text})
            print(f'Dating Guru: {response['answer']}')
    except KeyboardInterrupt: 
        print('Goodbye!')
    finally:
        history.redis_client.close()
        print('Goodbye')
