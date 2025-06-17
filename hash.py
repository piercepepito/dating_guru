from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_redis import RedisChatMessageHistory
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv
import yaml

import uuid
#Load and Split the documents
from langchain_community.document_loaders import TextLoader, UnstructuredWordDocumentLoader, PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import glob

import hashlib
import json


#load environment variables
load_dotenv()

TRACKING_FILE = "processed_files.json"
DATA_DIR = "files"
VECTORSTORE_DIR = "faiss_index"

#track processed files
def get_file_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def load_processed_files():
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_processed_files(processed):
    with open(TRACKING_FILE, 'w') as f:
        json.dump(processed, f, indent=2)

def get_new_or_modified_files(directory, processed_files):
    allowed_extensions = {'.pdf', '.docx', '.txt'}
    new_or_modified = []

    for root, _, files in os.walk(directory):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in allowed_extensions:
                continue  # Skip unsupported file types

            path = os.path.join(root, name)
            file_hash = get_file_hash(path)

            if path not in processed_files or processed_files[path] != file_hash:
                new_or_modified.append((path, file_hash))

    return new_or_modified


def get_file_loader(filename):
    ext = filename.split(".")[-1]
    if ext == 'pdf':
        return UnstructuredPDFLoader(filename).load()
    if ext == 'txt':
        return TextLoader(filename).load()
    if ext == 'docx':
        return UnstructuredWordDocumentLoader(filename).load()
    else: return None

def your_splitter(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)
'''
# Load existing vector store or create new one
embedding = OpenAIEmbeddings()
if os.path.exists(VECTORSTORE_DIR):
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embedding,
        allow_dangerous_deserialization=True
    )
else:
    vectorstore = FAISS.from_documents([], embedding)
'''
# Main sync function
def sync_new_files_to_faiss(data_dir):
    processed_files = load_processed_files()
    new_files = get_new_or_modified_files(data_dir, processed_files)

    if not new_files:
        print("No new or modified files found.")
        return

    embedding = OpenAIEmbeddings()
    if os.path.exists(VECTORSTORE_DIR):
        vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embedding,
        allow_dangerous_deserialization=True
        )
    else:
        vectorstore = None


    for path, file_hash in new_files:
        try:
            docs = get_file_loader(path)
            split_docs = your_splitter(docs)

            if vectorstore is None:
                # First time creation with initial docs
                vectorstore = FAISS.from_documents(split_docs, embedding)
            else:
                vectorstore.add_documents(split_docs)

            #vectorstore.add_documents(split_docs)
            processed_files[path] = file_hash
            print(f"✅ Processed and added: {path}")
        except Exception as e:
            print(f"❌ Error processing {path}: {e}")

    # Save updated state
    save_processed_files(processed_files)

    # Persist FAISS
    vectorstore.save_local(VECTORSTORE_DIR)

print('Syncing files')

sync_new_files_to_faiss(DATA_DIR)

#Create the retriever
embedding = OpenAIEmbeddings()
if os.path.exists(VECTORSTORE_DIR):
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embedding,
        allow_dangerous_deserialization=True
    )
else:
    print('Vector Store not found')

retriever = vectorstore.as_retriever()

#load yaml file
with open('utils/prompts.yaml', 'r') as file:
    prompts = yaml.safe_load(file)

#instantiate llm
llm = ChatOpenAI(temperature = 0.7)

#create a chatbot
session_id = str(uuid.uuid4())
history = RedisChatMessageHistory(
    session_id = session_id,
    redis_url="redis://localhost:6379"
)

memory = ConversationBufferMemory(
    chat_history = history,
    return_messages = True,
    memory_key ='chat_history'
)

dating_prompt = PromptTemplate.from_template(prompts['chatbot']['system'])

conversation = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever, memory=memory, condense_question_prompt = dating_prompt
)

#create a chat
def chat_with_memory():
    print("Type in 'exit' or 'quit' to stop chatbot.")
    try:
        while True:     
            input_text = input('You: ')
            if input_text.lower() == 'exit' or input_text.lower() == 'quit': break
            response = conversation.invoke({'question': input_text})
            print(f'Dating Guru: {response['answer']}')
    except KeyboardInterrupt: 
        print('Goodbye!')
    finally:
        history.redis_client.close()
        print('Goodbye')

chat_with_memory()