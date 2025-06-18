from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv
import yaml
import os

from utils import tools

#load environment variables
load_dotenv()

TRACKING_FILE = "processed_files.json"
DATA_DIR = "files"
VECTORSTORE_DIR = "faiss_index"

print('Finding and Syncing new files')
tools.sync_new_files_to_faiss(DATA_DIR, TRACKING_FILE, VECTORSTORE_DIR)

#Create the retriever
embedding = OpenAIEmbeddings()
if os.path.exists(VECTORSTORE_DIR):
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embedding,
        allow_dangerous_deserialization=True
    )
else:
    print('Vectorstore not found')

retriever = vectorstore.as_retriever()

#load yaml file
with open('utils/prompts.yaml', 'r') as file:
    prompts = yaml.safe_load(file)

#instantiate llm
llm = ChatOpenAI(temperature = 0.7)

#create a chatbot
history = tools.redis_session("redis://localhost:6379")
memory = tools.create_chatbot_memory(history)

dating_prompt = PromptTemplate.from_template(prompts['chatbot']['human'])

conversation = ConversationalRetrievalChain.from_llm(llm=llm, 
                                                     retriever=retriever, 
                                                     memory=memory, 
                                                     condense_question_prompt = dating_prompt)

#create a chat
tools.chat_with_memory(conversation, history)