from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_redis import RedisChatMessageHistory
from langchain.chains import ConversationChain

from dotenv import load_dotenv
import yaml

import uuid

#load environment variables
load_dotenv()
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
    return_messages = True
)
prompt = ChatPromptTemplate([
    ('system', prompts['chatbot']['system']),
    ('human', '{history}'),
    ('human', prompts['chatbot']['human'])
])
conversation = ConversationChain(
    llm = llm,
    prompt = prompt,
    memory = memory,
    verbose = True
)

#create a chat
def chat_with_memory():
    print("Type in 'exit' or 'quit' to stop chatbot.")
    try:
        while True:     
            input_text = input('You: ')
            if input_text.lower() == 'exit' or input_text.lower() == 'quit': break
            #if input_text.lower() in ['exit', 'quit']: break
            response = conversation.invoke({'input': input_text})
            print(f'Dating Guru: {response['response']}')
    except KeyboardInterrupt: 
        print('Goodbye!')
    finally:
        history.redis_client.close()

chat_with_memory()