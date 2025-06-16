from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from dotenv import load_dotenv
import yaml

#load environment variables
load_dotenv()
#load yaml file
with open('utils/prompts.yaml', 'r') as file:
    prompts = yaml.safe_load(file)

#instantiate llm
llm = ChatOpenAI(temperature = 0.7)

#create a chatbot
memory = ConversationBufferMemory()
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
    while True:     
        input_text = input('You: ')
        if input_text.lower() in ['exit', 'quit']: break
        response = conversation.invoke({'input': input_text})
        print(f'Dating Guru: {response['response']}')

chat_with_memory()