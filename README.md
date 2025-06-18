# Dating Guru: Your AI Wingman

With Tinder, Bumble and Hinge, making dating more accessible, there is now a need to have a dating coach. Someone that would help someone out on their love life. A DATING GURU to help out the single people in the world.

## Features
- AI chatbot powered by OpenAI (ChatGPT) for dating advice
- Uses FAISS and Retrieval-Augmented Generation (RAG) for context-aware answers
- Fast and scalable with Redis for short-term message storage

## Tech Stack
- **Redis** – In-memory data store for chat message caching
- **OpenAI GPT** – Natural language generation (LLM)
- **FAISS** – Vector search and retrieval (RAG)
- **Python** – Backend logic


## Getting Started:
1. Clone the repo
2. Install the dependencies
3. Install [Redis](https://redis.io/docs/latest/operate/oss_and_stack/install/install-stack/homebrew/)
4. Create an OpenAI account and top up for the LLM API KEY
5. Setup Environment
    1. In the root folder, create a .env file with OPENAI_API_KEY
    2. Add in the LLM API KEY
6. Open config.ini.example file
    1. Add in the redis URL. If you are using redis locally use this: `redis://localhost:6379`
    2. Change the filename to config.ini
4. Run redis in your terminal
    1. In your terminal type in: `redis-cli`
5. In your terminal, run `python main.py`


## Future Improvements
- PostgreSQL integration for persistent chat logs and behavior tracking
- Web front-end (React or Flask) to access the bot via browser