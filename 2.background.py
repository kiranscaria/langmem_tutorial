from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore

from langmem import create_memory_store_manager

from dotenv import load_dotenv

load_dotenv(override=True)


store = InMemoryStore(
    index = {
        "dims": 1536,
        "embed": "openai:text-embedding-3-small"
    }
)
llm = init_chat_model("groq:qwen-2.5-32b")

# Create memory manager Runnable to extract memories from conversations
memory_manager = create_memory_store_manager(
    "groq:qwen-2.5-32b",
    namespace=("memories",), # Store memories in the "memories" namespace (aka directory)
)

@entrypoint(store=store)  # Create a LangGraph workflow
async def chat(message: str):
    response = llm.invoke(message)

    # memory_manager extracts memories from conversation history
    to_process = {"messages": [{"role": "user", "content": message}] + [response]}
    await memory_manager.ainvoke(to_process)
    return response.content

# Run conversation as normal
async def main():
    response = await chat.ainvoke("I like dogs. My dogs' names are Sweety, Charlie & Rocky.")
    print(response)

    # To check the memories extracted. 
    print("--------------------MEMORIES EXTRACTED--------------------")
    print(store.search(("memories",)))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())