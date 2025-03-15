import time
from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore
from langmem import ReflectionExecutor, create_memory_store_manager

from dotenv import load_dotenv

load_dotenv(override=True)

store = InMemoryStore(
    index = {
        "dims": 1536,
        "embed": "openai:text-embedding-3-small"
    }
)
llm = init_chat_model("groq:qwen-2.5-32b")

# Create memory manager to extract memories from conversations
memory_menager = create_memory_store_manager(
    "groq:qwen-2.5-32b",
    namespace=("memories",)
)

# Wrap memory_manager to handle deferred background processing
executor = ReflectionExecutor(memory_menager)
store = InMemoryStore(
    index = {
        "dims": 1536,
        "embed": "openai:text-embedding-3-small"
    }
)

@entrypoint(store=store)
def chat(message: str):
    response = llm.invoke(message)
    # Format conversation for memory processing
    # Must follow OpenAI's message format
    to_process = {"messages": [{"role": "user", "content": message}] + [response]}

    # Wait for 30 minutes before processing
    # If new messages arrive before then:
    # 1. Cancel pending processing task
    # 2. Reschedule with new messages included
    delay = 3 # in seconds
    executor.submit(to_process, after_seconds=delay)
    return response.content

# Run conversation as normal
def main():
    response = chat.invoke("I like dogs. My dogs' names are Sweety, Charlie & Rocky.")
    print(response)

    # To check the memories extracted. 
    interval = 3
    print("--------------------MEMORIES EXTRACTED--------------------")
    for i in range(0, 10, interval):
        print(f"After {i} seconds...")
        print(store.search(("memories",)))
        time.sleep(interval)

if __name__ == "__main__":
    main()