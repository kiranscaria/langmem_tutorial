from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.types import Checkpointer
from langgraph.utils.config import get_store
from langmem import create_manage_memory_tool # Lets agent create, update and delete memories
from dotenv import load_dotenv

load_dotenv(override=True)


def prompt(state):
    """Prepare the messages for the LLM."""
    # Get store from configured contextvar;
    store = get_store()
    memories = store.search(
        # Search within the same namespace as the one we've configured for the agent
        ("memories",),
        query=state["messages"][-1].content
    )
    system_msg = f"""You're a helpful assistant. 

    ## memories
    <memories>
    {memories}
    </memories>
    """
    return [{"role": "system", "content": system_msg}, *state["messages"]] 

store = InMemoryStore(
    index={ # Store extracted memories
        "dims": 1536,
        "embed": "openai:text-embedding-3-small"
    }
)
checkpointer = MemorySaver() # Checkpoint graph state

agent = create_react_agent(
    "groq:qwen-2.5-32b",
    prompt=prompt,
    tools = [ # Add memory tools
        # The agent can call "manage memory" to
        # create, update, and delete memories by ID
        # Namesppaces add scope to memories. To
        # scope memories per-user, do ("memories", "{user_id}"):
        create_manage_memory_tool(namespace=("memories",)),
    ],
    # Our memories will be stored in this provided BaseStore instance
    store=store,
    # Add the graph "state" will be checkpointed after each node
    # completes executing for tracking the chat history and durable execution
    checkpointer=checkpointer
)

# Using the agent
config = {"configurable": {"thread_id": "thread-a"}}

# Use the agent. The agent hasn't saved any memories, so it doesn't know about us
response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Know which display mode I prefer?"}
        ]
    },
    config=config
)
print(response["messages"][-1].content)

agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "dark, Remember that."}
        ]
    },
    # We will continue the conversation (thread-a) by using the config with the same thread_id
    config=config
)

# New thread = new conversation!
new_config = {"configurable": {"thread_id": "thread-b"}}
# The agent will only be able to recall
# whatever it explicitly saved using the manage_memories tool
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Hey there. Do you remeber me? What are my preferences?"}]},
    config=new_config
)

print(response["messages"][-1].content)