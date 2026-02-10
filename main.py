import asyncio
import os
from RAW.agent import Agent
from RAW.llms import GroqLLM
from RAW.modals import Tool
from RAW.modals.tools import ToolParam
from RAW.utils import Logger

# Example tool
async def get_weather(location: str):
    """Get the weather for a location."""
    # Mock weather data
    return f"The weather in {location} is sunny with a temperature of 25°C."

async def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Please set GROQ_API_KEY environment variable.")
        return

    import logging
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger = Logger("Chatbot", level=log_level)
    
    # Initialize LLM
    model_name = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    llm = GroqLLM(api_key=api_key, logger=logger, model=model_name)

    # Initialize Tools
    weather_tool = Tool(
        name="get_weather",
        description="Get the weather for a specific location",
        parameters=[
            ToolParam(name="location", type="string", description="The city and state, e.g. San Francisco, CA", required=True)
        ],
        function=get_weather
    )
    
    # Initialize Agent
    agent = Agent(
        name="WeatherBot",
        base_prompt="You are a helpful weather assistant.",
        tools=[weather_tool],
        llm=llm,
        logger=logger
    )

    print("Chatbot started! Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        print("Bot: ", end="", flush=True)
        async for chunk in agent(user_input, stream=True):
            if isinstance(chunk, dict):
                if "content" in chunk:
                    content = chunk["content"]
                    if isinstance(content, dict): # Message object dumped
                         # Skip full message dumps, we want the stream chunks if possible or just final text
                         pass
                    elif isinstance(content, str): # Tool response or error
                         # print(content) # Maybe don't print tool output directly to user
                         pass
            else:
                 pass # String chunks ? Agent yield dicts.
                 
            # Agent yields:
            # {"agent_name": name, "content": chunk} where chunk is Message.dump() or str (tool response)
            # wait, agent.py yields:
            # 1. {"agent_name": self.name, 'content': chunk.model_dump()} (if Message)
            # 2. {"agent_name": self.name, 'content': chunk} (if str - tool response)
            
            # Streaming from LLM yields Message objects with partial content?
            # execute_stream yields Message objects constructed from chunks. 
            # Actually execute_stream yields:
            # 1. Message (partial content)
            # 2. Dict (tool call)
            # 3. Dict (tool response)
            
            # Agent.__call__ wraps these:
            # If chunk is Message -> yields dict with content=chunk.dump()
            # If chunk is other -> yields dict with content=chunk
            
            # So if we want to print the token stream:
            # The Agent logic in execute_stream accumulates content and yields a Message for each chunk?
            # No.
            # `yield Message(role="assistant", content=response.content ...)`
            # So it yields a Message object for EACH chunk of text.
            
            if isinstance(chunk, dict) and "content" in chunk:
                content_obj = chunk["content"]
                if isinstance(content_obj, dict) and "content" in content_obj and content_obj["content"]:
                     # It's a message dump
                     print(content_obj["content"], end="", flush=True)

        print() # Newline after response

if __name__ == "__main__":
    asyncio.run(main())
