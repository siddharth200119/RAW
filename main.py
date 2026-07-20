import asyncio
import sys
from RAW.llms import VLLM
from RAW.models import Message, jsonschema, Tool, ToolParam
from RAW.utils import Logger


async def run_info(llm):
    print("\n--- Testing info() ---")
    try:
        info = await llm.info()
        print(f"Model Name: {info.model_name}")
        print(f"Provider: {info.provider}")
        print(f"Max Tokens: {info.max_tokens}")
        print(f"Context Window: {info.context_window}")
        print(f"Capabilities: {[c.name for c in info.capabilities]}")
        print(f"Metadata: {info.metadata}")
    except Exception as e:
        print(f"Error: {e}")


async def run_generate(llm):
    print("\n--- Testing generate() ---")
    prompt = input("Enter prompt for generate: ") or "Explain quantum computing in one sentence."
    stream_choice = input("Stream response? (y/n, default n): ").strip().lower() == "y"
    schema_choice = input("Enforce dummy JSON schema? (y/n, default n): ").strip().lower() == "y"

    schema = None
    if schema_choice:
        schema = jsonschema({
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "The main answer summary"},
                "keywords": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["summary", "keywords"]
        })

    try:
        if stream_choice:
            print("Response stream: ", end="", flush=True)
            generator = await llm.generate(prompt=prompt, schema=schema, stream=True)
            async for chunk in generator:
                print(chunk, end="", flush=True)
            print()
        else:
            response = await llm.generate(prompt=prompt, schema=schema, stream=False)
            print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")


async def run_chat(llm):
    print("\n--- Testing chat() ---")
    prompt = input("Enter chat user message: ") or "Hello, what tools do you have?"
    stream_choice = input("Stream response? (y/n, default n): ").strip().lower() == "y"
    tools_choice = input("Attach get_weather tool? (y/n, default n): ").strip().lower() == "y"

    messages = [Message(role="user", content=prompt)]
    
    tools = None
    if tools_choice:
        async def get_weather(location: str):
            return f"Weather in {location} is 22°C and rainy."

        tools = [
            Tool(
                name="get_weather",
                description="Get weather details for a location",
                parameters=[
                    ToolParam(name="location", type="string", description="The city name", required=True)
                ],
                function=get_weather
            )
        ]

    try:
        if stream_choice:
            print("Response stream: ")
            generator = await llm.chat(messages=messages, stream=True, tools=tools)
            async for chunk in generator:
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                if chunk.tool_calls:
                    print(f"\n[Tool Call] {chunk.tool_calls}")
            print()
        else:
            response = await llm.chat(messages=messages, stream=False, tools=tools)
            print(f"Response role: {response.role}")
            print(f"Response content: {response.content}")
            if response.tool_calls:
                print(f"Response tool calls: {response.tool_calls}")
    except Exception as e:
        print(f"Error: {e}")


async def run_count_tokens(llm):
    print("\n--- Testing count_tokens() ---")
    text = input("Enter text to count tokens: ") or "Hello world! This is a token counting test."
    try:
        count = await llm.count_tokens(text)
        print(f"Token count: {count}")
    except Exception as e:
        print(f"Error: {e}")


async def run_embed(llm):
    print("\n--- Testing embed() ---")
    text = input("Enter text to embed: ") or "Machine learning model"
    try:
        embedding = await llm.embed(text)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding snippet (first 5 values): {embedding[:5]}")
    except Exception as e:
        print(f"Error: {e}")


async def run_stop(llm):
    print("\n--- Testing stop() ---")
    try:
        await llm.stop()
        print("LLM connection stopped/closed successfully.")
    except Exception as e:
        print(f"Error: {e}")


async def main():
    logger = Logger("VLLMTest", level="INFO")
    
    print("Initializing VLLM with:")
    print("  Model: Qwen/Qwen3.5-9B")
    print("  Base URL: http://192.168.10.198:8002")
    
    llm = VLLM(
        model='Qwen/Qwen3.5-9B', 
        base_url="http://192.168.10.198:8002",
        logger=logger
    )

    menu = {
        "1": ("info()", run_info),
        "2": ("generate()", run_generate),
        "3": ("chat()", run_chat),
        "4": ("count_tokens()", run_count_tokens),
        "5": ("embed()", run_embed),
        "6": ("stop()", run_stop),
    }

    while True:
        print("\n==============================")
        print("VLLM Method Test Runner Menu")
        print("==============================")
        for key, val in menu.items():
            print(f" {key}. Run {val[0]}")
        print(" q. Exit Test Runner")
        
        choice = input("\nSelect a method to run: ").strip().lower()
        if choice == "q":
            print("Exiting.")
            # Ensure we close connections
            try:
                await llm.stop()
            except:
                pass
            break
        elif choice in menu:
            name, func = menu[choice]
            await func(llm)
        else:
            print("Invalid choice, please select again.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExited by user.")
