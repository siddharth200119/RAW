"""
Example usage of GeminiLLM from RAW library
"""
import asyncio
import os
from RAW.llms.gemini import GeminiLLM, GeminiOptions
from RAW.modals import Message, Tool, ToolCall, LLMCapability
from RAW.modals.tools import ToolParam


async def basic_generation_example():
    """Example: Basic text generation with Gemini"""
    print("\n=== Basic Generation Example ===")
    
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY environment variable to run this example")
        return
    
    # Initialize with custom options
    options = GeminiOptions(
        temperature=0.7,
        max_output_tokens=256
    )
    llm = GeminiLLM(api_key=api_key, options=options)
    
    try:
        # Generate a response
        response = await llm.generate("Explain what Python is in one sentence.")
        print(f"Response: {response}")
    finally:
        await llm.stop()


async def chat_example():
    """Example: Multi-turn chat conversation"""
    print("\n=== Chat Example ===")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY environment variable to run this example")
        return
    
    llm = GeminiLLM(api_key=api_key)
    
    try:
        # Create a conversation
        messages = [
            Message(role="user", content="What is the capital of France?")
        ]
        
        response = await llm.chat(messages)
        print(f"User: {messages[0].content}")
        print(f"Assistant: {response.content}")
        
        # Continue the conversation
        messages.append(response)
        messages.append(Message(role="user", content="What's a famous landmark there?"))
        
        response2 = await llm.chat(messages)
        print(f"User: {messages[-1].content}")
        print(f"Assistant: {response2.content}")
    finally:
        await llm.stop()


async def tool_calling_example():
    """Example: Using tools with Gemini"""
    print("\n=== Tool Calling Example ===")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY environment variable to run this example")
        return
    
    llm = GeminiLLM(api_key=api_key)
    
    # Define a tool
    def get_weather(city: str, unit: str = "celsius") -> str:
        """Mock weather function"""
        return f"The weather in {city} is 22 degrees {unit}"
    
    weather_tool = Tool(
        name="get_weather",
        description="Get the current weather for a city",
        parameters=[
            ToolParam(
                name="city",
                type="string",
                description="The city name",
                required=True
            ),
            ToolParam(
                name="unit",
                type="string",
                description="Temperature unit (celsius or fahrenheit)",
                required=False,
                enums=["celsius", "fahrenheit"]
            )
        ],
        function=get_weather
    )
    
    try:
        messages = [
            Message(role="user", content="What's the weather like in Tokyo?")
        ]
        
        response = await llm.chat(messages, tools=[weather_tool])
        print(f"User: {messages[0].content}")
        
        if response.tool_calls:
            print(f"Tool call: {response.tool_calls[0].name}")
            print(f"Arguments: {response.tool_calls[0].arguments}")
            
            # Execute the tool
            tool_result = get_weather(**response.tool_calls[0].arguments)
            print(f"Tool result: {tool_result}")
        else:
            print(f"Assistant: {response.content}")
    finally:
        await llm.stop()


def check_capabilities():
    """Example: Check LLM capabilities"""
    print("\n=== Checking Capabilities ===")
    
    llm = GeminiLLM(api_key="dummy-key")
    
    print(f"GeminiLLM capabilities:")
    for cap in llm.capabilities:
        print(f"  - {cap.name}")
    
    # Check specific capability
    if LLMCapability.TOOLS in llm.capabilities:
        print("\n✓ Tool calling is supported")
    if LLMCapability.VISION in llm.capabilities:
        print("✓ Vision (image) input is supported")


async def streaming_example():
    """Example: Streaming text generation"""
    print("\n=== Streaming Generation Example ===")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY environment variable to run this example")
        return
    
    llm = GeminiLLM(api_key=api_key)
    
    try:
        print("Response: ", end="", flush=True)
        # Generate a streaming response
        stream = await llm.generate("what is a LLM?", stream=True)
        
        async for chunk in stream:
            print(chunk, end="", flush=True)
        print()  # Newline after stream
    finally:
        await llm.stop()


def main():
    """Run all examples"""
    print("RAW Library - GeminiLLM Examples")
    print("=" * 40)
    
    # Synchronous capability check (no API call needed)
    check_capabilities()
    
    # Async examples (require API key)
    print("\n" + "=" * 40)
    print("Running async examples...")
    print("(Set GEMINI_API_KEY environment variable)")
    print("=" * 40)
    
    asyncio.run(basic_generation_example())
    asyncio.run(streaming_example())
    asyncio.run(chat_example())
    asyncio.run(tool_calling_example())


if __name__ == "__main__":
    main()
