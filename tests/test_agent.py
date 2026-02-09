import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from RAW.agent import Agent
from RAW.modals import LLMCapability, Message, Role, Tool, ToolParam, ToolCall
from RAW.llms.base import BaseLLM

# Mock LLM
class MockLLM(BaseLLM):
    def __init__(self):
        self.capabilities = [LLMCapability.TOOLS]
        self.mock_chat = AsyncMock()

    async def chat(self, messages, schema=None, stream=False, tools=[]):
        return await self.mock_chat(messages, schema, stream, tools)

    async def embed(self, text: str):
        return [0.1, 0.2]

    async def stop(self):
        pass
        
    async def generate(self, prompt: str, schema=None, stream=False, tools=[]):
        pass

@pytest.fixture
def mock_llm():
    return MockLLM()

@pytest.fixture
def agent(mock_llm):
    return Agent(name="TestAgent", llm=mock_llm)

@pytest.mark.asyncio
async def test_agent_initialization(agent):
    assert agent.name == "TestAgent"
    assert len(agent.messages) == 1
    assert agent.messages[0].role == Role.SYSTEM

@pytest.mark.asyncio
async def test_agent_call_stream(agent, mock_llm):
    # Setup mock response
    mock_response = Message(role=Role.ASSISTANT, content="Hello")
    
    # Setup async generator for chat
    async def mock_chat_gen(*args, **kwargs):
        yield mock_response

    mock_llm.mock_chat.return_value = mock_chat_gen()

    # Call agent
    responses = []
    async for chunk in agent("Hi", stream=True):
        responses.append(chunk)

    # Verify
    assert len(responses) == 1
    assert responses[0]["content"]["content"] == "Hello"
    assert len(agent.messages) == 3 # System + User + Assistant

@pytest.mark.asyncio
async def test_agent_tool_call(agent, mock_llm):
    # Define tool
    async def multiply(a: int, b: int):
        return a * b

    tool = Tool(
        name="multiply",
        description="Multiply two numbers",
        parameters=[
            ToolParam(name="a", type="integer", required=True),
            ToolParam(name="b", type="integer", required=True)
        ],
        function=multiply
    )
    agent.tools = [tool]

    # Mock tool call response from LLM
    tool_call_msg = Message(
        role=Role.ASSISTANT,
        content="",
        tool_calls=[ToolCall(name="multiply", arguments={"a": 2, "b": 3}, id="call_1")]
    )
    
    # Mock chat generator to return tool call then final answer
    async def mock_chat_gen_1(*args, **kwargs):
        yield tool_call_msg

    # Mock second call (after tool execution)
    final_response = Message(role=Role.ASSISTANT, content="6")
    async def mock_chat_gen_2(*args, **kwargs):
        yield final_response

    # Configure mock LLM side effects
    mock_llm.mock_chat.side_effect = [mock_chat_gen_1(), mock_chat_gen_2()]

    # Run agent
    responses = []
    async for chunk in agent("Multiply 2 and 3", stream=True):
        responses.append(chunk)

    # Verify interaction
    # 1. Tool call msg from LLM
    # 2. Tool execution result from Agent
    # 3. Final response from LLM (technically in loop logic we might see tool response)
    
    # Check messages history for tool execution
    tool_msgs = [m for m in agent.messages if m.role == Role.TOOL]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content == "6"

@pytest.mark.asyncio
async def test_agent_no_stream(agent, mock_llm):
    mock_response = Message(role=Role.ASSISTANT, content="Hello")
    mock_llm.mock_chat.return_value = mock_response # Return directly, not generator

    responses = []
    async for chunk in agent("Hi", stream=False):
        responses.append(chunk)

    assert len(responses) == 1
    assert responses[0] == "Hello"
