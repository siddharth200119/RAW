from typing import List, Optional, AsyncGenerator, Union, Dict, Any
from RAW.models import Tool, LLMCapability, Message, File, FileType, Image
from RAW.llms.base import BaseLLM
from RAW.utils import Logger, logger, Router
import inspect
import json

class Agent:
    def __init__(
            self,
            name: str, 
            base_prompt: str = "", 
            tools: List[Tool] = [], 
            llm: BaseLLM = None,
            logger: Logger = logger,
            history: List[Message] = [],
            router: Optional[Router] = None
        ):
        self.logger = logger
        self.name = name
        self.logger.info(f"Initializing agent: {name}")
        self.llm = llm
        self.router = router
        
        if not self.llm:
            raise ValueError("Agent must be initialized with an LLM.")

        if LLMCapability.TOOLS not in self.llm.capabilities:
            self.logger.warning(f"The LLM assigned to the agent does not have tool usage capabilities. Tools will not be used.") 
        
        self.system_prompt = f"""
        You are an AI agent named {self.name}. Your role is as follows:
        {base_prompt}
        """
        self.messages: List[Message] = [
            Message(role="system", content=self.system_prompt),
            *history
        ]
        self.tools = tools
        self.available_tools = self.tools

    async def __call__(self, user_message: str, user_files: List[File], stream: bool = False, user_summary: Optional[str] = "") -> AsyncGenerator[Union[Dict[str, Any], str], None]:
        self.logger.info(f'USER MESSAGE: {user_message}')
        user_images = List[Image]
        #logic to process files
        for file in user_files:
            if file.file_type == FileType.IMAGE:
                if LLMCapability.VISION in self.llm.capabilities:
                    image = Image.from_file(file.download())
                    user_images.append(image)
                else:
                    # if the llm does not have capabilities.vision then use document parser.
                    pass
            else:
                #Simply use document parser from RAW.utils to extract the text and pass it in the user Message
                pass
        
        if not user_images:
            user_message = Message(role="user", content=user_message)
        else:
            user_message = Message(role="user", content=user_message, images=user_images)
        self.messages.append(user_message)

        if self.router:
            routing_result = await self.router.route(
                user_message=user_message,
                conversation_summary="",
                user_summary=user_summary,
                available_tools=self.tools
            )
            self.available_tools = [
                tool for tool in self.tools 
                if tool.name in routing_result['selected_tools']
            ]
            self.logger.info(f"Router selection: {routing_result['selected_tools']}")

        if stream:
            async for chunk in self.execute_stream():
                if isinstance(chunk, Message):
                    yield {"agent_name": self.name, 'content': chunk.model_dump()}
                else:
                    yield {"agent_name": self.name, 'content': chunk}
        else:
            async for chunk in self.execute_no_stream():
                if isinstance(chunk, Message):
                    yield {"agent_name": self.name, 'content': chunk.model_dump()}
                else:
                    yield {"agent_name": self.name, 'content': chunk}

    async def execute_stream(self) -> AsyncGenerator[Union[Message, Dict[str, Any]], None]:
        while True:
            chat_response_generator = await self.llm.chat(
                messages=self.messages, schema=None, stream=True, tools=self.available_tools
            )

            accumulated_content = ""
            final_message = None
            has_tool_calls = False

            async for response in chat_response_generator:
                if hasattr(response, "tool_calls") and response.tool_calls:
                    has_tool_calls = True
                    final_message = response
                elif response.content:
                    accumulated_content += response.content
                    yield Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=[],
                        images=[]
                    )

            if has_tool_calls and final_message:
                complete_message = final_message
                if accumulated_content:
                    complete_message.content = accumulated_content

                self.messages.append(complete_message)
                self.logger.info(
                    f"{self.name}: {complete_message.content}"
                )

                for tool_call in complete_message.tool_calls:
                    yield {
                        "tool_call": tool_call.name,
                        "arguments": tool_call.arguments
                    }
                    tool_func = next((t.function for t in self.tools if t.name == tool_call.name), None)

                    if tool_func:
                        try:
                            result = await self._execute_tool(tool_func, tool_call.arguments)
                            
                            tool_message = Message(
                                role="tool",
                                content=str(result),
                                tool_call_id=tool_call.id
                            )
                            self.messages.append(tool_message)
                            self.logger.info(
                                f"Tool executed successfully: {result}"
                            )
                            yield {"tool_response": tool_message.content}

                        except Exception as e:
                            error_msg = f"Error calling tool: {e}"
                            error_message = Message(
                                role="tool",
                                content=error_msg,
                                tool_call_id=tool_call.id
                            )
                            self.messages.append(error_message)
                            self.logger.error(
                                f"Error executing tool {tool_call.name}: {e}"
                            )
                            yield {"tool_response": error_msg}
                continue

            elif accumulated_content:
                complete_message = Message(
                    role="assistant",
                    content=accumulated_content,
                    tool_calls=[],
                    images=[]
                )
                self.messages.append(complete_message)
                self.logger.info(
                    f"{self.name}: {complete_message.content}"
                )
                return

            # If no tool calls and no content, break loop
            self.logger.warning(f"Agent {self.name} received empty response from LLM. Stopping.")
            return
                     
    async def execute_no_stream(self) -> AsyncGenerator[Union[Message, Dict[str, Any]], None]:
        while True:
            response: Message = await self.llm.chat(messages=self.messages, schema=None, stream=False, tools=self.available_tools)
            self.messages.append(response)
            self.logger.debug(
                f"Agent {self.name} generated response: {response.content}"
            )

            if response.tool_calls:
                for tool_call in response.tool_calls:
                    yield {
                        "tool_call": tool_call.name,
                        "arguments": tool_call.arguments
                    }
                    self.logger.info(
                        msg=f"Tool Called: {tool_call.name} with arguments {tool_call.arguments}"
                    )
                    tool_func = next((t.function for t in self.tools if t.name == tool_call.name), None)
                    if tool_func:
                        try:
                            result = await self._execute_tool(tool_func, tool_call.arguments)
                            
                            tool_message = Message(
                                role="tool",
                                content=str(result),
                                tool_call_id=tool_call.id
                            )
                            self.messages.append(tool_message)
                            self.logger.info(
                                f"Tool executed successfully: {result}"
                            )
                            yield {"tool_response": tool_message.content}

                        except Exception as e:
                            error_msg = f"Error calling tool: {e}"
                            error_message = Message(
                                role="tool",
                                content=error_msg,
                                tool_call_id=tool_call.id
                            )
                            self.messages.append(error_message)
                            self.logger.error(
                                f"Error executing tool {tool_call.name}: {e}"
                            )
                            yield {"tool_response": error_msg}
                continue
            if response.content:
                self.logger.info(
                    msg=f"{self.name}: {response.content}"
                )
                yield response.content
                return
            
            # If no tool calls and no content, break loop
            self.logger.warning(f"Agent {self.name} received empty response from LLM. Stopping.")
            return

    async def _execute_tool(self, tool_func, arguments):
        """Helper to execute tool functions with proper self/agent argument handling."""
        action_params = inspect.signature(tool_func).parameters
        pass_self = "agent" in action_params
        
        args = arguments
        kwargs = {"agent": self} if pass_self else {}
        
        if inspect.iscoroutinefunction(tool_func):
            return await tool_func(**args, **kwargs)
        elif inspect.isasyncgenfunction(tool_func):
            # Collect all items from async generator
            results = []
            async for res in tool_func(**args, **kwargs):
                results.append(res)
            return "\n".join(str(r) for r in results)
        elif inspect.isgeneratorfunction(tool_func):
            # Collect all items from generator
            results = []
            for res in tool_func(**args, **kwargs):
                results.append(res)
            return "\n".join(str(r) for r in results)
        else:
            return tool_func(**args, **kwargs)
