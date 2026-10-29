from typing import List, Optional, AsyncGenerator, Union, Dict, Any, Callable, Awaitable
from RAW.models import Tool, ToolCall, LLMCapability, Message, File, FileType, Image, Skill
from RAW.llms.base import BaseLLM
from RAW.utils import Logger, Router
import inspect
import uuid
import jsonschema as js_lib

class Agent:
    def __init__(
            self,
            name: str,
            base_prompt: str = "",
            tools: List[Tool] = [],
            llm: BaseLLM = None,
            logger: Optional[Logger] = None,
            history: List[Message] = [],
            router: Optional[Router] = None,
            skills: List[Skill] = [],
            on_message: Optional[Callable[[Message], Union[None, Awaitable[None]]]] = None,
        ):
        self.logger = logger or Logger(service_name=f"Agent:{name}")
        self.name = name
        self.logger.info(f"Initializing agent: {name}")
        self.llm = llm
        self.router = router
        # Called (sync or async) with every message added to the conversation,
        # in order, so callers can persist history exactly as the loop built it.
        # Errors raised by the hook propagate to the caller.
        self.on_message = on_message
        
        if not self.llm:
            raise ValueError("Agent must be initialized with an LLM.")

        if LLMCapability.TOOLS not in self.llm.capabilities:
            self.logger.warning(f"The LLM assigned to the agent does not have tool usage capabilities. Tools will not be used.") 
        
        self.base_prompt = base_prompt

        self.tools: List[Tool] = tools
        self.skills: List[Skill] = skills
        # Without a router, default to every skill and every tool (agent + skill tools).
        if self.router is None:
            self.available_skills = list(self.skills)
            self.available_tools = self._merge_tools(
                self.tools, self._tools_from_skills(self.skills)
            )
        else:
            self.available_skills = []
            self.available_tools = list(self.tools)

        self.conversation_summary: str = ""
        
        self.system_prompt =  self._build_system_prompt() 

        self.messages: List[Message] = [
            self.system_prompt,
            *history
        ]

        self.logger.info(
            f"""
            Agent {self.name} initialized with:
            - Tools: {[t.name for t in self.tools]}
            - Skills: {[s.name for s in self.skills]}
            """
        )

    async def __call__(
            self,
            user_message: str,
            user_files: List[File] = [],
            stream: bool = False,
            user_summary: Optional[str] = "",
            skill: Optional[Skill] = None,
        ) -> AsyncGenerator[Union[Dict[str, Any], str], None]:
        self.logger.info(f'USER MESSAGE: {user_message}')
        user_images: List[Image] = []
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
        await self._append(user_message)

        if skill is not None:
            self.available_skills = [skill]
            self.available_tools = self._merge_tools(skill.tools)
            self.logger.info(
                f"Skill override: {skill.name} "
                f"(tools: {[t.name for t in self.available_tools]})"
            )
        elif self.router:
            routing_result = await self.router.route(
                user_message=user_message,
                conversation_summary=self.conversation_summary,
                user_summary=user_summary,
                available_tools=self.tools,
                available_skills=self.skills
            )
            self.available_skills = [
                s for s in self.skills
                if s.name in routing_result['selected_skills']
            ]
            selected_tools = [
                tool for tool in self.tools
                if tool.name in routing_result['selected_tools']
            ]
            # Selected skills automatically bring their associated tools.
            self.available_tools = self._merge_tools(
                selected_tools, self._tools_from_skills(self.available_skills)
            )
            self.conversation_summary = routing_result.get(
                "conversation_summary", self.conversation_summary
            )
            self.logger.info(
                f"Router selection: tools={[t.name for t in self.available_tools]}, "
                f"skills={[s.name for s in self.available_skills]}"
            )
            self.logger.debug(f"Updated conversation summary: {self.conversation_summary}")
        else:
            self.available_skills = list(self.skills)
            self.available_tools = self._merge_tools(
                self.tools, self._tools_from_skills(self.skills)
            )
            self.logger.debug(
                f"No router: using all skills {[s.name for s in self.available_skills]} "
                f"and tools {[t.name for t in self.available_tools]}"
            )

        self.system_prompt = self._build_system_prompt()
        self.messages[0] = self.system_prompt
        if self.available_skills:
            self.logger.debug(
                f"System prompt updated with these skills: "
                f"{', '.join(s.name for s in self.available_skills)}"
            )
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

                await self._append(complete_message)
                self.logger.info(
                    f"{self.name}: {complete_message.content}"
                )

                async for event in self._run_tool_calls(complete_message.tool_calls):
                    yield event
                continue

            elif accumulated_content:
                complete_message = Message(
                    role="assistant",
                    content=accumulated_content,
                    tool_calls=[],
                    images=[]
                )
                await self._append(complete_message)
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
            await self._append(response)
            self.logger.debug(
                f"Agent {self.name} generated response: {response.content}"
            )

            if response.tool_calls:
                async for event in self._run_tool_calls(response.tool_calls):
                    yield event
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

    async def _append(self, message: Message) -> None:
        # Tool results reference their call by id, and streamed tool calls can
        # arrive without one, so every call gets an id before it is stored.
        for tool_call in message.tool_calls:
            if not tool_call.id:
                tool_call.id = f"call_{uuid.uuid4().hex[:24]}"
        self.messages.append(message)
        if self.on_message is not None:
            result = self.on_message(message)
            if inspect.isawaitable(result):
                await result

    async def _run_tool_calls(self, tool_calls: List[ToolCall]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run each tool call and append exactly one tool message per call, so the
        next LLM call never sees a tool call without a result.
        """
        for tool_call in tool_calls:
            yield {
                "tool_call": tool_call.name,
                "arguments": tool_call.arguments
            }
            self.logger.info(
                msg=f"Tool Called: {tool_call.name} with arguments {tool_call.arguments}"
            )
            tool = self._lookup_tool(tool_call.name)
            content = self._check_tool_call(tool, tool_call)
            if content is not None:
                self.logger.warning(f"Rejected tool call {tool_call.name}: {content}")
            else:
                try:
                    result = None
                    async for kind, payload in self._execute_tool(
                        tool.function, tool_call.arguments
                    ):
                        if kind == "stream":
                            # Intermediate tool yields (e.g. full table data) → client
                            yield {"tool_yield": payload}
                        else:
                            result = payload
                    content = str(result)
                    self.logger.info(
                        f"Tool executed successfully: {result}"
                    )
                except Exception as e:
                    content = f"Error calling tool: {e}"
                    self.logger.error(
                        f"Error executing tool {tool_call.name}: {e}"
                    )

            tool_message = Message(
                role="tool",
                content=content,
                tool_call_id=tool_call.id
            )
            await self._append(tool_message)
            yield {"tool_response": tool_message.content}

    def _check_tool_call(self, tool: Optional[Tool], tool_call: ToolCall) -> Optional[str]:
        """Return an error text for the model, or None if the call can run."""
        if tool is None:
            available = ", ".join(t.name for t in self.available_tools) or "none"
            return f"Unknown tool '{tool_call.name}'. Available tools: {available}"
        try:
            js_lib.validate(tool_call.arguments, tool.parameters_schema())
        except js_lib.ValidationError as e:
            path = "/".join(str(p) for p in e.absolute_path) or "(root)"
            return (
                f"Invalid arguments for tool '{tool.name}' at {path}: {e.message}. "
                f"Fix the arguments and call the tool again."
            )
        except js_lib.SchemaError as e:
            # A tool with a malformed parameter schema still runs, unvalidated.
            self.logger.warning(
                f"Tool {tool.name} has an invalid parameter schema, arguments not validated: {e.message}"
            )
        return None

    async def _execute_tool(self, tool_func, arguments):
        """
        Execute a tool and yield ``("stream", chunk)`` / ``("final", content)``.

        For async/sync generators: every yield except the last is streamed to the
        client; the last yield is the content sent to the LLM as the tool message.
        For plain functions: a single ``("final", result)`` is produced.
        """
        action_params = inspect.signature(tool_func).parameters
        pass_self = "agent" in action_params

        args = arguments
        kwargs = {"agent": self} if pass_self else {}

        if inspect.iscoroutinefunction(tool_func):
            result = await tool_func(**args, **kwargs)
            yield ("final", result)
        elif inspect.isasyncgenfunction(tool_func):
            last = None
            async for res in tool_func(**args, **kwargs):
                if last is not None:
                    yield ("stream", last)
                last = res
            yield ("final", last if last is not None else "")
        elif inspect.isgeneratorfunction(tool_func):
            last = None
            for res in tool_func(**args, **kwargs):
                if last is not None:
                    yield ("stream", last)
                last = res
            yield ("final", last if last is not None else "")
        else:
            yield ("final", tool_func(**args, **kwargs))

    def _merge_tools(self, *tool_lists: List[Tool]) -> List[Tool]:
        seen = set()
        merged: List[Tool] = []
        for tools in tool_lists:
            for tool in tools:
                if tool.name not in seen:
                    seen.add(tool.name)
                    merged.append(tool)
        return merged

    def _tools_from_skills(self, skills: List[Skill]) -> List[Tool]:
        tools: List[Tool] = []
        for skill in skills:
            tools.extend(skill.tools)
        return tools

    def _lookup_tool(self, name: str) -> Optional[Tool]:
        catalog = self._merge_tools(
            self.available_tools,
            self.tools,
            self._tools_from_skills(self.skills),
        )
        return next((t for t in catalog if t.name == name), None)

    def _build_system_prompt(self) -> Message:
        self.system_prompt = f"""
        You are an AI agent named {self.name}. Your role is as follows:
        {self.base_prompt}
        """

        for skill in self.available_skills:
            self.system_prompt += "\n\n" + skill.build_prompt()

        return Message(role="system", content=self.system_prompt)