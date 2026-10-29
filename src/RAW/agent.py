import asyncio
import os
import time
import uuid
from typing import List, Literal, Optional, AsyncGenerator, Awaitable, Callable, Sequence, Set, Union, Dict, Any
import jsonschema as js_lib  # type: ignore[import-untyped]
from RAW.models import Tool, ToolCall, ToolParam, LLMCapability, Message, File, FileType, Image, Skill, Usage, discover_skills
from RAW.llms.base import BaseLLM
from RAW.utils import Logger, Router
from RAW.utils.metrics import Metrics, NullMetrics
from RAW.utils.compaction import compact_messages, estimate_tokens
import inspect

LOAD_SKILL_TOOL = "load_skill"
READ_SKILL_RESOURCE_TOOL = "read_skill_resource"


class _TimeLimitReached(Exception):
    """Raised inside the loop when the agent's time limit runs out."""


_TOOL_CANCELLED = "Cancelled: the agent's time limit was reached before this tool finished."


class Agent:
    def __init__(
            self,
            name: str, 
            base_prompt: str = "", 
            tools: Optional[List[Tool]] = None,
            llm: Optional[BaseLLM] = None,
            logger: Optional[Logger] = None,
            history: Optional[List[Message]] = None,
            router: Optional[Router] = None,
            skills: Optional[List[Skill]] = None,
            max_iterations: int = 25,
            compactor_llm: Optional[BaseLLM] = None,
            context_window: Optional[int] = None,
            compact_trigger: float = 0.70,
            compact_target: float = 0.50,
            skills_dir: Optional[Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]]] = None,
            load_skill_tools: bool = False,
            skill_mode: Literal["catalog", "inline"] = "catalog",
            metrics: Optional[Metrics] = None,
            on_message: Optional[Callable[[Message], Union[None, Awaitable[None]]]] = None,
            time_limit_seconds: Optional[float] = 3600,
        ):
        if not 0 < compact_target < compact_trigger < 1:
            raise ValueError("compact_target and compact_trigger must satisfy 0 < target < trigger < 1.")
        if skill_mode not in ("catalog", "inline"):
            raise ValueError("skill_mode must be 'catalog' or 'inline'.")
        tools = tools if tools is not None else []
        skills = list(skills) if skills is not None else []
        if skills_dir is not None:
            # `tools.py` files in skill folders only run when load_skill_tools=True.
            dirs = [skills_dir] if isinstance(skills_dir, (str, os.PathLike)) else list(skills_dir)
            skills += discover_skills([os.fspath(d) for d in dirs], load_tools=load_skill_tools)
        names = [s.name for s in skills]
        if len(names) != len(set(names)):
            raise ValueError(f"Skill names must be unique, got {names}.")
        reserved = {LOAD_SKILL_TOOL, READ_SKILL_RESOURCE_TOOL} & {t.name for t in tools}
        if reserved:
            raise ValueError(f"Tool names {sorted(reserved)} are reserved for skills.")
        history = history if history is not None else []
        self.logger: Logger = logger if logger is not None else Logger(name)
        self.name = name
        # Receives counters/histograms/gauges (raw.agent.*, raw.llm.*, raw.tool.*); never raises into the run.
        self.metrics: Metrics = metrics if metrics is not None else NullMetrics()
        self._run_status = "success"
        # Tools that stay available whatever the router or `skill=` selects (e.g. a sub-agent's
        # ask_clarification). Set by RAW.subagent.
        self.control_tools: List[Tool] = []
        # How deep in a chain of sub-agents this agent runs; 0 for a top-level agent.
        self._depth = 0
        self._halt: Optional[Dict[str, Any]] = None
        # Context compaction: when the prompt passes `compact_trigger` of the usable
        # context window, older history is summarized down to `compact_target`.
        # The summary is written by `compactor_llm`, or by `llm` when that is None.
        self.compactor_llm = compactor_llm
        self.context_window = context_window
        self.compact_trigger = compact_trigger
        self.compact_target = compact_target
        # Summary of history that was compacted away; rendered in the system prompt.
        self.context_summary: str = ""
        # Provider-reported prompt size of the latest call and how many messages it covered.
        self._prompt_tokens: Optional[int] = None
        self._prompt_messages: int = 0
        self._warned_no_window = False
        # Max LLM calls per __call__; guards against endless tool-call loops.
        self.max_iterations = max_iterations
        # Wall-clock budget for one __call__ (LLM calls, compaction, async tools), so a
        # slow or looping run cannot hold the GPU. None disables it. Sync tools run on
        # the event loop and cannot be interrupted, only stopped after they return.
        self.time_limit_seconds = time_limit_seconds
        self._deadline: Optional[float] = None
        # Called (sync or async) with every message the run adds to the conversation, in
        # order, so callers can persist history exactly as the loop built it. Compaction
        # rewrites self.messages but is not reported. Errors raised by the hook propagate.
        self.on_message = on_message
        # Token usage summed over every LLM call this agent made / made in the last run.
        self.usage = Usage()
        self.last_run_usage = Usage()
        self.logger.info(f"Initializing agent: {name}")
        if not llm:
            raise ValueError("Agent must be initialized with an LLM.")
        self.llm: BaseLLM = llm
        self.router = router

        if LLMCapability.TOOLS not in self.llm.capabilities:
            self.logger.warning("The LLM assigned to the agent does not have tool usage capabilities. Tools will not be used.") 
            if skill_mode == "catalog" and skills:
                self.logger.warning("Skills cannot be loaded on demand without tool support; using skill_mode='inline'.")
                skill_mode = "inline"

        # "catalog": the prompt lists skill names and descriptions and the model loads a skill
        # with `load_skill` when it needs it. "inline": every skill's full prompt is in the system
        # prompt. Skills picked by the router or by `skill=` are always inline.
        self.skill_mode = skill_mode
        self._loaded_skills: Set[str] = set()
        self._catalog_skills: List[Skill] = []
        self._load_enabled = False
        self.base_prompt = base_prompt

        self.tools: List[Tool] = tools
        self.skills: List[Skill] = skills
        # Without a router, default to every skill (per skill_mode) and the agent's tools.
        self.available_skills: List[Skill] = []
        self.available_tools: List[Tool] = []
        if self.router is None:
            self._use_all_skills()
        else:
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
        started = time.perf_counter()
        self._run_status = "success"
        try:
            async for chunk in self._run(user_message, user_files, stream, user_summary, skill):
                yield chunk
        except (GeneratorExit, asyncio.CancelledError):
            self._run_status = "cancelled"
            raise
        except Exception:
            self._run_status = "error"
            raise
        finally:
            tags = {"agent": self.name, "status": self._run_status}
            self._count("raw.agent.runs", tags=tags)
            self._observe("raw.agent.run_duration_seconds", time.perf_counter() - started, tags=tags)

    async def _run(
            self,
            user_message: str,
            user_files: List[File],
            stream: bool,
            user_summary: Optional[str],
            skill: Optional[Skill],
        ) -> AsyncGenerator[Union[Dict[str, Any], str], None]:
        self.logger.info(f'USER MESSAGE: {user_message}')
        self.last_run_usage = Usage()
        self._halt = None
        self._deadline = (
            asyncio.get_running_loop().time() + self.time_limit_seconds
            if self.time_limit_seconds is not None else None
        )
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
            self._catalog_skills, self._load_enabled = [], False
            self.available_tools = self._merge_tools(skill.tools, self._builtin_tools([skill]))
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
            self._catalog_skills, self._load_enabled = [], False
            self.available_tools = self._merge_tools(
                selected_tools,
                self._tools_from_skills(self.available_skills),
                self._builtin_tools(self.available_skills),
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
            self._use_all_skills()
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
        try:
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
        except _TimeLimitReached:
            yield {"agent_name": self.name, 'content': self._time_limit_reached()}

    async def execute_stream(self) -> AsyncGenerator[Union[Message, Dict[str, Any]], None]:
        iterations = 0
        while True:
            if iterations >= self.max_iterations:
                yield self._max_iterations_reached()
                return
            iterations += 1

            async for event in self._iter_within_limit(self._compact_if_needed()):
                yield event

            sent_messages = len(self.messages)
            llm_started = time.perf_counter()
            try:
                chat_response_generator = await self._within_limit(self.llm.chat(
                    messages=self.messages, schema=None, stream=True, tools=self.available_tools
                ))
            except _TimeLimitReached:
                self._record_llm_call(llm_started, "time_limit")
                raise
            except Exception:
                self._record_llm_call(llm_started, "error")
                raise

            accumulated_content = ""
            final_message = None
            has_tool_calls = False
            # Content-less message carrying usage / raw provider blocks (end of stream).
            metadata_message = None

            try:
                async for response in self._iter_within_limit(chat_response_generator):
                    if response.usage:
                        self._add_usage(response.usage, sent_messages)
                    if hasattr(response, "tool_calls") and response.tool_calls:
                        has_tool_calls = True
                        final_message = response
                    elif not response.content:
                        metadata_message = response
                    elif response.content:
                        accumulated_content += response.content
                        yield Message(
                            role="assistant",
                            content=response.content,
                            tool_calls=[],
                            images=[]
                        )
            except _TimeLimitReached:
                self._record_llm_call(llm_started, "time_limit")
                raise
            except Exception:
                self._record_llm_call(llm_started, "error")
                raise
            self._record_llm_call(llm_started, "success")

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
                if self._halt:
                    return
                continue

            elif accumulated_content:
                complete_message = Message(
                    role="assistant",
                    content=accumulated_content,
                    tool_calls=[],
                    images=[]
                )
                if metadata_message:
                    complete_message.usage = metadata_message.usage
                    complete_message.think_content = metadata_message.think_content
                    complete_message.provider_content = metadata_message.provider_content
                    complete_message.provider = metadata_message.provider
                    complete_message.citations = metadata_message.citations
                await self._append(complete_message)
                self.logger.info(
                    f"{self.name}: {complete_message.content}"
                )
                return

            # If no tool calls and no content, break loop
            self.logger.warning(f"Agent {self.name} received empty response from LLM. Stopping.")
            self._run_status = "empty_response"
            return
                     
    async def execute_no_stream(self) -> AsyncGenerator[Union[Message, Dict[str, Any]], None]:
        iterations = 0
        while True:
            if iterations >= self.max_iterations:
                yield self._max_iterations_reached()
                return
            iterations += 1

            async for event in self._iter_within_limit(self._compact_if_needed()):
                yield event

            sent_messages = len(self.messages)
            llm_started = time.perf_counter()
            try:
                response: Message = await self._within_limit(
                    self.llm.chat(messages=self.messages, schema=None, stream=False, tools=self.available_tools)
                )
            except _TimeLimitReached:
                self._record_llm_call(llm_started, "time_limit")
                raise
            except Exception:
                self._record_llm_call(llm_started, "error")
                raise
            self._record_llm_call(llm_started, "success")
            if response.usage:
                self._add_usage(response.usage, sent_messages)
            await self._append(response)
            self.logger.debug(
                f"Agent {self.name} generated response: {response.content}"
            )

            if response.tool_calls:
                async for event in self._run_tool_calls(response.tool_calls):
                    yield event
                if self._halt:
                    return
                continue
            if response.content:
                self.logger.info(
                    msg=f"{self.name}: {response.content}"
                )
                yield response.content
                return
            
            # If no tool calls and no content, break loop
            self.logger.warning(f"Agent {self.name} received empty response from LLM. Stopping.")
            self._run_status = "empty_response"
            return

    async def _run_tool_calls(self, tool_calls: List[ToolCall]) -> AsyncGenerator[Dict[str, Any], None]:
        """Run a round of tool calls. If the time limit runs out, the calls not yet
        finished get a 'Cancelled' result so the history stays valid, then it re-raises."""
        for i, tool_call in enumerate(tool_calls):
            try:
                async for event in self._run_tool_call(tool_call):
                    yield event
            except _TimeLimitReached:
                for skipped in tool_calls[i + 1:]:
                    await self._append_tool_result(skipped, _TOOL_CANCELLED)
                raise

    async def _run_tool_call(self, tool_call: ToolCall) -> AsyncGenerator[Dict[str, Any], None]:
        """Run one tool call, append its result message and yield client events."""
        yield {
            "tool_call": tool_call.name,
            "arguments": tool_call.arguments
        }
        self.logger.info(
            msg=f"Tool Called: {tool_call.name} with arguments {tool_call.arguments}"
        )
        tool = self._lookup_tool(tool_call.name)
        tool_started = time.perf_counter()
        if tool is None:
            self._record_tool_call(tool_call.name, tool_started, "unknown")
            # Every tool call needs a result, or the next request is rejected.
            available = ", ".join(t.name for t in self.available_tools) or "none"
            error_msg = (
                f"Error calling tool: tool '{tool_call.name}' is not available. "
                f"Available tools: {available}"
            )
            self.logger.error(f"Tool not found: {tool_call.name}")
            yield await self._append_tool_result(tool_call, error_msg)
            return

        invalid = self._check_arguments(tool, tool_call)
        if invalid is not None:
            self._record_tool_call(tool_call.name, tool_started, "invalid_arguments")
            self.logger.warning(f"Rejected tool call {tool_call.name}: {invalid}")
            yield await self._append_tool_result(tool_call, invalid)
            return

        try:
            result = None
            async for kind, payload in self._iter_within_limit(
                self._execute_tool(tool.function, tool_call.arguments)
            ):
                if kind == "stream":
                    # Intermediate tool yields (e.g. full table data) → client
                    yield {"tool_yield": payload}
                else:
                    result = payload
        except _TimeLimitReached:
            self._record_tool_call(tool_call.name, tool_started, "time_limit")
            yield await self._append_tool_result(tool_call, _TOOL_CANCELLED)
            raise
        except Exception as e:
            self._record_tool_call(tool_call.name, tool_started, "error")
            self.logger.error(f"Error executing tool {tool_call.name}: {e}")
            yield await self._append_tool_result(tool_call, f"Error calling tool: {e}")
            return

        self.logger.info(f"Tool executed successfully: {result}")
        self._record_tool_call(tool_call.name, tool_started, "success")
        yield await self._append_tool_result(tool_call, str(result))

    async def _append_tool_result(self, tool_call: ToolCall, content: str) -> Dict[str, Any]:
        await self._append(Message(role="tool", content=content, tool_call_id=tool_call.id))
        return {"tool_response": content}

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

    def _check_arguments(self, tool: Tool, tool_call: ToolCall) -> Optional[str]:
        """Validate arguments against the tool's JSON Schema; return an error text for the model."""
        try:
            js_lib.validate(tool_call.arguments, tool.parameters_schema())
        except js_lib.ValidationError as e:
            path = "/".join(str(p) for p in e.absolute_path) or "(root)"
            return (
                f"Error calling tool: invalid arguments for tool '{tool.name}' at {path}: {e.message}. "
                f"Fix the arguments and call the tool again."
            )
        except js_lib.SchemaError as e:
            # A tool with a malformed parameter schema still runs, unvalidated.
            self.logger.warning(
                f"Tool {tool.name} has an invalid parameter schema, arguments not validated: {e.message}"
            )
        return None

    def _remaining_time(self) -> Optional[float]:
        if self._deadline is None:
            return None
        return self._deadline - asyncio.get_running_loop().time()

    async def _within_limit(self, awaitable: Awaitable[Any]) -> Any:
        """Await within the run's time limit, raising _TimeLimitReached when it runs out."""
        remaining = self._remaining_time()
        if remaining is None:
            return await awaitable
        if remaining <= 0:
            if inspect.iscoroutine(awaitable):
                awaitable.close()
            raise _TimeLimitReached()
        try:
            return await asyncio.wait_for(awaitable, remaining)
        except asyncio.TimeoutError:
            # A TimeoutError raised by the awaited code itself is not ours.
            if (self._remaining_time() or 0) > 0:
                raise
            raise _TimeLimitReached() from None

    async def _iter_within_limit(self, agen: AsyncGenerator[Any, None]) -> AsyncGenerator[Any, None]:
        try:
            while True:
                try:
                    item = await self._within_limit(agen.__anext__())
                except StopAsyncIteration:
                    return
                yield item
        finally:
            await agen.aclose()

    def _record_tool_call(self, tool: str, started: float, status: str) -> None:
        tags = {"agent": self.name, "tool": tool, "status": status}
        self._count("raw.tool.calls", tags=tags)
        self._observe("raw.tool.duration_seconds", time.perf_counter() - started, tags=tags)

    def _context_budget(self) -> Optional[int]:
        """Prompt tokens available per request: context window minus the reply reserve."""
        window = self.context_window
        reserve = 0
        try:
            info = self.llm.info()
            window = window or info.context_window
            reserve = info.max_tokens or 0
        except Exception:
            pass
        if not window:
            return None
        # A huge max_tokens must not leave no room for the prompt.
        return window - min(reserve, window // 2)

    def _current_prompt_tokens(self) -> int:
        """Provider-reported size of the last prompt plus an estimate for what was added since."""
        if self._prompt_tokens is not None and self._prompt_messages <= len(self.messages):
            return self._prompt_tokens + estimate_tokens(
                self.messages[self._prompt_messages:], model=getattr(self.llm, "model", "")
            )
        return estimate_tokens(self.messages, self.available_tools, getattr(self.llm, "model", ""))

    async def _compact_if_needed(self) -> AsyncGenerator[Dict[str, Any], None]:
        budget = self._context_budget()
        if budget is None:
            if not self._warned_no_window:
                self._warned_no_window = True
                self.logger.warning(
                    f"Agent {self.name}: the LLM reports no context window, so context compaction is "
                    "disabled. Pass context_window= to enable it."
                )
            return
        current = self._current_prompt_tokens()
        self._set_gauge("raw.agent.context_utilization", current / budget, {"agent": self.name})
        if current <= budget * self.compact_trigger:
            return

        summarizer = self.compactor_llm or self.llm
        model = getattr(self.llm, "model", "")
        target = int(budget * self.compact_target)
        self.logger.info(
            f"Agent {self.name}: context at {current}/{budget} tokens, compacting to about {target}."
        )
        result = await compact_messages(
            self.messages,
            summarizer,
            target_tokens=target,
            previous_summary=self.context_summary,
            tools=self.available_tools,
            summary_tokens=max(min(budget // 10, 2000), 200),
            chunk_tokens=max(int(self._summarizer_budget(summarizer, budget) * 0.6), 1000),
            model=model,
            measured_tokens=current,
        )
        if result.method == "none":
            return
        if result.method == "drop":
            self.logger.warning(
                f"Agent {self.name}: summarizing failed, dropped {result.summarized_messages} old messages."
            )
        if result.usage.input_tokens is not None or result.usage.output_tokens is not None:
            self._add_usage(result.usage)

        self._count("raw.agent.compactions", tags={"agent": self.name, "method": result.method})
        self._observe(
            "raw.agent.compaction_tokens_saved",
            max(result.tokens_before - result.tokens_after, 0),
            {"agent": self.name},
        )
        self.context_summary = result.summary
        self.messages = result.messages
        self.system_prompt = self._build_system_prompt()
        self.messages[0] = self.system_prompt
        # The provider-reported size is stale now; fall back to the local estimate.
        self._prompt_tokens = None
        self._prompt_messages = 0
        yield {"compaction": {
            "method": result.method,
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
            "summarized_messages": result.summarized_messages,
        }}

    def _summarizer_budget(self, summarizer: BaseLLM, default: int) -> int:
        try:
            return summarizer.info().context_window or default
        except Exception:
            return default

    @property
    def halted(self) -> Optional[Dict[str, Any]]:
        """Why the last run stopped early because a tool called ``halt``, else None."""
        return self._halt

    def halt(self, reason: Dict[str, Any]) -> None:
        """Called from a tool: stop the run after the current round of tool calls."""
        self._halt = reason
        self._run_status = "halted"

    def _absorb_usage(self, usage: Usage) -> None:
        """Add usage that was already reported by another agent (a sub-agent) to the totals."""
        self.usage = self.usage + usage
        self.last_run_usage = self.last_run_usage + usage

    def _count(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        try:
            self.metrics.increment(name, value, tags=tags)
        except Exception as e:
            self.logger.debug(f"Metrics increment {name} failed: {e}")

    def _observe(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        try:
            self.metrics.histogram(name, value, tags=tags)
        except Exception as e:
            self.logger.debug(f"Metrics histogram {name} failed: {e}")

    def _set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        try:
            self.metrics.gauge(name, value, tags=tags)
        except Exception as e:
            self.logger.debug(f"Metrics gauge {name} failed: {e}")

    def _record_llm_call(self, started: float, status: str) -> None:
        tags = {"agent": self.name, "model": str(getattr(self.llm, "model", "")), "status": status}
        self._count("raw.llm.calls", tags=tags)
        self._observe("raw.llm.call_duration_seconds", time.perf_counter() - started, tags=tags)

    def _add_usage(self, usage: Usage, sent_messages: Optional[int] = None) -> None:
        self.usage = self.usage + usage
        self.last_run_usage = self.last_run_usage + usage
        for direction, tokens in (
            ("input", usage.input_tokens), ("output", usage.output_tokens),
            ("cache_read", usage.cache_read_tokens), ("cache_write", usage.cache_write_tokens),
        ):
            if tokens:
                self._count("raw.llm.tokens", tokens, {"agent": self.name, "direction": direction})
        # Track the prompt size of the latest call (input is the uncached part, see Usage).
        if sent_messages is not None and usage.input_tokens is not None:
            self._prompt_tokens = (
                usage.input_tokens + (usage.cache_read_tokens or 0) + (usage.cache_write_tokens or 0)
            )
            self._prompt_messages = sent_messages

    def _time_limit_reached(self) -> Dict[str, Any]:
        self._run_status = "time_limit"
        self.logger.warning(
            f"Agent {self.name} reached its time limit ({self.time_limit_seconds:g} s). Stopping."
        )
        return {
            "error": "time_limit_reached",
            "time_limit_seconds": self.time_limit_seconds,
            "message": f"Agent {self.name} stopped after {self.time_limit_seconds:g} seconds without a final answer.",
        }

    def _max_iterations_reached(self) -> Dict[str, Any]:
        self._run_status = "max_iterations"
        self.logger.warning(
            f"Agent {self.name} reached max_iterations ({self.max_iterations}). Stopping."
        )
        return {
            "error": "max_iterations_reached",
            "max_iterations": self.max_iterations,
            "message": f"Agent {self.name} stopped after {self.max_iterations} LLM calls without a final answer.",
        }

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

    def _use_all_skills(self) -> None:
        """No router and no ``skill=``: offer every skill, inline or as an on-demand catalog."""
        if self.skill_mode == "catalog" and self.skills:
            loaded = [s for s in self.skills if s.name in self._loaded_skills]
            self.available_skills = []
            self._catalog_skills = [s for s in self.skills if s.name not in self._loaded_skills]
            self._load_enabled = True
            # A skill's own tools unlock when the skill is loaded.
            self.available_tools = self._merge_tools(
                self.tools, self._tools_from_skills(loaded), self._builtin_tools(self.skills)
            )
        else:
            self.available_skills = list(self.skills)
            self._catalog_skills, self._load_enabled = [], False
            self.available_tools = self._merge_tools(
                self.tools, self._tools_from_skills(self.skills), self._builtin_tools(self.skills)
            )

    def _builtin_tools(self, skills_in_play: List[Skill]) -> List[Tool]:
        """Tools RAW adds itself: skill loading, skill resources and `control_tools`."""
        builtins: List[Tool] = list(self.control_tools)
        if self._load_enabled:
            builtins.append(Tool(
                name=LOAD_SKILL_TOOL,
                description=(
                    "Load a skill's full instructions (and unlock its tools) when the request matches it. "
                    "Use the skill names from the 'Available skills' list."
                ),
                parameters=[ToolParam(
                    name="name", type="string", description="Name of the skill to load.",
                    enums=[s.name for s in self.skills], required=True,
                )],
                function=self._load_skill,
            ))
        if any(s.resources() for s in skills_in_play):
            builtins.append(Tool(
                name=READ_SKILL_RESOURCE_TOOL,
                description="Read a reference file that ships with a skill, as listed in the skill's Resources.",
                parameters=[
                    ToolParam(name="skill", type="string", description="Name of the skill.", required=True),
                    ToolParam(name="path", type="string", description="File path relative to the skill folder.", required=True),
                ],
                function=self._read_skill_resource,
            ))
        return builtins

    def _find_skill(self, name: str) -> Skill:
        skill = next((s for s in self.skills if s.name == name), None)
        if skill is None:
            raise ValueError(f"Unknown skill '{name}'. Available skills: {[s.name for s in self.skills]}")
        return skill

    def _load_skill(self, name: str) -> str:
        skill = self._find_skill(name)
        self._loaded_skills.add(skill.name)
        # Takes effect on the next LLM call of this run: the skill's tools become available.
        self._use_all_skills()
        self.logger.info(f"Agent {self.name} loaded skill: {skill.name}")
        return skill.build_prompt()

    def _read_skill_resource(self, skill: str, path: str) -> str:
        return self._find_skill(skill).read_resource(path)

    def _lookup_tool(self, name: str) -> Optional[Tool]:
        catalog = self._merge_tools(self.available_tools, self.tools)
        return next((t for t in catalog if t.name == name), None)

    def _build_system_prompt(self) -> Message:
        self.system_prompt = f"""
        You are an AI agent named {self.name}. Your role is as follows:
        {self.base_prompt}
        """

        for skill in self.available_skills:
            self.system_prompt += "\n\n" + skill.build_prompt()

        if self._catalog_skills:
            self.system_prompt += (
                "\n\n# Available skills\n"
                f"When a request matches one of these skills, call the `{LOAD_SKILL_TOOL}` tool with its name "
                "to get its full instructions before you answer:\n"
                + "\n".join(f"- {s.name}: {s.description}" for s in self._catalog_skills)
            )

        if self.context_summary:
            self.system_prompt += (
                "\n\n# Summary of earlier conversation\n"
                "Older messages were removed to save context. Rely on this summary for them:\n"
                + self.context_summary
            )

        return Message(role="system", content=self.system_prompt)