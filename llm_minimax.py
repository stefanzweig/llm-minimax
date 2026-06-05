import httpx
import json
import llm
from pydantic import Field
from typing import Optional

# MiniMax API base URL
API_BASE = "https://api.minimaxi.com/v1"

# Available MiniMax models
MODEL_IDS = [
    "MiniMax-M2.1",
    "MiniMax-M1",
    "MiniMax-Text-01",
    "MiniMax-VL-01",
]


@llm.hookimpl
def register_models(register):
    """Register MiniMax models with llm."""
    for model_id in MODEL_IDS:
        register(
            MiniMaxModel(model_id),
            AsyncMiniMaxModel(model_id),
            aliases=(model_id,),
        )


class _SharedMiniMax:
    """Shared implementation for sync and async MiniMax models."""

    needs_key = "minimax"
    key_env_var = "LLM_MINIMAX_KEY"
    can_stream = True
    supports_schema = True
    supports_tools = True

    class Options(llm.Options):
        temperature: Optional[float] = Field(
            description=(
                "Controls the randomness of the output. Lower values make the "
                "output more focused and deterministic, higher values make it "
                "more creative. Range: 0.0 to 1.0."
            ),
            default=None,
            ge=0.0,
            le=1.0,
        )
        top_p: Optional[float] = Field(
            description=(
                "Controls diversity via nucleus sampling. Tokens are selected "
                "from the smallest set whose cumulative probability exceeds "
                "top_p. Range: 0.0 to 1.0."
            ),
            default=None,
            ge=0.0,
            le=1.0,
        )
        max_tokens: Optional[int] = Field(
            description="Maximum number of tokens to generate in the response.",
            default=None,
            ge=1,
        )
        timeout: Optional[float] = Field(
            description=(
                "The maximum time in seconds to wait for a response. "
                "If the model does not respond within this time, "
                "the request will be aborted."
            ),
            default=None,
        )
        json_object: Optional[bool] = Field(
            description="Force the output to be valid JSON.",
            default=None,
        )

    def __init__(self, model_id):
        self.model_id = "minimax/{}".format(model_id)
        self.minimax_model_id = model_id

    def build_messages(self, prompt, conversation):
        """Build the messages array from prompt and conversation history."""
        messages = []

        # Add system message if present
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})

        # Add conversation history
        if conversation:
            for response in conversation.responses:
                # User message
                user_content = response.prompt.prompt
                if response.prompt.attachments:
                    # Handle attachments (images, etc.)
                    content = [{"type": "text", "text": user_content}] if user_content else []
                    for attachment in response.prompt.attachments:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": attachment.url or f"data:{attachment.resolve_type()};base64,{attachment.base64_content()}"
                            }
                        })
                    messages.append({"role": "user", "content": content if content else user_content})
                else:
                    messages.append({"role": "user", "content": user_content})

                # Assistant message
                assistant_text = response.text_or_raise()
                if assistant_text:
                    messages.append({"role": "assistant", "content": assistant_text})

        # Add current prompt
        if prompt.attachments:
            content = [{"type": "text", "text": prompt.prompt}] if prompt.prompt else []
            for attachment in prompt.attachments:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": attachment.url or f"data:{attachment.resolve_type()};base64,{attachment.base64_content()}"
                    }
                })
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt.prompt})

        return messages

    def build_request_body(self, prompt, conversation):
        """Build the request body for the API call."""
        body = {
            "model": self.minimax_model_id,
            "messages": self.build_messages(prompt, conversation),
            "stream": True,
        }

        # Add optional parameters
        if prompt.options.temperature is not None:
            body["temperature"] = prompt.options.temperature
        if prompt.options.top_p is not None:
            body["top_p"] = prompt.options.top_p
        if prompt.options.max_tokens is not None:
            body["max_tokens"] = prompt.options.max_tokens

        # Add tool calling support
        if prompt.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    }
                }
                for tool in prompt.tools
            ]

        # Add JSON schema support
        if prompt.schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"schema": prompt.schema},
            }
        elif prompt.options and prompt.options.json_object:
            body["response_format"] = {"type": "json_object"}

        return body

    def get_headers(self, key):
        """Get the request headers."""
        api_key = self.get_key(key)
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def parse_stream_event(self, line):
        """Parse a single SSE event line."""
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return None
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return None
        return None

    def extract_text(self, chunk):
        """Extract text content from a response chunk."""
        try:
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                return delta.get("content", "")
        except (KeyError, IndexError):
            pass
        return ""

    def extract_tool_calls(self, chunk):
        """Extract tool calls from a response chunk."""
        try:
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                tool_calls = delta.get("tool_calls", [])
                if tool_calls:
                    return tool_calls
        except (KeyError, IndexError):
            pass
        return None

    def extract_usage(self, chunk):
        """Extract usage information from a response chunk."""
        try:
            return chunk.get("usage")
        except (KeyError, TypeError):
            return None


class MiniMaxModel(_SharedMiniMax, llm.KeyModel):
    """Synchronous MiniMax model implementation."""

    def execute(self, prompt, stream, response, conversation, key):
        url = f"{API_BASE}/chat/completions"
        body = self.build_request_body(prompt, conversation)
        headers = self.get_headers(key)

        gathered_chunks = []
        tool_call_accumulators = {}

        with httpx.stream(
            "POST",
            url,
            headers=headers,
            json=body,
            timeout=prompt.options.timeout,
        ) as http_response:
            for line in http_response.iter_lines():
                line = line.strip()
                if not line:
                    continue
                chunk = self.parse_stream_event(line)
                if chunk is None:
                    continue

                gathered_chunks.append(chunk)

                # Extract and yield text content
                text = self.extract_text(chunk)
                if text:
                    yield text

                # Accumulate tool calls
                tool_calls = self.extract_tool_calls(chunk)
                if tool_calls:
                    for tc in tool_calls:
                        idx = tc.get("index", 0)
                        if idx not in tool_call_accumulators:
                            tool_call_accumulators[idx] = {
                                "id": tc.get("id", ""),
                                "name": "",
                                "arguments": "",
                            }
                        func = tc.get("function", {})
                        if func.get("name"):
                            tool_call_accumulators[idx]["name"] = func["name"]
                        if func.get("arguments"):
                            tool_call_accumulators[idx]["arguments"] += func["arguments"]

                # Check for usage info
                usage = self.extract_usage(chunk)
                if usage:
                    response.set_usage(
                        input=usage.get("prompt_tokens"),
                        output=usage.get("completion_tokens"),
                    )

        # Process accumulated tool calls
        for idx, tc in sorted(tool_call_accumulators.items()):
            if tc["name"]:
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                response.add_tool_call(
                    llm.ToolCall(
                        name=tc["name"],
                        arguments=args,
                    )
                )

        # Store full response
        if gathered_chunks:
            response.response_json = gathered_chunks[-1]


class AsyncMiniMaxModel(_SharedMiniMax, llm.AsyncKeyModel):
    """Asynchronous MiniMax model implementation."""

    async def execute(self, prompt, stream, response, conversation, key):
        url = f"{API_BASE}/chat/completions"
        body = self.build_request_body(prompt, conversation)
        headers = self.get_headers(key)

        gathered_chunks = []
        tool_call_accumulators = {}

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                url,
                headers=headers,
                json=body,
                timeout=prompt.options.timeout,
            ) as http_response:
                async for line in http_response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    chunk = self.parse_stream_event(line)
                    if chunk is None:
                        continue

                    gathered_chunks.append(chunk)

                    # Extract and yield text content
                    text = self.extract_text(chunk)
                    if text:
                        yield text

                    # Accumulate tool calls
                    tool_calls = self.extract_tool_calls(chunk)
                    if tool_calls:
                        for tc in tool_calls:
                            idx = tc.get("index", 0)
                            if idx not in tool_call_accumulators:
                                tool_call_accumulators[idx] = {
                                    "id": tc.get("id", ""),
                                    "name": "",
                                    "arguments": "",
                                }
                            func = tc.get("function", {})
                            if func.get("name"):
                                tool_call_accumulators[idx]["name"] = func["name"]
                            if func.get("arguments"):
                                tool_call_accumulators[idx]["arguments"] += func["arguments"]

                    # Check for usage info
                    usage = self.extract_usage(chunk)
                    if usage:
                        response.set_usage(
                            input=usage.get("prompt_tokens"),
                            output=usage.get("completion_tokens"),
                        )

        # Process accumulated tool calls
        for idx, tc in sorted(tool_call_accumulators.items()):
            if tc["name"]:
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                response.add_tool_call(
                    llm.ToolCall(
                        name=tc["name"],
                        arguments=args,
                    )
                )

        # Store full response
        if gathered_chunks:
            response.response_json = gathered_chunks[-1]
