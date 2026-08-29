import httpx
import json
import llm
from pydantic import Field
from typing import Optional

# MiniMax API base URL
API_BASE = "https://api.minimaxi.com/v1"

# Available MiniMax models
MODEL_IDS = [
    "MiniMax-M3",
    "MiniMax-M2.7",
    "MiniMax-M2.5",
    "MiniMax-M2.1",
    "MiniMax-M1",
    "MiniMax-Text-01",
    "MiniMax-VL-01",
    "MiniMax-Code",
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

    # Supported attachment types (images for vision models like M3, VL-01)
    attachment_types = {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
    }

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
                user_msg = self._build_user_message(response.prompt)
                messages.append(user_msg)

                # Assistant message
                assistant_text = response.text_or_raise()
                if assistant_text:
                    messages.append({"role": "assistant", "content": assistant_text})

        # Add current prompt
        messages.append(self._build_user_message(prompt))

        return messages

    def _build_user_message(self, prompt):
        """Build a user message, handling attachments if present."""
        if not prompt.attachments:
            # No attachments, simple text content
            return {"role": "user", "content": prompt.prompt or ""}

        # Has attachments, build multipart content
        content = []

        # Add text part if present
        if prompt.prompt:
            content.append({"type": "text", "text": prompt.prompt})

        # Add each attachment
        for attachment in prompt.attachments:
            mime_type = attachment.resolve_type()

            # Check if it's an image type we support
            if mime_type.startswith("image/"):
                # For URL attachments, use the URL directly
                if attachment.url:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": attachment.url}
                    })
                else:
                    # For file/path/content attachments, use base64 data URL
                    b64 = attachment.base64_content()
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64}"}
                    })

        # If only attachments and no text, ensure we have content
        if not content:
            return {"role": "user", "content": ""}

        return {"role": "user", "content": content}

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
        # MiniMax's API requires a `name` field inside `json_schema`;
        # derive one from prompt.schema.name if present, else fall back to
        # the model id so the request isn't rejected with error 2013.
        if prompt.schema:
            schema_name = (
                getattr(prompt.schema, "name", None)
                if not isinstance(prompt.schema, dict)
                else prompt.schema.get("name")
            ) or self.minimax_model_id
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": prompt.schema,
                },
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
            if http_response.status_code != 200:
                raise llm.ModelError(
                    f"MiniMax API error {http_response.status_code}: "
                    f"{http_response.read().text[:500]}"
                )
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
                if http_response.status_code != 200:
                    raise llm.ModelError(
                        f"MiniMax API error {http_response.status_code}: "
                        f"{http_response.read().text[:500]}"
                    )
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
