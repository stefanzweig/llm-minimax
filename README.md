# llm-minimax

[![PyPI](https://img.shields.io/pypi/v/llm-minimax.svg)](https://pypi.org/project/llm-minimax/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/stefanzweig/llm-minimax/blob/main/LICENSE)

LLM plugin for accessing [MiniMax](https://www.minimaxi.com/) chat, reasoning, coding, and vision models through the MiniMax API.

## Features

- Registers current MiniMax models for use with [LLM](https://llm.datasette.io/)
- Supports streaming responses
- Supports image attachments for multimodal models
- Supports JSON mode and JSON schema structured output
- Supports LLM tool/function calling
- Strips MiniMax `<think>...</think>` blocks from streamed output
- Records token usage when the API returns usage metadata

## Installation

Install this plugin in the same environment as LLM:

```bash
llm install llm-minimax
```

For local development, install it from a checkout:

```bash
git clone https://github.com/stefanzweig/llm-minimax.git
cd llm-minimax
llm install -e .
```

## Configuration

Set a key called `minimax` to your [MiniMax API key](https://platform.minimaxi.com/):

```bash
llm keys set minimax
```

Then paste your API key when prompted.

You can also provide the key with the `LLM_MINIMAX_KEY` environment variable:

```bash
export LLM_MINIMAX_KEY="your-api-key"
```

## Usage

Run a prompt with any registered MiniMax model:

```bash
llm -m MiniMax-M2.1 "Tell me a joke about artificial intelligence"
```

Models are registered with their full `minimax/` IDs and with aliases that omit the prefix. These two commands are equivalent:

```bash
llm -m minimax/MiniMax-M2.1 "Hello world"
llm -m MiniMax-M2.1 "Hello world"
```

Set a default model to avoid passing `-m` every time:

```bash
llm models default MiniMax-M2.1
llm "Tell me a joke about artificial intelligence"
```

Start an interactive chat session:

```bash
llm chat -m MiniMax-M2.1
```

List the MiniMax models registered by the plugin:

```bash
llm models -q minimax
```

## Available Models

- `minimax/MiniMax-M3` - Flagship multimodal model with long context
- `minimax/MiniMax-M2.7` - Strong reasoning and general-purpose model
- `minimax/MiniMax-M2.5` - Enhanced reasoning model
- `minimax/MiniMax-M2.1` - Previous-generation flagship reasoning model
- `minimax/MiniMax-M1` - Previous-generation model
- `minimax/MiniMax-Text-01` - Text-optimized model
- `minimax/MiniMax-VL-01` - Vision-language model
- `minimax/MiniMax-Code` - Coding-specialized model

Each model also has a short alias without the `minimax/` prefix, for example `MiniMax-M3`.

## Images and Attachments

Multimodal models such as `MiniMax-M3` and `MiniMax-VL-01` accept image input through LLM's `-a` attachment option:

```bash
# Image file
llm -m MiniMax-M3 "Describe this image" -a photo.jpg

# Image URL
llm -m MiniMax-M3 "What's in this picture?" -a https://example.com/image.png

# Multiple images
llm -m MiniMax-M3 "Compare these two images" -a img1.jpg -a img2.jpg
```

Supported image attachment types:

- PNG
- JPEG
- WebP
- GIF

## Options

The plugin supports these model options:

| Option | Type | Description |
| --- | --- | --- |
| `temperature` | float | Controls randomness. Must be between `0.0` and `1.0`. |
| `top_p` | float | Controls nucleus sampling. Must be between `0.0` and `1.0`. |
| `max_tokens` | int | Maximum number of tokens to generate. |
| `timeout` | float | Request timeout in seconds. |
| `json_object` | bool | Requests valid JSON object output. |

Example:

```bash
llm -m MiniMax-M2.1 \
  -o temperature 0.7 \
  -o max_tokens 500 \
  "Write a short poem"
```

## JSON Output

Use `-o json_object 1` to request a valid JSON object response:

```bash
llm -m MiniMax-M2.1 -o json_object 1 \
  'List 3 cities in California as JSON: {"cities": [{"name": "..."}]}'
```

The plugin also adds an instruction reminding the model to return JSON only, without Markdown code fences or thinking tags.

## Structured Output

Use LLM's `--schema` option to request JSON schema structured output:

```bash
llm -m MiniMax-M2.1 --schema '{
  "type": "object",
  "properties": {
    "cities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "population": {"type": "integer"}
        },
        "required": ["name", "population"]
      }
    }
  },
  "required": ["cities"]
}' 'List 3 major cities in California with their populations'
```

The plugin sends MiniMax a `response_format` request and adds a matching system instruction as a fallback for models that do not strictly enforce the response format.

## Tool Calling

This plugin advertises tool-calling support to LLM and forwards tool definitions to MiniMax as function tools. Tool calls returned by the API are collected from the stream and exposed back to LLM.

Use this with LLM features that provide tools to the model, such as compatible LLM plugins or commands that register tools.

## Thinking Tags

Some MiniMax models can emit internal reasoning inside `<think>...</think>` tags. The plugin removes those blocks from both streamed output and conversation history before sending previous assistant messages back to the API.

## Development

Install the package in editable mode:

```bash
llm install -e .
```

Check that the plugin is available:

```bash
llm models -q minimax
```

The package requires Python 3.9 or newer.

## API Reference

MiniMax API documentation: https://platform.minimaxi.com/document/guides/chat-model/introduction

LLM documentation: https://llm.datasette.io/

## License

Apache-2.0
