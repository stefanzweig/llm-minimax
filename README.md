# llm-minimax

[![PyPI](https://img.shields.io/pypi/v/llm-minimax.svg)](https://pypi.org/project/llm-minimax/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/stefanzweig/llm-minimax/blob/main/LICENSE)

LLM plugin to access [MiniMax](https://www.minimaxi.com/) models (M1, M2, etc.) via API.

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).

```bash
llm install llm-minimax
```

## Usage

Configure the model by setting a key called "minimax" to your [MiniMax API key](https://platform.minimaxi.com/):

```bash
llm keys set minimax
```

```
<paste key here>
```

You can also set the API key by assigning it to the environment variable `LLM_MINIMAX_KEY`.

Now run the model using `-m MiniMax-M2.1`, for example:

```bash
llm -m MiniMax-M2.1 "Tell me a joke about artificial intelligence"
```

You can set the [default model](https://llm.datasette.io/en/stable/setup.html#setting-a-custom-default-model) to avoid the extra `-m` option:

```bash
llm models default MiniMax-M2.1
llm "Tell me a joke about artificial intelligence"
```

## Images / Attachments

Multi-modal models like `MiniMax-M3` and `MiniMax-VL-01` support image input via the `-a` flag:

```bash
# Image file
llm -m MiniMax-M3 "Describe this image" -a photo.jpg

# Image URL
llm -m MiniMax-M3 "What's in this picture?" -a https://example.com/image.png

# Multiple images
llm -m MiniMax-M3 "Compare these two images" -a img1.jpg -a img2.jpg
```

Supported image types: PNG, JPEG, WebP, GIF.

## Available models

- `minimax/MiniMax-M3` - Latest flagship model, 1M context, native multimodal, frontier coding (released 2026-05-31)
- `minimax/MiniMax-M2.7` - Strong reasoning and general tasks
- `minimax/MiniMax-M2.5` - Enhanced reasoning capabilities
- `minimax/MiniMax-M2.1` - Previous generation flagship, strong reasoning capabilities
- `minimax/MiniMax-M1` - Previous generation model
- `minimax/MiniMax-Text-01` - Text-optimized model
- `minimax/MiniMax-VL-01` - Vision-language model
- `minimax/MiniMax-Code` - Coding-specialized model

All of these models have aliases that omit the `minimax/` prefix, for example:

```bash
llm -m MiniMax-M2.1 "Hello world"
```

## Options

The following options are supported:

| Option | Type | Description |
|--------|------|-------------|
| `temperature` | float | Controls randomness (0.0 to 1.0) |
| `top_p` | float | Nucleus sampling (0.0 to 1.0) |
| `max_tokens` | int | Maximum tokens to generate |
| `timeout` | float | Request timeout in seconds |
| `json_object` | bool | Force JSON output |

Example with options:

```bash
llm -m MiniMax-M2.1 -o temperature 0.7 -o max_tokens 500 "Write a short poem"
```

## JSON Output

Use `-o json_object 1` to force the output to be JSON:

```bash
llm -m MiniMax-M2.1 -o json_object 1 \
  'List 3 cities in California as JSON: [{"name": "..."}]'
```

## Chat

To chat interactively with the model, run `llm chat`:

```bash
llm chat -m MiniMax-M2.1
```

## Development

To set up this plugin locally, first checkout the code, then install:

```bash
cd llm-minimax
llm install -e .
```

Run with the plugin like this:

```bash
llm models -q minimax
```

## API Reference

MiniMax API documentation: https://platform.minimaxi.com/document/guides/chat-model/introduction

## License

Apache-2.0
