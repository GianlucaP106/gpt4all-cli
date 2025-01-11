# GPT4All Terminal Chatbot

This project provides a command-line interface (CLI) to interact with a local instance of GPT4All, allowing users to engage in text-based conversations directly from their terminal. The output is formatted using Markdown for better readability.

## Features

- **Local Model Interaction**: Run your own LLM locally without relying on external APIs.
- **Markdown Output**: Outputs are rendered as Markdown for enhanced formatting and clarity.
- **Configurable Tokens**: Set the maximum number of tokens to generate per response.
- **Persistent Context**: Maintain context across multiple interactions within a session.

## Requirements

Before running this application, ensure you have installed:

- Python 3.8 or higher

Ensure your system has a compatible GPU if using the "gpu" device option.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/GianlucaP106/gpt4all-cli
   cd gpt4all-cli
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Command Structure

```bash
python gpt4all-cli.py --model <path_to_model> [--out <output_file>] [-t <max_tokens>] [prompt]
```

- `--model`: Path to the `.gguf` model file. This is required.
- `-o`, `--out`: Optional path to save raw output of interactions.
- `-t`, `--tokens`: Maximum number of tokens per response (default: 1024).
- `[prompt]`: Initial prompt for conversation.

### Examples

1. **Start a Chat Session**:

   ```bash
   python gpt4all-cli.py --model model.gguf
   ```

2. **Provide an Initial Prompt**:

   ```bash
   python gpt4all-cli.py --model model.gguf "What is the weather like today?"
   ```

3. **Save Output to a File**:

   ```bash
   python gpt4all-cli.py --model model.gguf -o output.txt
   ```

4. **Set Maximum Tokens**:

   ```bash
   python gpt4all-cli.py --model model.gguf -t 512 "Tell me about AI."
   ```

### Interactive Mode

If no prompt is provided, the application enters an interactive mode where you can type your questions directly into the terminal.

```plaintext
>> What's the capital of France?
```

Press `Ctrl+C` to exit the session at any time.

---
