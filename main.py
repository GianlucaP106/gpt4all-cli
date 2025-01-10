from typing import Iterable
from gpt4all import GPT4All
import argparse
from rich.console import Console
from rich.markdown import Markdown
import sys
import time
import signal


MODELS = {
    "small": "Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf",
    "large": "phi-4-fp16.gguf",
}

SYSTEM_PROMPT = "You are a general purpose AI chat bot. Output in markdown."


class Config:
    def __init__(self, model: str):
        self.model: str = model


def parse_config() -> Config:
    parser = argparse.ArgumentParser(description="gpt-shell is a CLI that runs an LLM")
    parser.add_argument(
        "-l",
        "--large",
        action="store_true",
        help="Runs the prompt or chat with a larger language model",
    )
    parsers = parser.add_subparsers(dest="command")
    parsers.add_parser("models", help="Lists all availables models")
    d = parser.parse_args()
    c = Config("large" if d.large else "small")
    return c


def get_model(typ: str) -> GPT4All:
    model = GPT4All(
        MODELS[typ],
        model_path="./models",
        allow_download=False,
        device="gpu",
    )
    return model


def print_stream(stream: Iterable[str], delay: float = 0.02):
    console = Console()
    all_tokens = []
    for token in stream:
        all_tokens.append(token)
        md = Markdown("".join(all_tokens))
        console.clear()
        console.print(md)
        time.sleep(delay)


def chat(model: GPT4All):
    with model.chat_session(system_prompt=SYSTEM_PROMPT):
        while True:
            prompt = str(input(">> "))
            if prompt == "q":
                print("Goodbye")
                break

            if prompt == "h":
                print("Help...")
                continue

            stream = model.generate(
                prompt,
                max_tokens=1024,
                streaming=True,
            )
            print_stream(stream)
    return


def one(model: GPT4All, prompt: str):
    with model.chat_session(system_prompt=SYSTEM_PROMPT):
        stream = model.generate(
            prompt,
            max_tokens=1024,
            streaming=True,
        )
        print_stream(stream)


def list_models():
    for key, val in MODELS.items():
        print(key, " - ", val)


def set_signal_handler():
    def handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)


def main():
    set_signal_handler()
    config = parse_config()
    model = get_model(config.model)
    if not sys.stdin.isatty():
        prompt = sys.stdin.read()
        one(model, prompt)
    else:
        chat(model)


if __name__ == "__main__":
    main()
