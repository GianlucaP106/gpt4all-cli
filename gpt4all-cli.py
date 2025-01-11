from typing import Iterable
from gpt4all import GPT4All
import argparse
from rich.console import Console
from rich.markdown import Markdown
import sys
import time
import signal
import os


SYSTEM_PROMPT = "You are a general purpose AI chat bot. Output in markdown."


def init_model(path: str) -> GPT4All:
    if not os.path.exists(path):
        raise FileNotFoundError("Path is invalid")

    model_path = os.path.dirname(path)
    model = os.path.basename(path)
    model = GPT4All(
        model,
        model_path=model_path,
        allow_download=False,
        device="gpu",
    )
    return model


def get_model() -> GPT4All:
    parser = argparse.ArgumentParser(
        description="CLI that runs an LLM in the terminal."
    )
    parser.add_argument(
        "-m", "--model", help="Model path. Must be a .gguf file.", required=True
    )
    args = parser.parse_args()
    model_path = args.model
    model = init_model(model_path)
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


def set_signal_handler():
    def handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)


def main():
    set_signal_handler()
    model = get_model()
    if not sys.stdin.isatty():
        prompt = sys.stdin.read()
        one(model, prompt)
    else:
        chat(model)


if __name__ == "__main__":
    main()
