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


class Config:
    def __init__(
        self, tokens: int, model: GPT4All, chat: bool, prompt: str, out: str | None
    ):
        self.model = model
        self.tokens = tokens
        self.prompt = prompt
        self.out = out


def get_config() -> Config:
    parser = argparse.ArgumentParser(
        description="CLI that runs an LLM in the terminal."
    )
    parser.add_argument(
        "-m", "--model", help="Model path. Must be a .gguf file.", required=True
    )
    parser.add_argument(
        "-c",
        "--chat",
        action="store_true",
        help="Enter chat to further prompt (still used stdin for context)",
    )
    parser.add_argument("-o", "--out", help="Save the raw output.")
    parser.add_argument("args", nargs="*")
    parser.add_argument("-t", "--tokens", help="Max number of tokens.", default=1024)
    args = parser.parse_args()

    tokens = args.tokens
    if not isinstance(tokens, int):
        raise TypeError()

    model_path = args.model
    model = init_model(model_path)
    prompt = " ".join(args.args)

    out_path = args.out
    c = Config(tokens, model, bool(args.chat), prompt, out_path)
    return c


def print_stream(stream: Iterable[str], delay: float = 0.02, out: str | None = None):
    console = Console()
    all_tokens = []
    for token in stream:
        all_tokens.append(token)
        md = Markdown("".join(all_tokens))
        console.clear()
        console.print(md)
        time.sleep(delay)

    if out is not None:
        with open(out, "a") as file:
            file.write("".join(all_tokens))


def chat(
    model: GPT4All, max_tokens: int, context: str | None = None, out: str | None = None
):
    sys_prompt = (
        SYSTEM_PROMPT + "\nContext:\n" + context
        if context is not None
        else SYSTEM_PROMPT
    )
    with model.chat_session(system_prompt=sys_prompt):
        while True:
            prompt = str(input(">> "))
            stream = model.generate(
                prompt,
                max_tokens=max_tokens,
                streaming=True,
            )
            print_stream(stream, out=out)


def one(model: GPT4All, prompt: str, max_tokens: int, out: str | None = None):
    with model.chat_session(system_prompt=SYSTEM_PROMPT):
        stream = model.generate(
            prompt,
            max_tokens=max_tokens,
            streaming=True,
        )
        print_stream(stream, out=out)


def set_signal_handler():
    def handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)


def main():
    set_signal_handler()
    config = get_config()
    if not sys.stdin.isatty():
        prompt = sys.stdin.read() + config.prompt
        one(config.model, prompt, config.tokens, out=config.out)
    else:
        chat(config.model, config.tokens, context=config.prompt, out=config.out)


if __name__ == "__main__":
    main()
