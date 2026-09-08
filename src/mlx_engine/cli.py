import argparse

from .engine import Engine


def main():
    parser = argparse.ArgumentParser(description="mlx-engine")
    parser.add_argument(
        "--model",
        default="models/Qwen2.5-0.5B-Instruct-4bit",
        help="Path to a local MLX model folder",
    )
    parser.add_argument("--prompt", default="Explain LSTM cells")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument(
        "--raw",
        action="store_true",
        # Needed for parity testing: the Phase 0 mlx_lm.generate() baseline
        # ran without a chat template, so comparing against it requires
        # feeding our engine the same untemplated prompt.
        help="Skip the chat template and complete the prompt directly",
    )
    parser.add_argument("--stats", action="store_true", help="Print timing breakdown")
    args = parser.parse_args()

    engine = Engine(args.model)
    output = engine.generate(args.prompt, max_tokens=args.max_tokens, chat=not args.raw)
    print(output)

    if args.stats:
        print("---")
        print(engine.stats)


if __name__ == "__main__":
    main()