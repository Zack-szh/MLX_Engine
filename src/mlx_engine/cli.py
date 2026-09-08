# Command line entry point 
import argparse 

from .engine import Engine

def main(): 
    parser = argparse.ArgumentParser(description="mlx-engine")
    parser.add_argument(
        "--model",
        default="models/Qwen2.5-0.5B-Instruct-4bit",
        help="Path to a local MLX model folder",
    )
    parser.add_argument(
        "--prompt",
        default="Explan LSTM cells",
        help="Prompt to pass",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Tokens to generate",
    )
    args = parser.parse_args()

    engine = Engine(args.model)
    output = engine.generate(args.prompt, max_tokens=args.max_tokens)
    print(output)


if __name__ == "__main__": 
    main()