import time 
from pathlib import Path
from mlx_lm import generate
import mlx.core as mx
from mlx_lm.utils import load_model
from dataclasses import dataclass
from pathlib import Path

from .cache import KVCache
from .tokenizer import load_tokenizer

@dataclass
class Stats: 
    """ 
    Per-request measurements
    """

    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_s: float = 0.0
    decode_s: float = 0.0

    @property
    def prefill_tps(self) -> float: 
        # guard against divide-by-zero on tiny prefill 
        return self.prompt_tokens / self.prefill_s if self.prefill_s else 0.0

    @property
    def decode_tps(self) -> float: 
        return self.generated_tokens / self.decode_s if self.decode_s else 0.0

    def __str__(self) -> str:
        return (
            f"prompt {self.prompt_tokens} tok in {self.prefill_s * 1000:.1f} ms "
            f"({self.prefill_tps:.1f} tok/s)\n"
            f"decode {self.generated_tokens} tok in {self.decode_s * 1000:.1f} ms "
            f"({self.decode_tps:.1f} tok/s)\n"
            f"peak memory {mx.get_peak_memory() / 1e9:.3f} GB"
        )

class Engine: 
    """ 
    The generation loop, without using mlx_lm.generate() 
    Two phases per request: 
        1. prefill: one forward pass over whole prompt, filling the KV cache
        All L tokens go through together, GPU does large matmuls and saturate 
        compute units -> COMPUTE-BOUND. Cost scales with prompt length L

        2. decode: one forward pass per token, each appending a single KV pair
        L = 1 always. The matmuls are skinny, essentially all weights against the 
        singel vector that is the next token -> MEMORY-BANDWIDTH-BOUND. Cost is roughly
        constant per token. 
    """

    def __init__(self, model_path: str): 
        # load_model() returns (nn.Module, config_dict)
        self.model, _config = load_model(Path(model_path))
        self.tokenizer = load_tokenizer(model_path)

    def generate(self, prompt: str, max_tokens: int = 512) -> str: 
        # use mlx_lm.generate for now as wrapper
        # later replace step by step
        return generate(
            self.model, 
            self.tokenizer, 
            prompt=prompt, 
            max_tokens=max_tokens, 
            verbose=True,
        )
