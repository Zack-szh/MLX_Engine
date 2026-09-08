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
        """ 
        load_model() returns (nn.Module, config_dict), still borrowed from MLX-LM
        we will later replace the model loader with our own impl 
        WORK IN PROGRESS!!! 
        """
        self.model, _config = load_model(Path(model_path))
        self.tokenizer = load_tokenizer(model_path)
        self.stats = Stats()

    def make_cache(self): 
        """ 
        One cache per transformer layer, indexed by layer number
        """

        return [KVCache() for _ in self.model.layers]

    def encode(self, prompt: str, chat: bool = True) -> list[int]: 
        """ 
        text -> token_ids 

        Instruct-tuned models were fine-tuned on a specific conversational
        format and behave badly without it. Qwen2.5 uses ChatML, so
        "Explain LSTM cells" becomes:

            <|im_start|>system
            You are Qwen, created by Alibaba Cloud...<|im_end|>
            <|im_start|>user
            Explain LSTM cells<|im_end|>
            <|im_start|>assistant

        add_generation_prompt=True appends that trailing "<|im_start|>
        assistant", which is what puts the model in answering mode rather
        than continuing the user's sentence. Note this turns a 4-token
        prompt into 33 tokens, mostly boilerplate.

        chat=False skips all of it and does raw completion. We keep the
        option because the Phase 0 baseline was measured without a template.
        """

        if chat: 
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True
            )

        return self.tokenizer.encode(prompt)

    def _forward(self, tokens: mx.array, cache) -> mx.array: 
        """
        One forward pass, returns logits for the last position only

        tokens: (L, ) -> returns (1, vocab_size)  
        For now we do tokens[None] to add the batch dimension, 
        batch_dim = 1 for now, we will handle batching later.
        tokens[None]: (L, ) -> (1, L)

        The model returns (1, L, vocab_size) -- a full probability
        distribution for the token following EVERY position. During prefill
        we discard all but the last. Those earlier logits are not wasted
        work we could have skipped: producing them is how the K/V for those
        positions got computed and cached. We just have no use for the
        predictions themselves. (Training does use them all)

        For Qwen2.5-0.5B the vocab is 151,936, so a 33-token prompt
        produces a (1, 33, 151936) tensor -- about 10 MB in fp16 -- of which
        we keep one row. That is a real cost, and it is why production
        engines pass a "logits index" into the model so the lm_head only
        runs on the positions that matter.
        """

        logits = self.model(tokens[None], cache=cache)   # (1, L, vocab)
        return logits[:, -1, :]     # (1, vocab)

    def generate(self, prompt: str, max_tokens: int = 512, chat: bool = True) -> str: 
        """ 
                            generate()
                               │
                    ┌──────────┴──────────┐
                    │                     │
                PREFILL                DECODE
                    │                     │
            entire prompt at once    one token at a time
                    │                     │
                build KV cache          reuse KV cache
                    │                     │
                    └──────────┬──────────┘
                               │
                            output text
        """

        prompt_ids = self.encode(prompt, chat=chat)
        cache = self.make_cache()

        eos = self.tokenizer.eos_token_ids
        stats = Stats(prompt_tokens=len(prompt_ids))

        # ---------- PREFILL ----------
        t0 = time.perf_counter() 
        logits = self._forward(mx.array(prompt_ids), cache)

        # after getting logits, we find the single highest-scoring token
        token = mx.argmax(logits, axis=-1)

        # IMPORTANT!!! 
        # MLX is LAZY, meaning none of these operations is actually computed, we simply 
        # made a computation graph. .item() is a hard sync point. Since it would have to 
        # wait until the actual values are computed and in place 
        token.item()
        stats.prefill_s = time.perf_counter() - t0 

        # ---------- DECODE ----------
        # one token in, one token out, until we hit max_token limit
        # we never re-read the prompt, simply fetch from cache
        out: list[int] = [] 
        t0 = time.perf_counter() 

        for i in range(max_tokens):
            # sync, and bring token into python from MLX
            tok = token.item()

            # found eos, break 
            if tok in eos: 
                break 
            out.append(tok)

            # reaching token limit, break early
            if i == max_tokens - 1: 
                break 

            logits = self._forward(token, cache)
            # now we update token to the token we just generated/predicted, then loop
            token = mx.argmax(logits, axis=-1)

        stats.decode_s = time.perf_counter() - t0
        stats.generated_tokens = len(out)

        self.stats = stats

        # we hold cache in memory for inspection, real engine should free KV cache here
        self.cache = cache 

        # Batch decoder for now, will implement streaming decoder later 
        # TODO: Impl Streaming decoder
        return self.tokenizer.decode(out)


