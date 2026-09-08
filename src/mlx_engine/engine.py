from pathlib import Path
from mlx_lm import generate
from mlx_lm.utils import load_model

from .tokenizer import load_tokenizer

class Engine: 
    """ 
    Our goal is to replace mlx_lm.generate() with our own implementations
    mlx_lm.generate() -> 
    = tokenizer -> prefill -> KV Cache -> Decode Loop -> sample next token -> repeat

    Infernece conceptual steps: 
    tokens -> transformer -> logits -> probabilities -> next token -> repeat... 
    Inference is a loop that continuously predict next token. However doing this naively is 
    very expensive, that's where kv cache comes in. 

    For every token, attention creates Q(Query), K(Key), V(value), example: 
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    softmax(Attention(Q, K, V)) 

    when we generate the next token, we need K and V from all previous tokens
    instead of recalculating, we store them -> kv cache

    Conceptually: 
    KV Cache:
        K:
        [K₁ K₂ K₃ K₄ K₅]
        V:
        [V₁ V₂ V₃ V₄ V₅]
    
        When the next token arrives, we only calculate: 
        Q_6 = x_6 @ Wq
        K_6 = x_6 @ Wk
        V_6 = x_6 @ Wv
        then append 

    KV cache:
        [K₁ K₂ K₃ K₄ K₅ K₆]
        [V₁ V₂ V₃ V₄ V₅ V₆]

    Note that Q is not cached: 
    Q = What the token is looking for (the question)
    K = What information does the token represent
    V = What information should the token give if you pay attention to this token

    Q changes for every new token, and the decoding step does not need the old Q, no need to cache

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
