from pathlib import Path
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load_tokenizer as _mlx_load_tokenizer

def load_tokenizer(model_path: str) -> TokenizerWrapper:
     # looks for tokenizer.json file under model path
     return _mlx_load_tokenizer(Path(model_path))


