import mlx.core as mx

def causal_mask(n_queries: int, offset: int) -> mx.array: 
    """ 
    Boolean mask, true where a Query is allowed to attend to a key

    Query i is at absolute position: offset + i; 
    Key j is at position j. 
    Causality means query can see keys at or before its position: 
        (offset + i) >= j

    Returns mask of size (n_queries, offset + n_queries)

    Ex: Attention score before masking: 
                 k1   k2   k3   k4   k5
    q1          2.1  0.3 -1.2  4.5  2.0
    q2          1.1  3.8  0.5 -2.0  1.7
    q3          0.2  1.5  5.2  0.7  2.3

    After masking: 
                 k1   k2   k3   k4   k5
    q1          2.1  -∞   -∞   -∞   -∞
    q2          1.1  3.8  -∞   -∞   -∞
    q3          0.2  1.5  5.2  -∞   -∞

    query only attends to infomation(keys) before its position! 
    """

    # make this into column vec
    q_pos = mx.arange(offset, offset + n_queries)[:, None]
    # make this into row vec, then broadcast then compare
    k_pos = mx.arange(offset + n_queries)[None, :]

    return q_pos >= k_pos 

class KVCache: 
    """ 
    Contiguous KV cache for one layer, grown in blocks

    Everytime we grow the cache by {BLOCK}, instead of re-allocating for every token
    Offset tracks how much of the KV cache is actually valid, and where to append the next KV entry
    """
    BLOCK = 256

    def __init__(self): 
        self.keys = None    # (B, n_kv_heads, capacity, head_dim)
        self.values = None  # (B, n_kv_heads, capacity, head_dim)
        self.offset = 0     # number of vvalid tokens stored 

    def update_and_fetch(self, keys, values): 
        """ 
        Append current decode step's KV and return full history

        keys, values: (B, n_kv_heads, L, head_dim)
        L = prompt length on prefill, 1 for decode
        returns: (B, k_kv_heads, offset + L, head_dim)
        """
        prev = self.offset  # how many valid tokens are already in KV cache (where new KV starts)
        new = keys.shape[2] # either 1 or L, for decode and prefill (how many KV we are adding)\

        # reallocation, only when write would run past end 
        if self.keys is None or prev + new > self.keys.shape[2]: 
            self._grow(keys, values, needed=prev + new)

        # update offset 
        self.offset = prev + new 

        # in-place write into the pre-allocated buffer 
        # MLX supports slice assignment on arrays, it records an op in lazy graph
        # rather than mutating memory immediately 
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values

        # return ONLY the valid region, if we return a bunch of zeros after valid region
        # it would f up the attention mechanism
        return (
            self.keys[..., :self.offset, :], 
            self.values[..., :self.offset, :]
        )

    def _grow(self, keys, values, needed): 
        """ 
        Allocate a bigger buffer and copy the valid KV across
        """
        # learn the layer geometry from tensor shape
        B, n_kv_heads, _, k_head_dim = keys.shape
        # in our model Qwen2 K and V heads dim are equal, but we still read it separately
        v_head_dim = values.shape[3]

        # round up needed to next multiple of BLOCK
        capacity = ((needed + self.BLOCK - 1) // self.BLOCK) * self.BLOCK    

        new_k = mx.zeros((B, n_kv_heads, capacity, k_head_dim), keys.dtype)
        new_v = mx.zeros((B, n_kv_heads, capacity, v_head_dim), values.dtype)

        # copy the old data into new buffer
        if self.keys is not None and self.values is not None: 
            new_k[..., : self.offset, :] = self.keys[..., : self.offset, :]
            new_v[..., : self.offset, :] = self.values[..., : self.offset, :]

        # un-reference the old buffers, MLX frees them 
        self.keys = new_k 
        self.values = new_v
        # NOTE: By this point self.keys and values are guaranteed to be of type mx.array

    def make_mask(self, n_queries, return_array=False, window_size=None):
        """ 
        Creates the attention mask, three cases: 
        1. n_queries = 1 (DECODE)
            There is one query sitting at the newest position, and it is allowed to 
            see the entire cache. Nothing to mask. Returning None lets attention skips masking. 
            Which saves a lot of time since its generated once per token. 

        2. offset = 0 and n_queries > 1 (FRESH PREFILL)
            A standard lower-triangular mask. Instead of allocating a L x L array, we can let 
            MLX handle it internally, since queries and keys are aligned from index 0. 
            This saves memory: 
            Ex: 2000 token prompt -> that's 4 Million boolean allocation saved

        3. offset != 0 and n_queries > 0 (CHUNKED PREFILL)
            Now we build the real mask, this is not the same as case 2 since queries and keys not aligned.
            WORK IN PROGRESS!!! FOR NOW WE DO CASE 1 & 2
        """

        if n_queries == 1: 
            return None 
        if self.offset == 0 and not return_array: 
            return "causal"
        return causal_mask(n_queries, self.offset)

    def size(self): 
        """ 
        Valid tokens held, not capacity, capacity includes the zero padding as well
        """
        return self.offset 

    @property 
    def nbytes(self): 
        """ 
        How many bytes the KV cache takes up, inclduing padding. 
        Per token, per layer, for Qwen2.5-0.5B:

            2 kv_heads x 64 head_dim x 2 tensors (K and V) x 2 bytes (fp16)
              = 512 bytes

        Across 24 layers that is 12,288 bytes = 12 KB per token. At a 32k
        context: 393 MB of KV cache for a model whose 4-bit weights are
        only ~280 MB. The cache outgrows the model, it grows linearly with
        every concurrent request, and it is allocated in fixed blocks that
        strand memory. That is why a real engine pages it.
        """
        if self.keys is None or self.values is None: 
            return 0

        return self.keys.nbytes + self.values.nbytes 
    