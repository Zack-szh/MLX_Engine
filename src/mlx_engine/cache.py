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

