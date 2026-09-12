"""
Sampling: turing a vector of logits into one token id
we will employ various sampling methods, for fun! 
as of 9/12, we are naively choosing argmax(logits) - Greedy decoding
this greedy decoding approach results in text degeneration 
in smaller models we got into a literal cycle

LOG SPACE:
All probabilities we will be working with is in log space. 
Raw probabilities can be tiny and underflow fp16. 
In log space, multiplying probabilities becomes adding logs -> SIMPLE! 
For a masked token, its log prob is -inf, where exp(-inf) = 0

""" 

import math 
from typing import Callable, List, Optional

import mlx.core as mx 

# masked token would get log prob of NEG_INF, exp(-inf) = 0
NEG_INF = -float("inf")

def apply_top_k(logprobs: mx.array, k: int): 
    """ 
    Keeps only k highest-scoring tokens; mask everything else to -inf
    Note that we are not sorting all logprobs, that's too massive O(V log V), where V is vocab size
    Instead we use partition to find top-k in O(V)
    """

    # argpartiton puts smaller values first, hence we pass -logprobs 
    masked_idx = mx.argpartition(-logprobs, kth=k-1, axis=-1)[..., k:]
    # returns the logprobs, where masked_idx has logprob NEG_INF
    return mx.put_along_axis(logprobs, masked_idx, mx.array(NEG_INF, logprobs.dtype), axis=-1)


def apply_top_p(logprobs: mx.array, p: float): 
    """ 
    Nucleus sampling: keep the smallest set whose total probailities >= p
    AKA: Include the most probable logits, until the sum of its logprobs exceed p

    Paper: https://arxiv.org/pdf/1904.09751 
    """
    # NOTE: logprobs is in token_id order

    # sort logprobs in descending order
    order = mx.argsort(-logprobs, axis=-1)
    sorted_lp = mx.take_along_axis(logprobs, order, axis=-1)

    # since the threshold p is probability, not log-probability
    # we need to convert sorted_lp into sorted_prob
    sorted_prob = mx.exp(sorted_lp)
    # then we find cumulative sum of sorted_prob, and only keep the part 
    # where cum_sum <= p 
    # NOTE: we include the token that crosses the threshold
    # therefore we substract each token's self probability, to make it inclusive
    cum_sum = mx.cumsum(sorted_prob, axis=-1) - sorted_prob 
    keep = cum_sum < p

    # for kept tokens: retain original log-probability
    # for discarded token: set to NEG_INF
    # NOTE: survivors is still in rank order, we need to convert back to token_id order
    survivors = mx.where(keep, sorted_lp, NEG_INF)

    out = mx.full(logprobs.shape, NEG_INF, logprobs.dtype)

    # order and survivors have matching index
    # return masked logprob in original token_id order
    return mx.put_along_axis(out, order, survivors, axis=-1)


