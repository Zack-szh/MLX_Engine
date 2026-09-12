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


def apply_min_p(logprobs: mx.array, p: float): 
    """ 
    Keep tokens at least p times as likely as the most likely token

    This is a relative threshold rather than an absolute one -> self-scaling:
    
    When the top token has probability 0.9, min_p=0.05 admits
    only tokens above 0.045 (very few). When the top token has 0.1, the bar
    drops to 0.005 and many tokens qualify. Confident model -> narrow;
    uncertain model -> wide. Same instinct as top_p, one comparison instead
    of a sort.

    In log space the multiplication becomes an addition, so the whole filter
    is one add and one compare -- no sorting at all:

        prob >= p * max_prob
        log(prob) >= log(p) + log(max_prob)

    """
    threshold = mx.max(logprobs, axis=-1, keepdims=True) + math.log(p)
    return mx.where(logprobs < threshold, NEG_INF, logprobs)


def make_repetition_penalty(penalty: float, window: int = 20):
    """
    Paper: https://arxiv.org/pdf/1909.05858
    Penalizes tokens that appeared in recent context.

    This is the only tool here that attacks the repetition MECHANISM rather
    than adding randomness to escape it, so it stays fully deterministic.
    Measured on our 28-token cycle: greedy alone gives 51 unique tokens out
    of 300; greedy + penalty 1.1 gives 162, with no sampling noise at all.

    Returns a PROCESSOR, not a filter. The distinction matters:
      - a filter (top_k/top_p/min_p) takes logprobs and masks some to -inf
      - a processor takes RAW logits and rescales them
    Rescaling changes the shape of the distribution, so it has to happen
    before normalization -- do it after and the result is no longer a valid
    log-probability distribution.

    THE SIGN HANDLING IS DELIBERATE. A logit can be positive or negative and
    we want "less likely" in both cases:

        logit > 0:  divide    ->  4.0 / 1.1 =  3.64   (smaller)
        logit < 0:  multiply  -> -4.0 * 1.1 = -4.40   (more negative)

    Naively multiplying everything by 1/penalty would REWARD negative logits
    by pulling them toward zero -- the opposite of the intent.
    """
    if penalty < 0:
        raise ValueError(f"penalty must be non-negative, got {penalty}")

    def processor(tokens: List[int], logits: mx.array) -> mx.array:
        # Nothing generated yet -- first call, right after prefill.
        if not tokens:
            return logits

        # set() so a token appearing 5 times in the window is penalized once
        # rather than compounding to penalty**5. sorted() only for
        # determinism, so two runs build the same index array.
        # [None] adds the batch axis: (n,) -> (1, n), matching logits (1, V).
        recent = mx.array(sorted(set(tokens[-window:])))[None]

        selected = mx.take_along_axis(logits, recent, axis=-1)
        adjusted = mx.where(selected < 0, selected * penalty, selected / penalty)

        # Scatter the adjusted values back into place. Non-mutating: returns a
        # new array rather than writing through the one _forward handed us.
        return mx.put_along_axis(logits, recent, adjusted, axis=-1)

    return processor


def make_sampler(
    temp: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
    min_p: float = 0.0,
) -> Callable[[mx.array], mx.array]:
    """
    Compose the filters into one callable: logits (1, vocab) -> token id (1,).

    Zero disables each filter, so make_sampler() with no arguments is plain
    greedy -- which keeps the mlx_lm parity test valid unless a caller opts in.

    ORDER MATTERS. Ours is top_k -> top_p -> min_p, each narrowing what the
    previous one left. mlx_lm composes them top_p -> min_p -> top_k, so
    end-to-end sampled output will differ even with the same seed. That is
    fine and expected: the individual filters are byte-identical to theirs
    (verified), which is what makes them testable. Composition order is a
    design choice, not a correctness property.

    TEMPERATURE IS APPLIED LAST, at the draw -- not before filtering.
    Dividing logits by temp < 1 sharpens the distribution, temp > 1 flattens
    it. Doing that BEFORE top_p would change which tokens fall inside the
    nucleus, making the two parameters interact in ways neither is documented
    to. Filtering on the model's honest distribution and only then reshaping
    keeps them independent. (HuggingFace applies temperature first; mlx_lm
    applies it last, as we do.)
    """
    # temp == 0 is not a temperature -- dividing by zero is undefined. It is
    # the conventional spelling of "don't sample, take the argmax", the limit
    # as temp -> 0. Deterministic, which is what makes greedy output
    # byte-comparable against a reference implementation.
    if temp == 0.0:
        return lambda logits: mx.argmax(logits, axis=-1)

    # Built once at construction, not per token. Each entry closes over its
    # parameter so the sampler loop stays free of branching.
    filters = []
    if top_k > 0:
        filters.append(lambda lp: apply_top_k(lp, top_k))
    if 0.0 < top_p < 1.0:
        filters.append(lambda lp: apply_top_p(lp, top_p))
    if min_p > 0.0:
        filters.append(lambda lp: apply_min_p(lp, min_p))

    def sampler(logits: mx.array) -> mx.array:
        # Normalize to log-probabilities. Subtracting logsumexp is log_softmax:
        #     log(exp(x_i) / sum_j exp(x_j)) = x_i - logsumexp(x)
        # argmax and categorical are unaffected by this shift, but min_p is
        # not -- it compares against the max, which is only meaningful on a
        # normalized distribution. So we always normalize and stay consistent.
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        for f in filters:
            logprobs = f(logprobs)

        # categorical draws one index from the distribution these logits
        # imply. Masked entries are -inf, so exp gives exactly 0.0 and they
        # can never be drawn. Scaling by 1/temp here is the temperature.
        return mx.random.categorical(logprobs * (1.0 / temp), axis=-1)

    return sampler