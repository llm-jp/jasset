"""Order-free CER.

Each reference region is a block. Every block is matched to at most one contiguous span
of the normalised model output, spans do not overlap, and the total error is

    sum of Levenshtein(block, span) over matched blocks
  + sum of |block| over unmatched blocks
  + number of non-space output characters outside every span.

The minimum equals the minimum over block orderings of the edit distance between the
concatenated blocks and the output. Exact minimisation is NP-hard (edit distance with
block moves), so the scorer computes a lower bound (Lagrangian relaxation of the
"each block at most once" constraint) and an upper bound (a feasible assignment). When
ceil(lower) == upper the value is certified exact. Only the upper bound is ever reported
as the score; the relaxation value alone can be negative and must not be reported.

All costs are integers, so E* is an integer and ceil() of the bound is valid.
"""

import math

import numpy as np
from numba import njit

from jasset.metrics.normalize import is_cjk as is_cjk
from jasset.metrics.normalize import normalize_cer

SPEC_VERSION = "reading-metrics v1"
NORMALIZATION_VERSION = "nu_cer v1: NFKC; newline->space; collapse whitespace; keep space only between non-CJK"

_NEG = -1.0e9
_GARBAGE = -1.0e8
_BIG = 1.0e6
_TOL = 1e-7


def levenshtein(a, b):
    from rapidfuzz.distance import Levenshtein

    return Levenshtein.distance(a, b)


@njit(cache=True)
def _relaxed_kernel(chars, first, last, lamp, y, blocked, free, f, Ht):
    """Numba body of _relaxed_dp. Fills f and Ht (Ht[:, 0] must already be _NEG)."""
    N, n = chars.shape[0], y.shape[0]
    hprev = np.empty(N)
    for j in range(1, n + 1):
        fj = f[j - 1]
        for p in range(N):
            a, b = Ht[p, j - 1], fj - lamp[p]
            hprev[p] = a if a > b else b
        close = _NEG
        if blocked[j - 1]:
            for p in range(N):
                Ht[p, j] = _NEG
        else:
            yj = y[j - 1]
            run = 0.0
            for p in range(N):
                d = fj - lamp[p] if first[p] else hprev[p - 1]
                t = d + (2.0 if chars[p] == yj else 1.0)
                if hprev[p] > t:
                    t = hprev[p]
                if first[p] or t > run:
                    run = t
                Ht[p, j] = run if run >= _GARBAGE else _NEG
            for k in range(last.shape[0]):
                if Ht[last[k], j] > close:
                    close = Ht[last[k], j]
        g = fj + free[j - 1]
        f[j] = g if g > close else close


@njit(cache=True)
def _rows_kernel(row0, bchars, y, blocked):
    """Semi-global rows of one block against the output; blocked characters cannot be
    consumed and horizontal moves restart after them."""
    L, n = bchars.shape[0], y.shape[0]
    rows = np.empty((L + 1, n + 1))
    rows[0, :] = row0
    for i in range(1, L + 1):
        ch = bchars[i - 1]
        run = 0.0
        for j in range(n + 1):
            pv = rows[i - 1, j]
            if j == 0:
                t = pv
            else:
                if blocked[j - 1]:
                    dg = _NEG
                else:
                    dg = rows[i - 1, j - 1] + (2.0 if y[j - 1] == ch else 1.0)
                t = dg if dg > pv else pv
            if j == 0 or blocked[j - 1] or t > run:
                run = t
            rows[i, j] = run if run >= _GARBAGE else _NEG
    return rows


class _Instance:
    """Blocks (unique strings with multiplicity) and normalised output."""

    def __init__(self, blocks, multiplicity, output, blocked=None):
        if len(blocks) != len(multiplicity):
            raise ValueError("blocks and multiplicity differ in length")
        self.blocks = list(blocks)
        self.mult = np.asarray(multiplicity, dtype=np.int64)
        self.output = output
        self.n = len(output)
        self.K = len(blocks)
        self.lengths = np.array([len(b) for b in blocks], dtype=np.int64)
        if self.K and self.lengths.min() < 1:
            raise ValueError("Empty block")
        self.N_total = int((self.lengths * self.mult).sum())
        self.chars = np.array([ord(c) for b in blocks for c in b], dtype=np.int64)
        self.seg = (
            np.repeat(np.arange(self.K), self.lengths) if self.K else np.zeros(0, int)
        )
        self.first = np.zeros(len(self.chars), dtype=bool)
        self.start = np.zeros(self.K, dtype=np.int64)
        pos = 0
        for k, length in enumerate(self.lengths):
            self.first[pos] = True
            self.start[k] = pos
            pos += length
        self.last = self.start + self.lengths - 1
        self.y = np.array([ord(c) for c in output], dtype=np.int64)
        self.blocked = (
            np.zeros(self.n, dtype=bool)
            if blocked is None
            else np.asarray(blocked, dtype=bool)
        )
        space = np.array([c == " " for c in output], dtype=bool)
        self.free = (space | self.blocked).astype(np.float64)
        self.Ht = None  # reused table of _relaxed_dp


def _relaxed_dp(inst, lam):
    """Relaxed DP: blocks may be reused. Returns (f, Ht) where Ht is the pre-open table."""
    N, n = len(inst.chars), inst.n
    f = np.zeros(n + 1)
    if N == 0:
        for j in range(1, n + 1):
            f[j] = f[j - 1] + inst.free[j - 1]
        return f, np.full((0, n + 1), _NEG)
    if inst.Ht is None:
        inst.Ht = np.empty((N, n + 1))
    Ht = inst.Ht
    Ht[:, 0] = _NEG
    _relaxed_kernel(
        inst.chars,
        inst.first,
        inst.last,
        np.asarray(lam, dtype=np.float64)[inst.seg],
        inst.y,
        inst.blocked,
        inst.free,
        f,
        Ht,
    )
    return f, Ht


def _backtrack(inst, lam, f, Ht):
    """Recover one optimal relaxed assignment. Returns (spans, outside positions).

    Each span is dict(k, s, e, match, sub, dele, ins). Ties prefer leaving output
    characters outside spans and opening blocks as late as possible.
    """
    n = inst.n
    spans, outside = [], []
    j = n

    def H(k, i, jj):  # H after the open step, i >= 1
        return max(Ht[inst.start[k] + i - 1, jj], f[jj] - lam[k])

    while j > 0:
        if f[j] <= f[j - 1] + inst.free[j - 1] + _TOL:
            outside.append(j - 1)
            j -= 1
            continue
        closes = Ht[inst.last, j] if inst.K else np.zeros(0)
        k = int(np.argmax(closes))
        if not abs(closes[k] - f[j]) <= _TOL:
            raise RuntimeError("Backtrack failed at boundary")
        i, jj, mode = int(inst.lengths[k]), j, "Ht"
        ops = dict(k=k, s=None, e=j, match=0, sub=0, dele=0, ins=0)
        block = inst.blocks[k]
        while True:
            if mode == "H":
                if i == 0 or f[jj] - lam[k] > Ht[inst.start[k] + i - 1, jj] + _TOL:
                    ops["dele"] += i
                    ops["s"] = jj
                    break
                mode = "Ht"
            v = Ht[inst.start[k] + i - 1, jj]
            if jj == 0:
                raise RuntimeError("Backtrack reached column 0 inside a block")
            diag_pred = f[jj - 1] - lam[k] if i == 1 else H(k, i - 1, jj - 1)
            same = block[i - 1] == inst.output[jj - 1]
            score = 2.0 if same else 1.0
            if not inst.blocked[jj - 1] and abs(diag_pred + score - v) <= _TOL:
                ops["match" if same else "sub"] += 1
                i, jj, mode = i - 1, jj - 1, "H"
            elif not inst.blocked[jj - 1] and abs(H(k, i, jj - 1) - v) <= _TOL:
                ops["ins"] += 1
                jj, mode = jj - 1, "H"
            elif i >= 2 and abs(Ht[inst.start[k] + i - 2, jj] - v) <= _TOL:
                ops["dele"] += 1
                i, mode = i - 1, "Ht"
            else:
                raise RuntimeError("Backtrack failed inside block")
        spans.append(ops)
        j = ops["s"]
    spans.reverse()
    outside.reverse()
    return spans, outside


def _uses(inst, spans):
    uses = np.zeros(inst.K, dtype=np.int64)
    for sp in spans:
        uses[sp["k"]] += 1
    return uses


def _cost(inst, spans):
    """Cost of a feasible assignment on the original (unblocked) output."""
    uses = _uses(inst, spans)
    if (uses > inst.mult).any():
        raise ValueError("Infeasible assignment")
    covered = np.zeros(inst.n, dtype=bool)
    for sp in spans:
        if covered[sp["s"] : sp["e"]].any():
            raise ValueError("Overlapping spans")
        covered[sp["s"] : sp["e"]] = True
    space = np.array([c == " " for c in inst.output], dtype=bool)
    sub = sum(sp["sub"] for sp in spans)
    dele = sum(sp["dele"] for sp in spans) + int(
        ((inst.mult - uses) * inst.lengths).sum()
    )
    ins = sum(sp["ins"] for sp in spans) + int((~covered & ~space).sum())
    return dict(E=sub + dele + ins, S=sub, D=dele, I=ins)


def _gain(sp):
    return 2 * sp["match"] + sp["sub"]


def _best_span(block, inst, blocked):
    """Best single span of `block` inside the unblocked output. Returns span dict or None."""
    n, y = inst.n, inst.y
    L = len(block)
    if n == 0:
        return None
    colblock = np.concatenate([[False], blocked])
    rows = _rows_kernel(
        np.zeros(n + 1), _codes(block), y, np.asarray(blocked, dtype=np.bool_)
    )
    e = int(np.argmax(rows[L]))
    gain = rows[L][e]
    if gain <= _TOL:
        return None
    ops = dict(k=None, s=None, e=e, match=0, sub=0, dele=0, ins=0)
    i, j = L, e
    while i > 0:
        v = rows[i][j]
        same = j > 0 and block[i - 1] == inst.output[j - 1]
        if (
            j > 0
            and not colblock[j]
            and abs(rows[i - 1][j - 1] + (2.0 if same else 1.0) - v) <= _TOL
        ):
            ops["match" if same else "sub"] += 1
            i, j = i - 1, j - 1
        elif abs(rows[i - 1][j] - v) <= _TOL:
            ops["dele"] += 1
            i -= 1
        elif j > 0 and not colblock[j] and abs(rows[i][j - 1] - v) <= _TOL:
            ops["ins"] += 1
            j -= 1
        else:
            raise RuntimeError("Best-span backtrack failed")
    # trailing zero-gain insertions at the start are impossible (H[0][j] = 0 for all j)
    ops["s"] = j
    return ops


def _greedy(inst, fixed=(), remaining=None, blocked=None):
    """Greedy completion: repeatedly fix the (block, span) pair with the largest gain."""
    spans = list(fixed)
    mult = inst.mult.copy() if remaining is None else np.asarray(remaining).copy()
    blocked = inst.blocked.copy() if blocked is None else blocked.copy()
    best = {}
    stale = set(k for k in range(inst.K) if mult[k] > 0)
    while True:
        for k in stale:
            best[k] = _best_span(inst.blocks[k], inst, blocked) if mult[k] > 0 else None
        stale = set()
        candidates = [
            (_gain(sp), -k, k)
            for k, sp in best.items()
            if sp is not None and mult[k] > 0
        ]
        if not candidates:
            return spans
        _, _, k = max(candidates)
        sp = best[k]
        if blocked[sp["s"] : sp["e"]].any():
            stale.add(k)
            continue
        spans.append(dict(sp, k=k))
        blocked[sp["s"] : sp["e"]] = True
        mult[k] -= 1
        stale.add(k)
        for other, osp in best.items():
            if (
                osp is not None
                and other != k
                and osp["e"] > sp["s"]
                and osp["s"] < sp["e"]
            ):
                stale.add(other)


def _repair(inst, spans):
    """Keep spans of blocks within multiplicity and the best uses of over-used blocks,
    then complete greedily for the remaining multiplicity."""
    uses = _uses(inst, spans)
    fixed = []
    for k in range(inst.K):
        mine = [sp for sp in spans if sp["k"] == k]
        if uses[k] > inst.mult[k]:
            mine = sorted(mine, key=_gain, reverse=True)[: int(inst.mult[k])]
        fixed.extend(mine)
    blocked = inst.blocked.copy()
    for sp in fixed:
        blocked[sp["s"] : sp["e"]] = True
    remaining = inst.mult - _uses(inst, fixed)
    return _greedy(inst, fixed, remaining, blocked)


def length_bound(N, output):
    """Lower bound on E* from lengths alone.

    Every non-space output character is either outside all spans (one insertion) or
    inside a span, whose edit distance is at least its length minus the block length;
    every reference character is unmatched or needs an output character. Hence
    E* >= (non-space output characters) - N and E* >= N - (output length).
    """
    nonspace = sum(c != " " for c in output)
    return max(0, nonspace - N, N - len(output))


def solve(
    blocks,
    multiplicity,
    output,
    max_iterations=30,
    theta=1.0,
    exact_limit=12,
    stop_at=None,
):
    """Lower/upper bounds for E* with a feasible assignment. Returns a dict.

    With ``stop_at``, the search ends as soon as the lower bound reaches it (used for
    CER clipped at 1, where stop_at = N); the upper bound is then not refined and may
    be absent (``E_ub`` None) when the length bound alone decides.
    """
    inst = _Instance(blocks, multiplicity, output)
    N, n = inst.N_total, inst.n
    total_blocks = int(inst.mult.sum())
    lb0 = length_bound(N, output)
    if stop_at is not None and lb0 >= stop_at:
        return dict(
            E_ub=None,
            LB=float(lb0),
            LB_int=lb0,
            certified=False,
            relaxed_value=None,
            iterations=0,
            method="length_bound",
            S=None,
            D=None,
            I=None,
            spans=[],
            N=N,
            n=n,
            stopped=True,
        )
    if total_blocks <= exact_limit:
        expanded = [k for k in range(inst.K) for _ in range(int(inst.mult[k]))]
        value, spans = exact_bitmask(
            [inst.blocks[k] for k in expanded], output, with_spans=True
        )
        spans = [dict(sp, k=expanded[sp["k"]]) for sp in spans]
        cost = _cost(inst, spans)
        if cost["E"] != value:
            raise RuntimeError("Exact DP value and recovered assignment disagree")
        return dict(
            E_ub=value,
            LB=float(value),
            LB_int=value,
            certified=True,
            relaxed_value=None,
            iterations=0,
            method="exact",
            S=cost["S"],
            D=cost["D"],
            I=cost["I"],
            spans=spans,
            N=N,
            n=n,
            stopped=False,
        )
    lam = np.zeros(inst.K)
    greedy = _greedy(inst)
    best_ub, best_spans = _cost(inst, greedy)["E"], greedy
    best_lb, best_lb_spans = float(lb0), None
    relax_value = None
    iterations = 0
    stall = 0
    for it in range(max_iterations):
        iterations = it + 1
        f, Ht = _relaxed_dp(inst, lam)
        spans, _ = _backtrack(inst, lam, f, Ht)
        lb = N + n - f[n] - float((inst.mult * lam).sum())
        if it == 0:
            relax_value = lb
        if lb > best_lb + _TOL:
            best_lb, best_lb_spans, stall = lb, spans, 0
        else:
            stall += 1
            if stall >= 3:
                theta, stall = theta / 2, 0
        uses = _uses(inst, spans)
        feasible = spans if (uses <= inst.mult).all() else _repair(inst, spans)
        cost = _cost(inst, feasible)
        if cost["E"] < best_ub:
            best_ub, best_spans = cost["E"], feasible
        if math.ceil(best_lb - _TOL) >= best_ub:
            break
        if stop_at is not None and math.ceil(best_lb - _TOL) >= stop_at:
            break
        g = (uses - inst.mult).astype(np.float64)
        norm = float((g * g).sum())
        if norm == 0.0:
            break
        step = theta * (best_ub - lb) / norm
        lam = np.maximum(0.0, lam + step * g)
    stopped = stop_at is not None and math.ceil(best_lb - _TOL) >= stop_at
    if (
        not stopped
        and math.ceil(best_lb - _TOL) < best_ub
        and best_lb_spans is not None
    ):
        refined = _refine(inst, best_lb_spans)
        if refined is not None:
            cost = _cost(inst, refined)
            if cost["E"] < best_ub:
                best_ub, best_spans = cost["E"], refined
    if not stopped and math.ceil(best_lb - _TOL) < best_ub:
        improved = _local_search(inst, best_spans)
        cost = _cost(inst, improved)
        if cost["E"] < best_ub:
            best_ub, best_spans = cost["E"], improved
    cost = _cost(inst, best_spans)
    lb_int = math.ceil(best_lb - _TOL)
    return dict(
        E_ub=int(best_ub),
        LB=float(best_lb),
        LB_int=int(min(lb_int, best_ub)),
        certified=bool(lb_int >= best_ub),
        relaxed_value=float(relax_value),
        iterations=iterations,
        method="bounds",
        S=cost["S"],
        D=cost["D"],
        I=cost["I"],
        spans=[dict(sp) for sp in best_spans],
        N=N,
        n=n,
        stopped=stopped,
    )


def _extend(inst, row0, block, blocked):
    """Semi-global rows for one block starting from boundary values row0."""
    return _rows_kernel(
        np.asarray(row0, dtype=np.float64),
        _codes(block),
        inst.y,
        np.asarray(blocked, dtype=np.bool_),
    )


def _codes(block):
    return np.array([ord(c) for c in block], dtype=np.int64)


def _walk_rows(rows, block, inst, blocked, e):
    """Backtrack one block's rows from end column e. Returns span dict (k unset)."""
    colblock = np.concatenate([[False], blocked])
    ops = dict(k=None, s=None, e=int(e), match=0, sub=0, dele=0, ins=0)
    i, j = len(block), int(e)
    while i > 0:
        v = rows[i][j]
        same = j > 0 and block[i - 1] == inst.output[j - 1]
        if (
            j > 0
            and not colblock[j]
            and abs(rows[i - 1][j - 1] + (2.0 if same else 1.0) - v) <= _TOL
        ):
            ops["match" if same else "sub"] += 1
            i, j = i - 1, j - 1
        elif abs(rows[i - 1][j] - v) <= _TOL:
            ops["dele"] += 1
            i -= 1
        elif j > 0 and not colblock[j] and abs(rows[i][j - 1] - v) <= _TOL:
            ops["ins"] += 1
            j -= 1
        else:
            raise RuntimeError("Row backtrack failed")
    ops["s"] = j
    return ops


def exact_bitmask(blocks, output, blocked=None, max_blocks=14, with_spans=False):
    """Exact E* by DP over subsets of individual blocks (multiplicity expanded).

    `blocked` marks output positions that no span may contain (they cost nothing outside).
    With `with_spans`, also returns the optimal spans (k indexes `blocks`).
    """
    if len(blocks) > max_blocks:
        raise ValueError("Too many blocks for the exact DP")
    inst = _Instance(blocks, [1] * len(blocks), output, blocked)
    blocked = inst.blocked
    n, K = inst.n, len(blocks)
    F = np.concatenate([[0.0], np.cumsum(inst.free)])
    g = {0: F.copy()}
    cand = {}
    order = sorted(range(1 << K), key=lambda m: (bin(m).count("1"), m))
    for mask in order:
        if mask:
            c = cand.pop(mask)
            g[mask] = np.maximum.accumulate(c - F) + F
        for k in range(K):
            if mask >> k & 1:
                continue
            H = _extend(inst, g[mask], blocks[k], blocked)[-1]
            nxt = mask | 1 << k
            cand[nxt] = np.maximum(cand[nxt], H) if nxt in cand else H
    full = (1 << K) - 1
    value = int(round(sum(len(b) for b in blocks) + n - g[full][n]))
    if not with_spans:
        return value
    spans, mask, j = [], full, n
    while mask:
        if j > 0 and abs(g[mask][j] - (g[mask][j - 1] + inst.free[j - 1])) <= _TOL:
            j -= 1
            continue
        found = False
        for k in range(K):
            if not mask >> k & 1:
                continue
            rows = _extend(inst, g[mask ^ 1 << k], blocks[k], blocked)
            if abs(rows[-1][j] - g[mask][j]) <= _TOL:
                sp = _walk_rows(rows, blocks[k], inst, blocked, j)
                sp["k"] = k
                spans.append(sp)
                mask ^= 1 << k
                j = sp["s"]
                found = True
                break
        if not found:
            raise RuntimeError("Exact backtrack failed")
    spans.reverse()
    return value, spans


def _exact_spans_on_instance(inst, mult, blocked, fixed):
    """Solve the remaining multiplicity exactly on a blocked instance; None if too large."""
    expanded = [k for k in range(inst.K) for _ in range(int(mult[k]))]
    if len(expanded) > 10:
        return None
    if not expanded:
        return list(fixed)
    value, spans = exact_bitmask(
        [inst.blocks[k] for k in expanded], inst.output, blocked, with_spans=True
    )
    return list(fixed) + [dict(sp, k=expanded[sp["k"]]) for sp in spans]


def _local_search(inst, spans, max_passes=5):
    """1-opt: re-place each block's span given the others; add spans for unused multiplicity."""
    spans = [dict(sp) for sp in spans]
    for _ in range(max_passes):
        improved = False
        blocked = inst.blocked.copy()
        for sp in spans:
            blocked[sp["s"] : sp["e"]] = True
        for idx in sorted(range(len(spans)), key=lambda i: _gain(spans[i])):
            sp = spans[idx]
            blocked[sp["s"] : sp["e"]] = False
            best = _best_span(inst.blocks[sp["k"]], inst, blocked)
            if best is not None and _gain(best) > _gain(sp):
                spans[idx] = dict(best, k=sp["k"])
                improved = True
            blocked[spans[idx]["s"] : spans[idx]["e"]] = True
        remaining = inst.mult - _uses(inst, spans)
        for k in range(inst.K):
            while remaining[k] > 0:
                best = _best_span(inst.blocks[k], inst, blocked)
                if best is None:
                    break
                spans.append(dict(best, k=k))
                blocked[best["s"] : best["e"]] = True
                remaining[k] -= 1
                improved = True
        if not improved:
            break
    return spans


def _refine(inst, spans):
    """Fix blocks used exactly at multiplicity; solve the conflicting rest exactly."""
    uses = _uses(inst, spans)
    fixed = [sp for sp in spans if uses[sp["k"]] == inst.mult[sp["k"]]]
    blocked = inst.blocked.copy()
    for sp in fixed:
        blocked[sp["s"] : sp["e"]] = True
    remaining = inst.mult - _uses(inst, fixed)
    exact = _exact_spans_on_instance(inst, remaining, blocked, fixed)
    if exact is not None:
        return exact
    return _greedy(inst, fixed, remaining, blocked)


def order_free_cer(references, prediction, max_iterations=30, detail=False, clip=False):
    """Per-image record from raw region strings and the raw model output.

    ``cer_clipped`` is min(1, E*/N) as reported (upper bound, clipped). With ``clip`` the
    search stops once the lower bound reaches N, since the clipped value is then 1; the
    unclipped ``cer`` and the S/D/I counts may be None for such images.
    """
    if not isinstance(references, list) or any(
        not isinstance(t, str) for t in references
    ):
        raise ValueError("A list of region reference strings is required")
    normalized = [normalize_cer(t) for t in references]
    blocks = [t for t in normalized if t]
    if not blocks:
        raise ValueError("Nonempty normalized reference blocks required")
    unique = sorted(set(blocks))
    mult = [blocks.count(b) for b in unique]
    Y = normalize_cer(prediction)
    result = solve(
        unique, mult, Y, max_iterations, stop_at=sum(map(len, blocks)) if clip else None
    )
    spans = result.pop("spans")
    N = result["N"]
    lb_int = result["LB_int"]
    if result["E_ub"] is None:
        cer_clipped, clipped_exact = 1.0, True
    else:
        cer_clipped = min(1.0, result["E_ub"] / N)
        clipped_exact = result["certified"] or lb_int >= N
    clipped_gap = cer_clipped - min(1.0, lb_int / N)
    block_exact = sum(1 for sp in spans if sp["sub"] + sp["dele"] + sp["ins"] == 0)
    record = dict(
        N=result["N"],
        n=result["n"],
        K=len(blocks),
        block_types=len(unique),
        empty_regions=len(references) - len(blocks),
        E_ub=result["E_ub"],
        LB=result["LB"],
        LB_int=lb_int,
        certified=result["certified"],
        cer=None if result["E_ub"] is None else result["E_ub"] / N,
        cer_clipped=cer_clipped,
        clipped_exact=bool(clipped_exact),
        clipped_gap=clipped_gap,
        stopped=result["stopped"],
        S=result["S"],
        D=result["D"],
        I=result["I"],
        block_exact=block_exact,
        blocks=len(blocks),
        iterations=result["iterations"],
        method=result["method"],
        relaxed_value=result["relaxed_value"],
        spec_version=SPEC_VERSION,
        normalization=NORMALIZATION_VERSION,
    )
    if detail:
        record["spans"] = [dict(sp, block=unique[sp["k"]]) for sp in spans]
    return record
