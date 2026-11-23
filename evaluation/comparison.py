"""Comparison formulas used once a compressed configuration exists (project plan, section 16).

    Quality retention  QR = Q_compressed / Q_full_cache x 100
    Memory reduction   MR = (M_full_cache - M_compressed) / M_full_cache x 100
    Compression ratio  CR = M_full_cache / M_compressed

Q is a quality score on the SAME samples (e.g. pass@1 or functional-correctness rate);
M is KV-cache bytes (or peak GPU bytes, reported separately). For the full-cache baseline
these are trivially 100 %, 0 % and 1x, and are not reported as results.
"""


def quality_retention(q_compressed, q_baseline):
    if q_compressed is None or q_baseline in (None, 0):
        return None
    return q_compressed / q_baseline * 100.0


def memory_reduction(m_baseline, m_compressed):
    if m_baseline in (None, 0) or m_compressed is None:
        return None
    return (m_baseline - m_compressed) / m_baseline * 100.0


def compression_ratio(m_baseline, m_compressed):
    if m_baseline is None or m_compressed in (None, 0):
        return None
    return m_baseline / m_compressed


def pass_at_1(records):
    """pass@1 with greedy decoding = share of executable code tasks (unit_tests / testgen)
    whose single generated answer is functionally correct. Failed/OOM runs count as wrong."""
    code = [r for r in records if r.get("eval_type") in ("unit_tests", "testgen")]
    if not code:
        return None
    return sum(1 for r in code if r.get("status") == "ok" and r.get("functional_correctness") is True) / len(code)
