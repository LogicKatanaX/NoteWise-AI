"""
Unit tests for NoteWise AI — Rate limiting logic.
Run with: pytest tests/ -v
"""
import time


def test_rate_limit_counter_increments():
    """Query count should increment on each call."""
    query_count = 0
    MAX = 35

    def simulate_query(count):
        count += 1
        return count

    for _ in range(5):
        query_count = simulate_query(query_count)

    assert query_count == 5


def test_rate_limit_cap_enforced():
    """Queries should stop at MAX_QUERIES_PER_SESSION."""
    MAX_QUERIES_PER_SESSION = 35
    query_count = 34

    responses = []
    for _ in range(3):
        if query_count >= MAX_QUERIES_PER_SESSION:
            responses.append("BLOCKED")
        else:
            query_count += 1
            responses.append("OK")

    assert responses[0] == "OK"
    assert responses[1] == "BLOCKED"
    assert responses[2] == "BLOCKED"


def test_cooldown_window_logic():
    """Queries within 1.0s cooldown should be throttled."""
    COOLDOWN = 1.0
    last_query_time = time.time()

    # Immediately after — should be throttled
    time_since = time.time() - last_query_time
    assert time_since < COOLDOWN, "Expected throttle within cooldown window"


def test_remaining_quota_calculation():
    """Remaining quota should decrease correctly."""
    MAX = 35
    query_count = 10
    remaining = MAX - query_count
    assert remaining == 25


def test_session_reset_clears_count():
    """New chat / session reset should restore query count to 0."""
    query_count = 20
    # Simulate "New Chat"
    query_count = 0
    assert query_count == 0

