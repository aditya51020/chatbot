"""
Land Chatbot — Sequential + Concurrent Load Test
Run: python load_test.py

Tests:
  1. 100 sequential queries (baseline latency)
  2. 20 concurrent users (concurrency stress)

No external tools needed — uses stdlib only.
"""

import time
import json
import statistics
import threading
import urllib.request
import urllib.error

API_URL  = "http://localhost:8000/chat"
TIMEOUT  = 30   # seconds per request

# Representative query mix (structured + hybrid paths)
QUERIES = [
    "SHREE BALAJI ARTH MOVERS ji ne kabhi bid kiya hai ?",
    "CS.pdf mein tender amount kya tha?",
    "agency name kya hai?",
    "maturity date kya hai?",
    "agency ka naam batao",
    "bid amount kitna tha?",
    "contractor ka naam?",
    "ref number kya hai?",
    "casting date kab thi?",
    "schedule discount kya tha?",
] * 10   # 100 total


def _post_query(query: str) -> tuple[float, bool]:
    """Returns (latency_ms, success)."""
    payload = json.dumps({"query": query}).encode()
    req     = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            # consume streaming NDJSON
            for _ in resp:
                pass
        ms = (time.perf_counter() - t0) * 1000
        return ms, True
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        return ms, False


def run_sequential(queries: list[str]) -> dict:
    print(f"\n{'='*50}")
    print(f"SEQUENTIAL TEST — {len(queries)} queries")
    print('='*50)
    latencies = []
    errors = 0
    for i, q in enumerate(queries, 1):
        ms, ok = _post_query(q)
        latencies.append(ms)
        if not ok:
            errors += 1
        if i % 10 == 0:
            print(f"  {i}/{len(queries)}  avg so far: {statistics.mean(latencies):.0f}ms")

    return {
        "total":      len(queries),
        "errors":     errors,
        "min_ms":     round(min(latencies)),
        "max_ms":     round(max(latencies)),
        "avg_ms":     round(statistics.mean(latencies)),
        "p50_ms":     round(statistics.median(latencies)),
        "p95_ms":     round(sorted(latencies)[int(len(latencies) * 0.95)]),
        "p99_ms":     round(sorted(latencies)[int(len(latencies) * 0.99)]),
    }


def run_concurrent(queries: list[str], users: int = 20) -> dict:
    print(f"\n{'='*50}")
    print(f"CONCURRENT TEST — {users} users × {len(queries)//users} queries")
    print('='*50)
    results: list[tuple[float, bool]] = []
    lock = threading.Lock()

    def worker(chunk):
        for q in chunk:
            r = _post_query(q)
            with lock:
                results.append(r)

    chunks = [queries[i::users] for i in range(users)]
    threads = [threading.Thread(target=worker, args=(c,)) for c in chunks]

    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total_wall = (time.perf_counter() - t_start) * 1000

    latencies = [r[0] for r in results]
    errors    = sum(1 for r in results if not r[1])

    return {
        "users":          users,
        "total":          len(results),
        "errors":         errors,
        "wall_ms":        round(total_wall),
        "throughput_qps": round(len(results) / (total_wall / 1000), 2),
        "avg_ms":         round(statistics.mean(latencies)),
        "p50_ms":         round(statistics.median(latencies)),
        "p95_ms":         round(sorted(latencies)[int(len(latencies) * 0.95)]),
        "p99_ms":         round(sorted(latencies)[int(len(latencies) * 0.99)]),
    }


if __name__ == "__main__":
    print("Land Chatbot Load Test")
    print(f"Target: {API_URL}")

    # Warmup
    print("\nWarming up (2 queries)...")
    _post_query(QUERIES[0])
    _post_query(QUERIES[1])

    seq  = run_sequential(QUERIES)
    conc = run_concurrent(QUERIES, users=20)

    print(f"\n{'='*50}")
    print("RESULTS SUMMARY")
    print('='*50)
    print("\n[Sequential — 100 queries]")
    for k, v in seq.items():
        print(f"  {k:<15}: {v}")

    print("\n[Concurrent — 20 users]")
    for k, v in conc.items():
        print(f"  {k:<15}: {v}")

    # Save report
    report = {"sequential": seq, "concurrent": conc,
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open("load_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nReport saved: load_test_report.json")

    # Pass/fail verdict
    print("\n[VERDICT]")
    passed = True
    if seq["p95_ms"] > 3000:
        print(f"  ❌ Sequential p95 ({seq['p95_ms']}ms) > 3000ms threshold")
        passed = False
    if conc["p95_ms"] > 5000:
        print(f"  ❌ Concurrent p95 ({conc['p95_ms']}ms) > 5000ms threshold")
        passed = False
    if seq["errors"] > 0 or conc["errors"] > 0:
        print(f"  ❌ Errors: seq={seq['errors']}, conc={conc['errors']}")
        passed = False
    if passed:
        print("  ✅ All thresholds passed — system is load-ready")
