# Hope Trading Engine — Code Audit Report

## Critical Issues (HIGH) — Fix Immediately

### 1. Ring Buffer Off-by-One in `tick_processor.rs` (L186-201)
**Severity:** HIGH  
**Impact:** Reads uninitialized memory during buffer warm-up (first 10-50 ticks), corrupts volatility/drift estimates, triggers false signals at market open.  
**Root Cause:** Index calculation doesn't account for buffer initialization phase.  
**Fix:** Add guard: `if tick_count < window_size { return default; }`

### 2. Floating-Point Drift Accumulation (L237-247)
**Severity:** HIGH  
**Impact:** Naive summation of squared returns without Kahan summation. Error compounds after 1000 ticks, diverging volatility by 0.5-2%.  
**Root Cause:** No numerically stable variance computation.  
**Fix:** Use Welford's online algorithm or Kahan summation for `sum_sq_diff`.

### 3. Transformer Race Condition (transformer.rs L120)
**Severity:** HIGH  
**Impact:** Inference thread writes `latest_prob` without synchronization. Main thread reads stale probabilities (5-10 ticks delayed), drifting signal decisions.  
**Root Cause:** `ArcSwap::store()` is single-writer-safe but reader timing isn't coordinated.  
**Fix:** Add `RwLock` or timestamp-based staleness check before read.

### 4. Cooldown Counter Off-by-One (engine.rs L265-266)
**Severity:** HIGH  
**Impact:** `saturating_sub(1)` before state transition allows trades 1 tick early when cooldown expires.  
**Root Cause:** Decrement-then-check logic inverted.  
**Fix:** Check cooldown first, then decrement on next tick.

### 5. Stale Proposal Timeout (engine.rs L331-338)
**Severity:** HIGH  
**Impact:** Proposal freshness checked at tick T but quote received at T-2. Broker may have invalidated it; buy commands fail silently.  
**Root Cause:** Timestamp logic doesn't account for network latency.  
**Fix:** Check `quote.tick_id >= proposal.tick_id` not just recency.

### 6. Contract Timeout Memory Leak (engine.rs L283-297)
**Severity:** HIGH  
**Impact:** Zombie contracts accumulate in `tracked_contracts` if timeout fires before `OpenContract(closed)` arrives. Memory leaks linearly.  
**Root Cause:** Timeout removes entry but never completes subscription cleanup.  
**Fix:** Send `forget_all()` on timeout, don't just delete.

---

## Medium Issues — Schedule Fix

### 7. RiskManager Loss Streak Not Reset (risk.rs L45)
- Break-even trades don't reset the loss streak counter
- Fix: Check if PnL > 0, not just PnL != 0

### 8. Reversal Tracking Ambiguity (tick_processor.rs L120-135)
- Flat ticks (price == prev_price) cause direction=0, ambiguous reversal detection
- Fix: Carry over previous non-flat direction or skip flat bars

### 9. Reconnect Backoff Too Aggressive (websocket_client.rs L380-390)
- Backoff resets on every attempted reconnect, not just successful ones
- Fix: Reset only on successful connection

### 10. Transformer Cache TTL (transformer.rs L95-105)
- No staleness check on cached inference probability
- Fix: Add age check, recompute if > 5 ticks old

### 11. Training Purge Gap Edge Case (train_fixed.py L225)
- Purge gap = seq_len + 10 only if target horizon is fixed
- Fix: Assert `target_quote_future = quote.shift(-10)` in code

---

## Low Issues — Polish

### 12. Config Silent Failures (config.rs L120-135)
- `.parse::<bool>()` silently defaults to false on invalid input
- Fix: Return `Err` or log warning

### 13. Export Corruption Risk (export_db.py L180-190)
- Partial writes if script crashes mid-transaction
- Fix: Use atomic rename on completion

### 14. Backtest Subprocess Timeout (grid_backtest.py L450)
- 300s timeout too short for large datasets
- Fix: Make configurable or use heartbeat mechanism

### 15. Inconsistent Crypto Library (sign_model.py L25 vs train_fixed.py L162)
- Both use `cryptography` but sign_model imports RSA (unused)
- Fix: Audit and deduplicate

### 16. Silent Tick Collection Skips (tick_collector.py L420-430)
- Malformed ticks silently dropped without logging
- Fix: Log every skip with reason

---

## Strengths

✓ FSM state machine well-defined with comprehensive validation  
✓ WebSocket client has robust message routing and heartbeat  
✓ Tick collector implements proper async/exponential backoff  
✓ Strategy engine applies filters correctly with dynamic thresholds  
✓ Recently fixed: clippy warnings, pytest ONNX export  
✓ Recently improved: training data leakage, reproducible DataLoaders  

---

## Recommendations

**Immediate (this week):**
1. Fix #1-6 (HIGH issues) — these affect trading correctness and capital safety
2. Write regression tests for each fix
3. Re-run full test suite and backtest on historical data

**Short-term (2-3 weeks):**
4. Implement #7-11 (MEDIUM issues)
5. Add integration tests with mock Deriv API
6. Profile memory usage under 24h continuous run

**Longer-term:**
7. Add tracing/observability for all state transitions
8. Implement formal property-based testing for FSM
9. Set up fuzz testing for tick processor edge cases
