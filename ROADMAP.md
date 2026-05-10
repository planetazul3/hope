# Hope Trading Engine — Fix Roadmap

This document is a step-by-step implementation guide to resolve every finding
from `AUDIT_REPORT.md`. Each phase is self-contained, has its own tests, and
can be implemented independently by any agent or developer.

**Rules for every phase:**
- Run `cargo fmt`, `cargo clippy --locked --offline --all-targets`, and
  `cargo test --locked --offline` after every Rust change.
- Run `python3 -m pytest tests/` after every Python change.
- Commit with a descriptive message at the end of each phase.
- Do NOT skip phases or combine them — they are ordered by risk.

---

## Phase 1 — CRITICAL: Break-Even Trade Leaves Loss Streak Dangling

**File:** `src/risk.rs`  
**Severity:** HIGH  
**Root cause:** When `profit == 0.0` the `consecutive_losses` counter is never
reset. After a string of losses followed by a break-even trade, the engine
enters cooldown on the very next loss, even though the streak was interrupted.

### Step 1.1 — Open the file
Read `src/risk.rs`, lines 33–54.

### Step 1.2 — Change the condition on line 43
Find this exact block:
```rust
        if profit < 0.0 {
            self.consecutive_losses += 1;
            self.losses += 1;
        } else if profit > 0.0 {
            self.consecutive_losses = 0;
            self.wins += 1;
        }
        // If profit == 0.0, we treat it as a non-win, non-loss trade for streak purposes
```
Replace it with:
```rust
        if profit < 0.0 {
            self.consecutive_losses += 1;
            self.losses += 1;
        } else {
            // Break-even (profit == 0.0) counts as a streak reset — it interrupted
            // the losing run and should not allow cooldown to fire on the next loss.
            self.consecutive_losses = 0;
            if profit > 0.0 {
                self.wins += 1;
            }
        }
```

### Step 1.3 — Add a regression test at the bottom of `src/risk.rs`
Add this test inside `mod tests { ... }` after the existing test:
```rust
    #[test]
    fn break_even_resets_loss_streak() {
        let mut risk = RiskManager::new(3);
        risk.on_trade_closed(-1.0);
        risk.on_trade_closed(-1.0);
        // Break-even should reset the streak
        let outcome = risk.on_trade_closed(0.0);
        assert_eq!(outcome.consecutive_losses, 0);
        assert!(!outcome.enter_cooldown);
        // Next loss starts a fresh streak of 1, not 3
        let outcome = risk.on_trade_closed(-1.0);
        assert_eq!(outcome.consecutive_losses, 1);
        assert!(!outcome.enter_cooldown);
    }
```

### Step 1.4 — Verify
Run: `cargo test --locked --offline risk`  
Expected: `2 passed`.

---

## Phase 2 — CRITICAL: Transformer Race Condition (Stale Probability Read)

**File:** `src/transformer.rs`  
**Severity:** HIGH  
**Root cause:** `ArcSwap<f64>` is used to pass the latest inference probability
from the inference thread to the main thread. `ArcSwap` is correct for
single-writer scenarios, but there is no coordination to detect a stale read —
the main thread can call `probability_up()` before the inference thread has
finished its first run, reading the initial default value of `0.5` for many
ticks. This is already race-condition-safe (no UB), but the staleness is
undetected. We fix it by wrapping the probability with the tick count at which
it was produced, so the caller knows how old the value is.

### Step 2.1 — Add `tick_count` to `TransformerModel`
At the top of `src/transformer.rs`, add this import on the line after
`use std::sync::Arc;`:
```rust
use std::sync::atomic::AtomicU64;
```

### Step 2.2 — Change the struct definition
Find:
```rust
pub struct TransformerModel {
    sequence_length: usize,
    queue: Arc<ArrayQueue<Vec<f32>>>,
    pool: Arc<ArrayQueue<Vec<f32>>>,
    latest_prob: Arc<ArcSwap<f64>>,
    is_running: Arc<AtomicBool>,
    _handle: Option<thread::JoinHandle<()>>,
}
```
Replace with:
```rust
pub struct TransformerModel {
    sequence_length: usize,
    queue: Arc<ArrayQueue<Vec<f32>>>,
    pool: Arc<ArrayQueue<Vec<f32>>>,
    latest_prob: Arc<ArcSwap<f64>>,
    /// Tick counter at which latest_prob was last written by the inference thread.
    prob_tick: Arc<AtomicU64>,
    is_running: Arc<AtomicBool>,
    _handle: Option<thread::JoinHandle<()>>,
}
```

### Step 2.3 — Initialize `prob_tick` in `load()`
Find this line:
```rust
        let latest_prob = Arc::new(ArcSwap::from_pointee(0.5));
```
Replace with:
```rust
        let latest_prob = Arc::new(ArcSwap::from_pointee(0.5));
        let prob_tick: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
```

### Step 2.4 — Clone `prob_tick` before the thread spawn
Find:
```rust
        let pool_clone = Arc::clone(&pool);
        let queue_clone = Arc::clone(&queue);
        let latest_prob_clone = Arc::clone(&latest_prob);
        let is_running_clone = Arc::clone(&is_running);
```
Add one more line at the end of that block:
```rust
        let prob_tick_clone = Arc::clone(&prob_tick);
```

### Step 2.5 — Add a local tick counter in the inference thread
Find this line inside the thread closure:
```rust
                while is_running_clone.load(Ordering::Acquire) {
```
Add a mutable counter on the very next line:
```rust
                let mut inference_count: u64 = 0;
```

### Step 2.6 — Increment the counter and store it after every successful inference
Find:
```rust
                                        latest_prob_clone.store(Arc::new(prob as f64));
```
Replace with:
```rust
                                        latest_prob_clone.store(Arc::new(prob as f64));
                                        inference_count += 1;
                                        prob_tick_clone.store(inference_count, Ordering::Release);
```

### Step 2.7 — Store `prob_tick` in the struct
Find:
```rust
        Ok(Self {
            sequence_length,
            queue,
            pool,
            latest_prob,
            is_running,
            _handle: Some(_handle),
        })
```
Replace with:
```rust
        Ok(Self {
            sequence_length,
            queue,
            pool,
            latest_prob,
            prob_tick,
            is_running,
            _handle: Some(_handle),
        })
```

### Step 2.8 — Expose staleness in `probability_up()`
Find:
```rust
        **self.latest_prob.load()
    }
```
Replace with:
```rust
        let prob = **self.latest_prob.load();
        let ready = self.prob_tick.load(Ordering::Acquire) > 0;
        if ready { prob } else { 0.5 }
    }
```
This ensures the model returns the neutral prior `0.5` until at least one
inference has completed, rather than an uninitialised or stale default.

### Step 2.9 — Verify
Run: `cargo clippy --locked --offline --all-targets`  
Run: `cargo test --locked --offline`  
Expected: no new warnings, all 20 unit tests pass.

---

## Phase 3 — CRITICAL: Contract Timeout Leaves Subscription Orphaned

**File:** `src/engine.rs`  
**Severity:** HIGH  
**Root cause:** When a contract timeout fires (lines 283–297), the code removes
the contract from `tracked_contracts` and calls `safe_reset()`, but does NOT
cancel the open-contract subscription on the Deriv API. Deriv keeps pushing
`proposal_open_contract` messages for that subscription ID forever, filling the
inbound queue.

### Step 3.1 — Understand the context
Read `src/engine.rs` lines 279–308. The timeout block calls `safe_reset()`.
Find the `safe_reset()` method:
```
grep -n "fn safe_reset" src/engine.rs
```
Read that method to understand what it resets.

### Step 3.2 — Find `send_forget` or subscription cancel call
Run:
```
grep -n "forget\|Forget\|unsubscribe\|cancel_sub" src/engine.rs src/execution.rs src/websocket_client.rs
```
Note the exact method/variant name used to cancel an open-contract subscription.

### Step 3.3 — Add subscription cancel on timeout
Find this exact block in `src/engine.rs`:
```rust
                            warn!(?self.active_contract_id, "active contract tracked for too long; forcing clear");
                            if let Some(contract_id) = self.active_contract_id {
                                self.tracked_contracts.write().remove(&contract_id);
                                // ADR 0008: Register forced clear as a loss to prevent risk manager bypass
                                let _ = self.risk.on_trade_closed(-self.config.stake);
                            }
                            self.set_active_contract(None);
                            self.contract_started_at = None;
                            self.pending_subscription_req_id = None;
                            self.safe_reset();
```
After `self.tracked_contracts.write().remove(&contract_id);` and before
`self.set_active_contract(None);`, add a call to send a `forget_all` or
`forget` command using the same mechanism used elsewhere in the engine. The
exact call depends on what `grep` found in Step 3.2. The pattern will look like:
```rust
                                // Cancel the open-contract subscription so Deriv stops
                                // streaming updates for this orphaned contract.
                                let req_id = self.next_req_id();
                                let _ = try_send_forget_all(&mut self.execution, command_tx, req_id);
```
Use the actual method name found in Step 3.2.

### Step 3.4 — Write a test
In `src/engine.rs` tests (or a new integration test in `tests/`), add a test
that verifies a timeout clears all state fields:
- `active_contract_id` is `None`
- `contract_started_at` is `None`
- `pending_subscription_req_id` is `None`

### Step 3.5 — Verify
Run: `cargo test --locked --offline`  
Run: `cargo clippy --locked --offline --all-targets`

---

## Phase 4 — CRITICAL: Stale Proposal Validation Uses Wall Clock Not Tick-ID

**File:** `src/engine.rs`  
**Severity:** HIGH  
**Root cause:** Lines 331–338 compare `tick_started_at.duration_since(ready.received_at)`
against `self.proposal_timeout`. This is a wall-clock check. If the system is
under load and ticks are queued, a proposal that Deriv has already expired
(because its underlying tick advanced) passes this check and a buy is issued,
which Deriv immediately rejects. This wastes a request and can cause the FSM
to stall in `OrderPending`.

### Step 4.1 — Understand `ProposalReady`
Run:
```
grep -n "struct ProposalReady\|ProposalReady {" src/engine.rs
```
Read the fields. Note which field holds the proposal quote and what the quote
contains.

### Step 4.2 — Understand what tick info is available
Look at `TickSnapshot` in `src/tick_processor.rs` for the `epoch` field.
The current `snapshot` (tick being processed) has a fresh epoch.
`ready.quote` has an epoch or ask_price derived when the proposal was received.

### Step 4.3 — Add a max-tick-age guard
Find:
```rust
                if let Some(ready) = self.pending_proposal.take() {
                    if tick_started_at.duration_since(ready.received_at) > self.proposal_timeout {
                        warn!("discarding stale proposal");
```
Change the condition to also check tick age. If `snapshot.epoch` is available
and `ready.received_epoch` is stored, add:
```rust
                if let Some(ready) = self.pending_proposal.take() {
                    let wall_clock_stale =
                        tick_started_at.duration_since(ready.received_at) > self.proposal_timeout;
                    // Also discard if the market has moved more than duration_ticks ticks
                    // since the proposal was priced — Deriv will reject it anyway.
                    let tick_age = snapshot.epoch.saturating_sub(ready.received_epoch);
                    let tick_stale = tick_age > self.config.duration_ticks as u64 * 2;
                    if wall_clock_stale || tick_stale {
                        warn!(
                            wall_clock_stale,
                            tick_stale,
                            tick_age,
                            "discarding stale proposal"
                        );
```

If `ProposalReady` does not yet store `received_epoch`, add that field:
1. Find the `ProposalReady` struct and add `received_epoch: u64`.
2. Where it is constructed, set `received_epoch: snapshot.epoch` (the epoch of
   the tick when the proposal arrived).

### Step 4.4 — Verify
Run: `cargo test --locked --offline`  
Confirm the existing `test_stale_proposal_discard` still passes.

---

## Phase 5 — MEDIUM: Reconnect Backoff Resets on Every Attempt, Not Just Success

**File:** `src/websocket_client.rs`  
**Severity:** MEDIUM  
**Root cause:** Line 249 resets `current_backoff` on `Ok(...)` from
`connect_async`. This is actually correct — the backoff resets only after a
successful TCP connection is made. Reading the code at lines 246–274 confirms
the agent's finding was a false positive. **No change required.** Document this
as verified-correct in the roadmap and move on.

**Action:** Skip this phase. The reconnect backoff is correct as-is.

---

## Phase 6 — MEDIUM: Reversal Tracking Ambiguous on Flat Ticks

**File:** `src/tick_processor.rs`  
**Severity:** MEDIUM  
**Root cause:** Lines 138–148. When `direction == Direction::Flat` (price
unchanged), `last_trend_direction` is NOT updated and `ticks_since_reversal`
is NOT incremented. The streak is forced to 0 (line 124–125). This is correct
behaviour — flat ticks are treated as noise. **No change required.** The code
comment on line 137 documents this intent.

**Action:** Skip this phase. The flat-tick handling is intentional and correct.

---

## Phase 7 — MEDIUM: Transformer Cache Has No Staleness TTL

**File:** `src/transformer.rs`  
**Severity:** MEDIUM  
**Root cause:** If the inference thread stalls or the queue fills up, the main
thread keeps returning the last-seen probability indefinitely. Phase 2 already
detects "never had a result" (returns 0.5 until `prob_tick > 0`). We now add
detection for "result is too old".

### Step 7.1 — Add a call-site counter to `TransformerModel`
After Phase 2 is merged, open `src/transformer.rs`.

Find the struct field `prob_tick: Arc<AtomicU64>` and add one more field below it:
```rust
    /// Number of times probability_up() has been called.
    call_count: Arc<AtomicU64>,
```

### Step 7.2 — Initialize in `load()`
After:
```rust
        let prob_tick: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
```
Add:
```rust
        let call_count: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
```

### Step 7.3 — Store in struct
Add `call_count,` to the `Ok(Self { ... })` block.

### Step 7.4 — Increment and check in `probability_up()`
Replace the end of `probability_up()`:
```rust
        let prob = **self.latest_prob.load();
        let ready = self.prob_tick.load(Ordering::Acquire) > 0;
        if ready { prob } else { 0.5 }
    }
```
With:
```rust
        let call = self.call_count.fetch_add(1, Ordering::Relaxed) + 1;
        let last_inference = self.prob_tick.load(Ordering::Acquire);
        let prob = **self.latest_prob.load();
        // If inference hasn't run yet, return neutral prior.
        if last_inference == 0 {
            return 0.5;
        }
        // If inference is more than 60 calls behind, the inference thread is
        // likely stalled or the queue is full. Return neutral prior so the
        // strategy falls back to the Gaussian filter.
        if call.saturating_sub(last_inference) > 60 {
            warn!("transformer inference is stale by {} calls; returning neutral prior", call - last_inference);
            return 0.5;
        }
        prob
    }
```

### Step 7.5 — Verify
Run: `cargo clippy --locked --offline --all-targets`  
Run: `cargo test --locked --offline`

---

## Phase 8 — LOW: Config Parsing Silently Ignores Invalid Bool Values

**File:** `src/config.rs`  
**Severity:** LOW  
**Root cause:** `parse_or_default` at line 188 already returns an `Err` when
parsing fails (line 196). The agent's finding was incorrect — invalid values
do produce an error. **No change required.**

**Action:** Skip this phase. Verified correct.

---

## Phase 9 — LOW: Export DB Vulnerable to Corrupt Output on Crash

**File:** `scripts/export_db.py`  
**Severity:** LOW  
**Root cause:** Lines 210–235 write chunks to the CSV file directly. If the
process crashes mid-way, the output file is partially written. Readers of
`data/ticks.csv` (including the training pipeline) will silently consume
truncated data.

### Step 9.1 — Use an atomic rename for the final file
Find the section at the end of `export_ticks()`:
```python
        logger.info("Successfully exported to %s", csv_path)
```
Before the `except`/`finally` block, locate where chunks write to `csv_path`
(line 213). Change the write strategy to use a temporary file and rename at
the end.

**At the top of the `export_ticks()` function**, right after the `csv_path`
variable is resolved, add:
```python
    import tempfile, shutil
    tmp_fd, tmp_csv_path = tempfile.mkstemp(
        suffix=".csv.tmp",
        dir=os.path.dirname(csv_path) or ".",
    )
    os.close(tmp_fd)
```

**Replace all writes to `csv_path`** in the chunk loop with `tmp_csv_path`:
```python
                chunk.to_csv(tmp_csv_path, index=False, header=header, mode='a' if not first_chunk else mode, compression=compression)
```

**After the chunk loop succeeds**, atomically replace the target file:
```python
        shutil.move(tmp_csv_path, csv_path)
        logger.info("Successfully exported to %s", csv_path)
```

**In the `except` block**, clean up the temp file:
```python
    except Exception:
        logger.exception("Export failed")
        try:
            os.unlink(tmp_csv_path)
        except OSError:
            pass
        sys.exit(1)
```

### Step 9.2 — Verify
Run: `python3 -m pytest tests/`  
Check that no existing tests break.

---

## Phase 10 — LOW: Silent Tick Drop Leaves No Audit Trail

**File:** `scripts/tick_collector.py`  
**Severity:** LOW  
**Root cause:** Lines 101–107. Ticks where `epoch` or `quote` is not a valid
number, or where `quote <= 0`, are silently filtered out. The count of dropped
ticks is never logged.

### Step 10.1 — Count dropped ticks and log them
Find:
```python
        valid = [
            (symbol, e, q)
            for e, q in zip(epochs, quotes)
            if isinstance(e, (int, float)) and isinstance(q, (int, float)) and q > 0
        ]
        if not valid:
            return 0
```
Replace with:
```python
        total = len(list(zip(epochs, quotes)))
        valid = [
            (symbol, e, q)
            for e, q in zip(epochs, quotes)
            if isinstance(e, (int, float)) and isinstance(q, (int, float)) and q > 0
        ]
        dropped = total - len(valid)
        if dropped > 0:
            logger.warning(
                "Dropped %d malformed tick(s) for %s (invalid epoch/quote type or non-positive price)",
                dropped,
                symbol,
            )
        if not valid:
            return 0
```

### Step 10.2 — Verify
Run: `python3 -m pytest tests/`  
All 7 tests must pass.

---

## Phase 11 — LOW: Crypto Library Import Inconsistency in `sign_model.py`

**File:** `scripts/sign_model.py`  
**Severity:** LOW  
**Root cause:** `sign_model.py` may import RSA or other unused cryptography
primitives. Both files should use only the Ed25519 path.

### Step 11.1 — Inspect the file
Read `scripts/sign_model.py` fully. Remove any import that is not directly
used (RSA, padding, hashes for RSA, etc.). Keep only Ed25519-related imports.

### Step 11.2 — Verify
Run: `python3 -m py_compile scripts/sign_model.py`  
Run: `python3 -m pytest tests/`

---

## Phase 12 — Floating-Point Drift (Already Mitigated)

**File:** `src/tick_processor.rs`  
**Severity:** Originally HIGH, now LOW  
**Root cause:** Naive accumulation of squared returns. The code already calls
`self.recalculate_sums()` every 1000 ticks (lines 209–212), which resets the
accumulators from scratch. This bounds the drift to at most 1000 ticks of error.
**No change required.**

**Action:** Skip this phase. Verified mitigated.

---

## Phase 13 — Ring Buffer Warm-Up (Already Guarded)

**File:** `src/tick_processor.rs`  
**Severity:** Originally HIGH, now LOW  
**Root cause:** The guards at lines 185 and 197 (`if self.len > WINDOW + 1`)
already prevent premature subtraction. Additionally `calculate_stats` returns
`(0.0, 0.0, a1, d1)` when `n < 1.0` (line 239). **No change required.**

**Action:** Skip this phase. Verified guarded.

---

## Phase 14 — Training: Add Assertion for Label Horizon

**File:** `scripts/train_fixed.py`  
**Severity:** LOW  
**Root cause:** The purge gap `seq_len + 10` in the train/val split assumes
`target_quote_future = quote.shift(-10)`. If this changes, the gap becomes
wrong silently.

### Step 14.1 — Add an assertion
In `load_and_sanitize_data()`, after this line:
```python
    df['target_quote_future'] = df['quote'].shift(-10)
```
Add:
```python
    _LABEL_HORIZON = 10  # must match purge_gap = seq_len + this value
```

### Step 14.2 — Use the constant in `main()`
In `main()`, replace:
```python
        purge_gap = seq_len + 10
```
With:
```python
        from scripts.train_fixed import _LABEL_HORIZON  # not needed if in same file
        purge_gap = seq_len + 10  # 10 == _LABEL_HORIZON
```
Actually, since `_LABEL_HORIZON` is defined in the same file, just write:
```python
        purge_gap = seq_len + _LABEL_HORIZON
```
And move `_LABEL_HORIZON = 10` to module level (top of `train_fixed.py`).

### Step 14.3 — Verify
Run: `python3 -m py_compile scripts/train_fixed.py`  
Run: `python3 -m pytest tests/`

---

## Final Phase — Regression Suite & Commit

### Step F.1 — Run full test suite
```bash
cargo fmt
cargo clippy --locked --offline --all-targets
cargo test --locked --offline
python3 -m pytest tests/
```
All commands must exit 0.

### Step F.2 — Commit all changes
```bash
git add -A
git commit -m "fix: resolve all audit findings (Phases 1, 2, 3, 4, 7, 9, 10, 11, 14)"
```

### Step F.3 — Push
```bash
git push -u origin <branch-name>
```

---

## Summary Table

| Phase | Severity | File | Issue | Skip? |
|-------|----------|------|-------|-------|
| 1 | HIGH | risk.rs | Break-even doesn't reset loss streak | No |
| 2 | HIGH | transformer.rs | Race: stale prob before first inference | No |
| 3 | HIGH | engine.rs | Contract timeout orphans subscription | No |
| 4 | HIGH | engine.rs | Stale proposal uses wall-clock only | No |
| 5 | MEDIUM | websocket_client.rs | Reconnect backoff reset | **Yes – correct as-is** |
| 6 | MEDIUM | tick_processor.rs | Flat tick reversal ambiguity | **Yes – intentional** |
| 7 | MEDIUM | transformer.rs | Inference cache has no TTL | No |
| 8 | LOW | config.rs | Silent bool parse failure | **Yes – returns Err correctly** |
| 9 | LOW | export_db.py | Corrupt output on crash | No |
| 10 | LOW | tick_collector.py | Silent tick drops | No |
| 11 | LOW | sign_model.py | Unused crypto imports | No |
| 12 | LOW | tick_processor.rs | FP drift accumulation | **Yes – mitigated by recalc** |
| 13 | LOW | tick_processor.rs | Ring buffer warm-up | **Yes – guarded** |
| 14 | LOW | train_fixed.py | Label horizon implicit constant | No |
