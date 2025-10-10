# Summary: Possible Reasons for Code Slowdown in A3_modified.py

After analyzing the differences before and after syncing with upstream (ci-group), here are the **possible reasons for code slowdown** in `examples/A3_modified.py`:

## 🎯 Primary Issue: Duration Semantics Ambiguity

The upstream sync added a new `continue_simple_runner()` function that has an **ambiguous duration parameter**:

```python
def continue_simple_runner(model, data, duration=10.0, steps_per_loop=100):
    while data.time < duration:  # ⚠️ Treats duration as ABSOLUTE time
        mujoco.mj_step(model, data, nstep=steps_per_loop)
```

### The Problem

Your configuration has these values:
```python
duration_flat: int = 15
duration_rugged: int = 30
duration_elevated: int = 55
```

**Question**: Are these meant to be:
1. **Absolute times** (current behavior): Run until simulation reaches this time
2. **Incremental durations** (possible original intent): Add this much time

### Impact on Simulation Time

| Interpretation | Flat | Rugged (added) | Elevated (added) | **TOTAL** |
|----------------|------|----------------|------------------|-----------|
| **Absolute** (current) | 0→15s (15s) | 15→30s (15s) | 30→55s (25s) | **55 seconds** |
| **Incremental** (possible) | 0→15s (15s) | 15→45s (30s) | 45→100s (55s) | **100 seconds** |

**Difference: 45 seconds (82% increase!)**

## 🔍 Why This Causes "Slowdown"

If your original implementation (before sync) used **incremental durations**, and the upstream version uses **absolute times**, then:

1. ⏱️ **Simulations run for less time**: 55s instead of 100s per robot
2. 🏃 **Robots travel shorter distances**: Less time to reach checkpoints/targets
3. 📉 **Lower fitness scores**: Less distance covered = worse fitness
4. 🎯 **Fewer checkpoint passes**: Robots might not reach rugged/elevated terrains
5. 🔄 **Different EA behavior**: Selection pressure changes, different solutions emerge
6. ⚡ **Perceived "slowdown"**: EA not making progress as expected

## 🧪 How to Verify

Run the test script to see the difference:

```bash
python test_duration_semantics.py
```

This will show you exactly how the two interpretations differ.

## ✅ Recommended Fix

### Option 1: Change to Incremental Duration (Recommended if original behavior was incremental)

**File**: `src/ariel/utils/runners.py`

```python
def continue_simple_runner(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration: float = 10.0,
    steps_per_loop: int = 100,
) -> None:
    """Continue a simple headless simulation for an additional duration."""
    end_time = data.time + duration  # ✓ Make it incremental
    while data.time < end_time:
        mujoco.mj_step(model, data, nstep=steps_per_loop)
```

**And update config to be clear**:
```python
duration_flat: int = 15          # Initial simulation time
duration_rugged_extra: int = 30  # Additional time for rugged (renamed for clarity)
duration_elevated_extra: int = 55  # Additional time for elevated (renamed for clarity)
```

### Option 2: Keep Absolute Times but Fix Config Values

If absolute times are correct, update config to match:

```python
duration_flat: int = 15              # Run until time = 15
total_time_rugged: int = 30          # Run until time = 30 (15s added)
total_time_elevated: int = 55        # Run until time = 55 (25s added)
```

### Option 3: Restore Original Behavior

If you know what the original implementation was before the sync, you could restore that behavior.

## 📋 Quick Check

To determine which interpretation is correct, check:

1. **Are robots reaching checkpoints?**
   - If very few robots pass checkpoints → likely running for too short
   
2. **What are typical fitness values?**
   - Compare with results before the sync
   
3. **How long should simulations actually run?**
   - Do you expect robots to have 55s or 100s to complete the course?

## 📄 Full Details

See `SLOWDOWN_ANALYSIS.md` for complete technical analysis including:
- Detailed code comparison
- Additional potential issues
- Multiple fix options
- Testing recommendations

## 🚀 Next Steps

1. **Determine intended behavior**: Should durations be incremental or absolute?
2. **Choose a fix option**: Based on your determination
3. **Update code/config**: Implement the chosen fix
4. **Test**: Verify robots behave as expected
5. **Compare results**: Check if EA performance is restored

---

**Bottom Line**: The duration parameter ambiguity in `continue_simple_runner` is the most likely cause of unexpected behavior after syncing with upstream. The fix is simple once you determine the intended semantics.
