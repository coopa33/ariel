# Code Slowdown Analysis: A3_modified.py

## Summary
After syncing with upstream (ci-group/ariel), there are several potential reasons for code slowdown in `examples/A3_modified.py`.

## Changes from Upstream Sync

The merge commit `b465e55` added the following new function to `src/ariel/utils/runners.py`:

```python
def continue_simple_runner(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration: float = 10.0,
    steps_per_loop: int = 100,
) -> None:
    """Continue a simple headless simulation for a given duration."""
    while data.time < duration:
        mujoco.mj_step(model, data, nstep=steps_per_loop)
```

## Potential Slowdown Issues

### 1. **Duration Parameter Semantic Issue** ⚠️ CRITICAL

**Problem**: The `continue_simple_runner` function interprets `duration` as an **absolute time** to run until, not an **incremental duration** to add.

**Current Behavior**:
```python
# In examples/A3_modified.py, line 536-555:
simple_runner(model, data, duration=15)  # Runs until data.time = 15

# Then if checkpoint passed:
continue_simple_runner(model, data, duration=30)  # Runs until data.time = 30 (only 15 more seconds!)

# Then if next checkpoint passed:
continue_simple_runner(model, data, duration=55)  # Runs until data.time = 55 (only 25 more seconds!)
```

**Configuration Values** (lines 60-62):
```python
duration_flat: int = 15      # Initial simulation time
duration_rugged: int = 30    # TOTAL time including rugged terrain
duration_elevated: int = 55  # TOTAL time including elevated terrain
```

**Potential Issue**: If these values were previously interpreted as **incremental durations**, your robots would have run for:
- Before: 15 + 30 + 55 = 100 seconds total
- After sync: 15 + 15 + 25 = 55 seconds total

This would explain significant performance differences!

### 2. **Missing State Reset or Controller Updates**

**Problem**: `continue_simple_runner` doesn't reset any controller state or perform any setup that `simple_runner` does.

**Comparison**:
```python
# simple_runner (lines 34-41):
def simple_runner(...):
    mujoco.mj_resetData(model, data)        # ✓ Resets simulation
    data.ctrl = RNG.normal(scale=0.1, size=model.nu)  # ✓ Sets control
    while data.time < duration:
        mujoco.mj_step(model, data, nstep=steps_per_loop)

# continue_simple_runner (lines 64-65):
def continue_simple_runner(...):
    while data.time < duration:             # ✗ No reset, no control setup
        mujoco.mj_step(model, data, nstep=steps_per_loop)
```

**Impact**: The controller behavior might differ between the initial and continuation phases, affecting robot performance.

### 2. **Checkpoint Evaluation Logic**

**Problem**: Checkpoints are evaluated AFTER the initial simulation completes, using the final position.

**Code** (lines 542-555):
```python
simple_runner(model, data, duration=duration)
# Simulation completes, robot position is now fixed

# Check if checkpoint was passed during simulation:
if passed_checkpoint(sim_config.checkpoint_rugged, controller.tracker.history["xpos"][0]):
    continue_simple_runner(...)  # Continue from current position
```

**Potential Issues**:
- If the robot's position at time=15 doesn't meet the checkpoint, no additional simulation runs
- Fitness evaluation might be based on incomplete simulations
- The checkpoint positions might need adjustment based on the actual duration semantics

### 3. **Performance Implications for Evolutionary Algorithm**

If simulations are running for shorter durations than expected:

1. **Fitness Evaluation**: Robots evaluated for less time → different fitness scores
2. **Selection Pressure**: Different individuals selected as "best" 
3. **Convergence**: EA might converge to different solutions
4. **Computational Time**: Shorter simulations = faster, but potentially less accurate results

## Recommendations

### Option 1: Fix the Duration Semantics (RECOMMENDED)

Change `continue_simple_runner` to accept incremental duration:

```python
def continue_simple_runner(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration: float = 10.0,
    steps_per_loop: int = 100,
) -> None:
    """Continue a simple headless simulation for an additional duration."""
    end_time = data.time + duration  # Make it incremental
    while data.time < end_time:
        mujoco.mj_step(model, data, nstep=steps_per_loop)
```

And update the config to reflect incremental values:

```python
@dataclass
class EAConfig:
    duration_flat: int = 15          # Initial simulation time
    duration_rugged: int = 15        # Additional time for rugged (was 30)
    duration_elevated: int = 25      # Additional time for elevated (was 55)
```

### Option 2: Update Configuration to Match New Semantics

Keep the function as-is, but rename config values for clarity:

```python
@dataclass
class EAConfig:
    duration_flat: int = 15              # Initial simulation time
    total_time_after_rugged: int = 30    # Total time including rugged checkpoint
    total_time_after_elevated: int = 55  # Total time including elevated checkpoint
```

### Option 3: Verify Current Behavior Matches Intent

If the current behavior IS correct:
1. Verify that robots are reaching expected checkpoints
2. Confirm fitness values are in expected ranges
3. Check that evolutionary progress is occurring as expected

## Testing Recommendations

1. **Add timing assertions** to verify actual simulation durations
2. **Log checkpoint pass rates** to ensure robots are performing as expected
3. **Compare fitness distributions** before and after sync
4. **Profile simulation runs** to measure actual computational time

## Conclusion

The most likely cause of slowdown or performance change is the **ambiguous duration parameter semantics** in `continue_simple_runner`. The function treats `duration` as an absolute time rather than an increment, which may not match the original intention of the configuration values.

**Recommended Action**: Modify `continue_simple_runner` to use incremental durations (Option 1) and update configuration accordingly.
