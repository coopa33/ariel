# Slowdown Analysis Results

This directory contains a comprehensive analysis of potential code slowdown issues in `examples/A3_modified.py` after syncing with upstream (ci-group/ariel).

## 📁 Files in This Analysis

### 1. **QUICKSTART_ANALYSIS.md** ⭐ START HERE
   - **Purpose**: User-friendly summary with quick recommendations
   - **Best for**: Getting a fast overview and actionable fixes
   - **Contains**: 
     - Simple explanation of the issue
     - Comparison table showing 55s vs 100s difference
     - Three fix options with code examples
     - Quick verification steps

### 2. **SLOWDOWN_ANALYSIS.md**
   - **Purpose**: Comprehensive technical analysis
   - **Best for**: Understanding all technical details
   - **Contains**:
     - Complete code comparisons
     - Detailed explanation of duration semantics
     - Analysis of all potential performance issues
     - Multiple solution approaches
     - Testing recommendations

### 3. **TIMELINE_VISUALIZATION.txt**
   - **Purpose**: Visual timeline comparison
   - **Best for**: Seeing the difference at a glance
   - **Contains**:
     - ASCII timeline showing 55s vs 100s
     - Checkpoint locations
     - Code implementation comparison
     - EA impact analysis

### 4. **test_duration_semantics.py**
   - **Purpose**: Executable demonstration
   - **Best for**: Seeing the issue in action
   - **Run with**: `python test_duration_semantics.py`
   - **Shows**: How duration parameters are interpreted differently

## 🎯 TL;DR - The Main Issue

The `continue_simple_runner()` function added from upstream interprets `duration` as an **absolute time** instead of an **incremental duration**.

**Your config:**
```python
duration_flat: int = 15      # ✓ Clear: initial simulation
duration_rugged: int = 30    # ⚠️ Ambiguous: total time (30s) or added time (30s)?
duration_elevated: int = 55  # ⚠️ Ambiguous: total time (55s) or added time (55s)?
```

**Current behavior:**
- Total simulation time: **55 seconds**
- Breakdown: 15s + 15s + 25s

**Possible original intent:**
- Total simulation time: **100 seconds**  
- Breakdown: 15s + 30s + 55s

**Impact:** 82% difference in simulation time!

## ✅ Quick Fix

**File:** `src/ariel/utils/runners.py`

**Change this:**
```python
def continue_simple_runner(model, data, duration=10.0, steps_per_loop=100):
    while data.time < duration:  # ⚠️ Absolute time
        mujoco.mj_step(model, data, nstep=steps_per_loop)
```

**To this:**
```python
def continue_simple_runner(model, data, duration=10.0, steps_per_loop=100):
    end_time = data.time + duration  # ✓ Incremental time
    while data.time < end_time:
        mujoco.mj_step(model, data, nstep=steps_per_loop)
```

## 🔍 How to Verify

### Step 1: Check Current Behavior
```bash
python test_duration_semantics.py
```

### Step 2: Check Your Results
Look at your simulation logs:
- How long do simulations actually run?
- Are robots reaching checkpoints?
- What are typical fitness values?

### Step 3: Compare with Expectations
- Do you expect 55s or 100s total simulation time?
- Are checkpoint locations reasonable for the actual simulation time?

## 📊 Impact on Your Code

### If simulations should be 100s (not 55s):

**Symptoms you might see:**
- ❌ Robots not traveling far enough
- ❌ Lower fitness scores than expected
- ❌ EA making slower progress
- ❌ Fewer robots passing checkpoints
- ❌ Different solutions being selected

**Why it feels like "slowdown":**
- EA appears to make less progress per generation
- Takes more generations to achieve same results
- Selection pressure is different

### If simulations should be 55s (current behavior is correct):

Then you might need to:
- ✓ Adjust checkpoint locations
- ✓ Adjust fitness function expectations
- ✓ Verify duration values match intent

## 🚀 Recommended Action Plan

1. **Determine Intent** (5 minutes)
   - Decide: Should total simulation be 55s or 100s?
   - Check if robots are reaching checkpoints as expected
   - Review fitness value ranges

2. **Choose Fix** (2 minutes)
   - If 100s → Apply the code fix above
   - If 55s → Update config comments for clarity

3. **Test** (10 minutes)
   - Run test script
   - Run a few EA generations
   - Verify checkpoint pass rates

4. **Validate** (30 minutes)
   - Check fitness distributions
   - Compare with previous results
   - Ensure EA is progressing as expected

## 📝 Additional Notes

### Other Potential Issues Found

While the duration semantics is the PRIMARY issue, the analysis also identified:

1. **Missing controller reset**: `continue_simple_runner` doesn't reset control state
   - Impact: Probably minor, controller is still active
   - Fix: Not urgent unless you see unexpected behavior

2. **Checkpoint logic timing**: Relies on absolute time values
   - Impact: Works correctly if durations are interpreted correctly
   - Fix: Ensure duration semantics match checkpoint expectations

### Files Modified by Upstream Sync

The merge commit `b465e55` added:
- `src/ariel/utils/runners.py` - Contains the `continue_simple_runner` function
- Many other upstream changes (see `.devcontainer/`, `.typings/`, etc.)

Your `examples/A3_modified.py` is a custom file that uses the upstream runners.

## 📞 Need Help?

1. **Read**: Start with `QUICKSTART_ANALYSIS.md`
2. **Visualize**: Check `TIMELINE_VISUALIZATION.txt`
3. **Test**: Run `test_duration_semantics.py`
4. **Deep Dive**: Read `SLOWDOWN_ANALYSIS.md` for all details

## 🎓 Understanding the Root Cause

The issue stems from an **ambiguous parameter name**:

```python
def continue_simple_runner(duration):
    # Is 'duration' the END TIME or the AMOUNT OF TIME to add?
```

Better parameter names would be:
- `target_time` (for absolute) or
- `additional_duration` (for incremental)

This is a classic API design issue that can lead to subtle bugs!

---

**Generated by:** Automated analysis of upstream sync changes  
**Date:** 2025-10-10  
**Repository:** coopa33/ariel  
**Upstream:** ci-group/ariel
