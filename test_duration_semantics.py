#!/usr/bin/env python3
"""
Test script to demonstrate the duration semantics issue in continue_simple_runner.

This script shows how the current implementation interprets durations as absolute times
rather than incremental additions, which could lead to unexpected simulation durations.
"""

class MockData:
    """Mock MuJoCo data object for testing."""
    def __init__(self):
        self.time = 0.0
    
    def step(self, amount=1):
        """Simulate a time step."""
        self.time += amount


def simple_runner_behavior(data, duration):
    """Simulates the behavior of simple_runner."""
    # Resets data.time to 0
    data.time = 0.0
    
    # Run until duration
    while data.time < duration:
        data.step(1)  # Each step adds 1 second
    
    print(f"After simple_runner(duration={duration}): data.time = {data.time}")


def continue_simple_runner_current_behavior(data, duration):
    """Simulates CURRENT behavior of continue_simple_runner (absolute time)."""
    # Does NOT reset data.time
    # Runs until absolute time reaches duration
    
    start_time = data.time
    while data.time < duration:
        data.step(1)
    
    elapsed = data.time - start_time
    print(f"After continue_simple_runner(duration={duration}): "
          f"data.time = {data.time}, elapsed = {elapsed}")


def continue_simple_runner_expected_behavior(data, duration):
    """Simulates EXPECTED behavior of continue_simple_runner (incremental time)."""
    # Does NOT reset data.time
    # Runs for an ADDITIONAL duration
    
    start_time = data.time
    end_time = data.time + duration
    
    while data.time < end_time:
        data.step(1)
    
    elapsed = data.time - start_time
    print(f"After continue_simple_runner_incremental(duration={duration}): "
          f"data.time = {data.time}, elapsed = {elapsed}")


def test_current_implementation():
    """Test the current implementation behavior."""
    print("=" * 70)
    print("CURRENT IMPLEMENTATION (Absolute Time Interpretation)")
    print("=" * 70)
    
    data = MockData()
    
    # Simulate the A3_modified.py behavior
    print("\n1. Initial flat terrain simulation:")
    simple_runner_behavior(data, duration=15)
    
    print("\n2. Continue for rugged terrain (if checkpoint passed):")
    continue_simple_runner_current_behavior(data, duration=30)
    
    print("\n3. Continue for elevated terrain (if checkpoint passed):")
    continue_simple_runner_current_behavior(data, duration=55)
    
    print(f"\n📊 TOTAL SIMULATION TIME: {data.time} seconds")
    print()


def test_expected_implementation():
    """Test the expected incremental implementation behavior."""
    print("=" * 70)
    print("EXPECTED IMPLEMENTATION (Incremental Time Interpretation)")
    print("=" * 70)
    
    data = MockData()
    
    # If durations were meant to be incremental
    print("\n1. Initial flat terrain simulation:")
    simple_runner_behavior(data, duration=15)
    
    print("\n2. Continue for rugged terrain (if checkpoint passed):")
    continue_simple_runner_expected_behavior(data, duration=30)
    
    print("\n3. Continue for elevated terrain (if checkpoint passed):")
    continue_simple_runner_expected_behavior(data, duration=55)
    
    print(f"\n📊 TOTAL SIMULATION TIME: {data.time} seconds")
    print()


def test_configuration_interpretation():
    """Show different interpretations of the configuration."""
    print("=" * 70)
    print("CONFIGURATION VALUE INTERPRETATIONS")
    print("=" * 70)
    
    print("\nFrom EAConfig:")
    print("  duration_flat: int = 15")
    print("  duration_rugged: int = 30")
    print("  duration_elevated: int = 55")
    
    print("\n📌 Interpretation 1: ABSOLUTE TIMES (current behavior)")
    print("  - Flat terrain:     0 →  15 seconds (15 seconds total)")
    print("  - Rugged terrain:  15 →  30 seconds (15 seconds added)")
    print("  - Elevated terrain: 30 →  55 seconds (25 seconds added)")
    print("  - TOTAL: 55 seconds")
    
    print("\n📌 Interpretation 2: INCREMENTAL DURATIONS (possible original intent)")
    print("  - Flat terrain:     0 →  15 seconds (15 seconds)")
    print("  - Rugged terrain:  15 →  45 seconds (30 seconds added)")
    print("  - Elevated terrain: 45 → 100 seconds (55 seconds added)")
    print("  - TOTAL: 100 seconds")
    
    print("\n⚠️  DIFFERENCE: 45 seconds (82% increase in simulation time!)")
    print()


def main():
    """Run all tests."""
    test_configuration_interpretation()
    print()
    test_current_implementation()
    print()
    test_expected_implementation()
    
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The difference in interpretation could cause:
1. ⏱️  Simulations to run for different durations (55s vs 100s)
2. 🎯 Different checkpoint pass rates
3. 📊 Different fitness evaluations
4. 🔄 Different EA convergence behavior
5. ⚡ Perceived "slowdown" if expectations were based on longer runs

Recommendation: Clarify the intended behavior and update code/config accordingly.
""")


if __name__ == "__main__":
    main()
