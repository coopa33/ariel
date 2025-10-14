"""Extract position history from EA run data."""

import pickle
from pathlib import Path
import numpy as np

def extract_position_history_from_run(run_id=2, generation=36, data_path="__data__/A3_modified"):
    """Extract position history from the best performer of a specific generation."""
    
    data_dir = Path(data_path)
    gen_dir = data_dir / f"run_{run_id}" / f"generation_{generation:03d}"
    
    if not gen_dir.exists():
        print(f"Generation {generation} data not found at {gen_dir}")
        return None
    
    # Load best performers
    best_performers_file = gen_dir / "best_performers.pkl"
    
    try:
        with open(best_performers_file, 'rb') as f:
            best_performers = pickle.load(f)
        
        print(f"Loaded best performers: {type(best_performers)}")
        
        if isinstance(best_performers, dict):
            print(f"Best performers keys: {best_performers.keys()}")
            
            # Look for position history in the dictionary
            position_history = None
            possible_keys = ['position_history', 'trajectory', 'positions', 'path_history', 'simulation_data']
            
            for key in possible_keys:
                if key in best_performers:
                    position_history = best_performers[key]
                    print(f"Found position data in key: {key}")
                    print(f"Position data type: {type(position_history)}")
                    if hasattr(position_history, '__len__'):
                        print(f"Position data length: {len(position_history)}")
                    break
            
            # If not found, explore all keys
            if position_history is None:
                print("Position history not found in top-level keys. Exploring all keys...")
                for key, value in best_performers.items():
                    print(f"{key}: {type(value)}")
                    if hasattr(value, '__len__') and not isinstance(value, (str, bytes)):
                        try:
                            print(f"  -> Length: {len(value)}")
                            if len(value) > 0:
                                print(f"  -> First item type: {type(value[0])}")
                        except:
                            pass
                    
                    # Check if this value has position data
                    if hasattr(value, '__dict__'):
                        nested_attrs = list(value.__dict__.keys())
                        print(f"  -> Nested attributes: {nested_attrs}")
                        for nested_attr in possible_keys:
                            if hasattr(value, nested_attr):
                                position_history = getattr(value, nested_attr)
                                print(f"Found position data in {key}.{nested_attr}")
                                break
                    elif isinstance(value, dict):
                        for nested_key in possible_keys:
                            if nested_key in value:
                                position_history = value[nested_key]
                                print(f"Found position data in {key}.{nested_key}")
                                break
                    
                    if position_history is not None:
                        break
            
            return position_history
            
        elif isinstance(best_performers, list) and len(best_performers) > 0:
            best_robot = best_performers[0]
            print(f"Best robot type: {type(best_robot)}")
            
            # Explore the structure to find position history
            if hasattr(best_robot, '__dict__'):
                print(f"Best robot attributes: {list(best_robot.__dict__.keys())}")
                
                # Look for position history in various possible attributes
                position_history = None
                possible_attrs = ['position_history', 'trajectory', 'positions', 'path_history', 'simulation_data']
                
                for attr in possible_attrs:
                    if hasattr(best_robot, attr):
                        position_history = getattr(best_robot, attr)
                        print(f"Found position data in attribute: {attr}")
                        print(f"Position data type: {type(position_history)}")
                        if hasattr(position_history, '__len__'):
                            print(f"Position data length: {len(position_history)}")
                        break
                
                # If not found, check nested attributes
                if position_history is None:
                    for attr_name in best_robot.__dict__.keys():
                        attr_value = getattr(best_robot, attr_name)
                        if hasattr(attr_value, '__dict__'):
                            print(f"Checking nested attribute: {attr_name}")
                            for nested_attr in possible_attrs:
                                if hasattr(attr_value, nested_attr):
                                    position_history = getattr(attr_value, nested_attr)
                                    print(f"Found position data in {attr_name}.{nested_attr}")
                                    break
                        if position_history is not None:
                            break
                
                # If still not found, print all nested structure
                if position_history is None:
                    print("Position history not found. Exploring structure...")
                    for attr_name in best_robot.__dict__.keys():
                        attr_value = getattr(best_robot, attr_name)
                        print(f"{attr_name}: {type(attr_value)}")
                        if hasattr(attr_value, '__dict__'):
                            nested_attrs = list(attr_value.__dict__.keys())
                            print(f"  -> {nested_attrs}")
                
                return position_history
            else:
                print(f"Best robot is not an object with attributes: {best_robot}")
                return None
        else:
            print(f"Unexpected best performers format: {type(best_performers)}")
            return None
            
    except Exception as e:
        print(f"Error loading best performers: {e}")
        return None

if __name__ == "__main__":
    print("Extracting position history from run_2, generation 36...")
    position_history = extract_position_history_from_run(run_id=2, generation=36)
    
    if position_history is not None:
        print(f"\nPosition history found!")
        print(f"Type: {type(position_history)}")
        print(f"Length: {len(position_history) if hasattr(position_history, '__len__') else 'Unknown'}")
        
        if hasattr(position_history, '__len__') and len(position_history) > 0:
            print(f"First position: {position_history[0]}")
            print(f"Last position: {position_history[-1]}")
            
            # Save it to a simple format for the plotting function
            import json
            output_file = Path("__data__/A3_modified/run_2_position_history.json")
            with open(output_file, 'w') as f:
                json.dump(position_history, f, indent=2)
            print(f"Saved position history to: {output_file}")
        else:
            print("Position history is empty or has no length")
    else:
        print("No position history found.")