from pathlib import Path

# Here is a main.py that runs Phase 2 (data understanding/feature engineering) and Phase 3 (visualization).
# It assumes you have phase2.py and phase3.py scripts in your project.
#
### Explanation:
#### Loads and processes the dataset using phase2.py.
#### Runs visualizations using phase3.py.
#### Each phase is executed in order.

def run_script(script_path):
    with open(script_path) as f:
        code = f.read()
        exec(code, globals())

if __name__ == "__main__":
    print("=== Phase 2: Data Understanding & Feature Engineering ===")
    run_script(Path("phase2.py"))
    print("\n=== Phase 3: Visualization ===")
    run_script(Path("phase3.py"))