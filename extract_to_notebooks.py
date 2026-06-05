import os
import glob
import ast
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

# Imports string
HEADER_CELL = """import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np
import pandas as pd
import pulp
import pyomo.environ as pyo
from scipy.optimize import linprog, minimize, milp, LinearConstraint, Bounds
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize

from src.data_loader import get_data
"""

def extract_functions(filepath):
    with open(filepath, 'r') as f:
        code = f.read()
    tree = ast.parse(code)
    funcs = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            funcs[node.name] = ast.get_source_segment(code, node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    funcs[target.id] = ast.get_source_segment(code, node)
    return funcs

# Extract from optimization.py
opt_funcs = extract_functions('src/optimization.py')

# Extract from rl_env.py
rl_funcs = extract_functions('src/rl_env.py')

# Extract from modules
mod_funcs = {}
for file in glob.glob('src/modules/*.py'):
    if not file.endswith('__init__.py'):
        mod_funcs.update(extract_functions(file))

# Mapping problem to functions
mapping = {
    1: ['SCENARIOS', 'solve_bai01', '_solve'], # _solve is a helper for bai01
    2: ['solve_bai02'],
    3: ['solve_bai03'],
    4: ['solve_bai04'],
    5: ['PROJECTS', 'solve_bai05'],
    6: ['_entropy_weights', 'solve_bai06'],
    7: ['VietnamDigitalProblem', 'solve_bai07'],
    8: ['solve_bai08'],
    9: ['solve_bai09'],
    10: ['solve_bai10'],
    11: ['VietnamEconomyEnv', 'solve_bai11'],
    12: ['solve_m1_macro', 'solve_m2_allocation', 'solve_m3_priority', 'solve_m4_labor', 'solve_m5_topsis', 'solve_m6_risk', 'solve_bai12', 'solve_bai12_dashboard']
}

for bai, fnames in mapping.items():
    nb = new_notebook()
    
    # Markdown
    nb.cells.append(new_markdown_cell(f"# Bài {bai}\nĐây là notebook chứa mã nguồn đầy đủ của bài {bai}."))
    
    # Header
    if bai == 11:
        header = HEADER_CELL + "from stable_baselines3 import DQN\nimport gymnasium as gym\nfrom gymnasium import spaces\n"
    else:
        header = HEADER_CELL
    nb.cells.append(new_code_cell(header))
    
    # Code cells
    for fname in fnames:
        if fname in opt_funcs:
            code = opt_funcs[fname]
        elif fname in rl_funcs:
            code = rl_funcs[fname]
        elif fname in mod_funcs:
            code = mod_funcs[fname]
        else:
            print(f"Warning: {fname} not found!")
            continue
            
        nb.cells.append(new_code_cell(code))
        
    # Test cell
    test_code = f"if __name__ == '__main__':\n    # Test chạy hàm\n    pass"
    if fnames[-1].startswith('solve_bai'):
        test_code = f"if __name__ == '__main__':\n    res = {fnames[-1]}()\n    # In ra một số key để kiểm tra\n    if isinstance(res, dict):\n        print(res.keys())"
    nb.cells.append(new_code_cell(test_code))
    
    # Save notebook
    out_path = f"notebooks/bai{bai:02d}_notebook.ipynb"
    with open(out_path, 'w') as f:
        nbformat.write(nb, f)
        
print("Successfully generated all notebooks!")
