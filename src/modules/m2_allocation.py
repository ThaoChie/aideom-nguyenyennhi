"""
Module 2: Linear Programming Budget Allocation.
"""
import numpy as np
from scipy.optimize import linprog

def solve_m2_allocation(budget, scenario_cfg):
    """
    LP phân bổ tối ưu 4 hạng mục.
    
    Args:
        budget: Ngân sách đầu tư
        scenario_cfg: Dictionary cấu hình kịch bản
        
    Returns:
        dict: Chứa phân bổ I, D, AI, H
    """
    w = scenario_cfg['alloc_weights']
    c = [-w[0], -w[1], -w[2], -w[3]]
    A_ub = [[1, 1, 1, 1]]
    b_ub = [float(budget)]
    mins = [budget*0.10, budget*0.08, budget*0.08, budget*0.08]
    bounds = [(mins[i], budget*0.55) for i in range(4)]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        x = res.x
    else:
        x = np.array(w) * budget
        
    return {
        'I':  round(float(x[0]), 2),
        'D':  round(float(x[1]), 2),
        'AI': round(float(x[2]), 2),
        'H':  round(float(x[3]), 2),
    }
