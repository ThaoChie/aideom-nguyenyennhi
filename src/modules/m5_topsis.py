"""
Module 5: TOPSIS region ranking.
"""
import numpy as np
from src.data_loader import get_data

def solve_m5_topsis(scenario_cfg):
    """
    TOPSIS xếp hạng vùng.
    
    Args:
        scenario_cfg: Dictionary cấu hình kịch bản
        
    Returns:
        list: Danh sách các vùng đã xếp hạng
    """
    data = get_data()
    regions = data.regions_names_vi.tolist()
    X = data.X_regions
    is_benefit = [True, True, True, True, True, False]
    
    norm = np.sqrt((X**2).sum(axis=0))
    R = X / (norm + 1e-9)
    
    # entropy weights
    P = R / (R.sum(axis=0) + 1e-9)
    P = np.clip(P, 1e-9, 1)
    E = -1/np.log(6) * np.sum(P * np.log(P), axis=0)
    w = (1 - E) / ((1 - E).sum() + 1e-9)
    V = R * w
    
    ideal      = np.array([V[:, j].max() if is_benefit[j] else V[:, j].min() for j in range(6)])
    anti_ideal = np.array([V[:, j].min() if is_benefit[j] else V[:, j].max() for j in range(6)])
    
    D_plus  = np.sqrt(((V - ideal)**2).sum(axis=1))
    D_minus = np.sqrt(((V - anti_ideal)**2).sum(axis=1))
    C = D_minus / (D_plus + D_minus + 1e-9)
    idx = np.argsort(-C)
    
    return [{'region': regions[i], 'score': round(float(C[i]), 4), 'rank': int(np.where(idx == i)[0][0]+1)} for i in range(6)]
