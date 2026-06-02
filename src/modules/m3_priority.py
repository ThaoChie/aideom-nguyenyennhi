"""
Module 3: Sector priority ranking using multi-criteria evaluation.
"""
import numpy as np
from src.data_loader import get_data

def solve_m3_priority(scenario_cfg):
    """
    Xếp hạng ưu tiên ngành.
    
    Args:
        scenario_cfg: Dictionary cấu hình kịch bản
        
    Returns:
        list: Danh sách các ngành đã xếp hạng
    """
    data = get_data()
    X = data.X_sectors
    SECTORS = data.sectors_names_vi.tolist()
    
    good, bad = X[:, :6], X[:, 6]
    Gn = (good - good.min(0)) / (np.ptp(good, axis=0) + 1e-9)
    R  = (bad - bad.min()) / (np.ptp(bad) + 1e-9)
    
    w  = np.array([0.15, 0.15, 0.20, 0.15, 0.10, 0.20])
    w /= w.sum()
    
    score = Gn @ w - 0.15 * R
    idx   = np.argsort(-score)
    
    return [{'sector': SECTORS[i], 'score': round(float(score[i]), 4)} for i in idx]
