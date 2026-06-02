"""
Module 6: Risk Assessment.
"""
import numpy as np

def solve_m6_risk(allocation, gdp_forecast, scenario_cfg):
    """
    Đánh giá rủi ro tổng hợp.
    
    Args:
        allocation: Phân bổ ngân sách
        gdp_forecast: Dự báo GDP
        scenario_cfg: Cấu hình kịch bản
        
    Returns:
        dict: Chứa thông tin về rủi ro
    """
    ai_share = allocation['AI'] / max(sum(allocation.values()), 1)
    gdp_growth = (gdp_forecast['gdp'][-1] / max(gdp_forecast['gdp'][0], 1) - 1)
    
    risk_score = float(np.clip(0.6 - gdp_growth * 0.5 - ai_share * 0.3, 0.1, 0.9))
    level = 'Thấp' if risk_score < 0.35 else 'Trung bình' if risk_score < 0.65 else 'Cao'
    
    return {
        'risk_score': round(risk_score, 3),
        'level': level,
        'gdp_growth_5y': round(gdp_growth * 100, 2),
        'ai_budget_share': round(ai_share * 100, 2),
    }
