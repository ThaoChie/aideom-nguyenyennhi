import pytest
from src.modules.m6_risk import solve_m6_risk

def test_solve_m6_risk():
    allocation = {'I': 100, 'D': 100, 'AI': 100, 'H': 100} # AI share = 25%
    gdp_forecast = {'gdp': [1000, 1100, 1200, 1300, 1400]} # 40% growth
    cfg = {}
    
    res = solve_m6_risk(allocation, gdp_forecast, cfg)
    
    assert 'risk_score' in res
    assert 'level' in res
    assert 'gdp_growth_5y' in res
    assert 'ai_budget_share' in res
    
    # 0.6 - 0.4*0.5 - 0.25*0.3 = 0.6 - 0.2 - 0.075 = 0.325
    assert abs(res['risk_score'] - 0.325) < 0.001
    assert res['level'] == 'Thấp'
