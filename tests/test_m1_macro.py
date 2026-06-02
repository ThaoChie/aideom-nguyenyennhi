import pytest
from src.modules.m1_macro import solve_m1_macro

def test_solve_m1_macro():
    cfg = {
        'tfp_growth': 0.02,
        'alloc_weights': [0.25, 0.25, 0.25, 0.25]
    }
    budget = 100
    res = solve_m1_macro(budget, cfg)
    
    assert 'years' in res
    assert 'gdp' in res
    assert len(res['years']) == 5
    assert len(res['gdp']) == 5
    assert res['years'][0] == 2026
    assert res['gdp'][0] > 0
