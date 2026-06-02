import pytest
from src.modules.m2_allocation import solve_m2_allocation

def test_solve_m2_allocation():
    cfg = {
        'alloc_weights': [0.4, 0.25, 0.20, 0.15]
    }
    budget = 1000
    res = solve_m2_allocation(budget, cfg)
    
    assert 'I' in res
    assert 'D' in res
    assert 'AI' in res
    assert 'H' in res
    
    total = res['I'] + res['D'] + res['AI'] + res['H']
    assert abs(total - budget) < 1.0  # Allow small floating point differences
