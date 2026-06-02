import pytest
from src.modules.m4_labor import solve_m4_labor

def test_solve_m4_labor():
    cfg = {
        'ai_adoption': 0.3
    }
    res = solve_m4_labor(cfg)
    
    assert 'sectors' in res
    assert 'net_jobs' in res
    assert 'total_lost' in res
    assert 'total_created' in res
    assert 'net_total' in res
    assert res['total_lost'] >= 0
    assert len(res['sectors']) == 21
    assert len(res['net_jobs']) == 21
