import pytest
from src.modules.m3_priority import solve_m3_priority

def test_solve_m3_priority():
    cfg = {}
    res = solve_m3_priority(cfg)
    
    assert isinstance(res, list)
    assert len(res) == 21
    assert 'sector' in res[0]
    assert 'score' in res[0]
    # Check sorting
    assert res[0]['score'] >= res[-1]['score']
