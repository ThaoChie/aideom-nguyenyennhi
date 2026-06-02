import pytest
from src.modules.m5_topsis import solve_m5_topsis

def test_solve_m5_topsis():
    cfg = {}
    res = solve_m5_topsis(cfg)
    
    assert isinstance(res, list)
    assert len(res) == 6
    assert 'region' in res[0]
    assert 'score' in res[0]
    assert 'rank' in res[0]
    assert res[0]['rank'] == 1
    assert res[0]['score'] >= res[-1]['score']
