"""
Module 4: Labor Impact Simulation.
"""
from src.data_loader import get_data

def solve_m4_labor(scenario_cfg):
    """
    Tác động lao động AI.
    
    Args:
        scenario_cfg: Dictionary cấu hình kịch bản
        
    Returns:
        dict: Chứa thông tin về dịch chuyển việc làm
    """
    data = get_data()
    EMPLOYMENT = data.sectors_employment
    AUTOMATION_RISK = data.sectors_automation_risk
    SECTORS = data.sectors_names_vi.tolist()
    
    ai_rate = scenario_cfg['ai_adoption']
    jobs_lost    = EMPLOYMENT * AUTOMATION_RISK * ai_rate
    jobs_created = jobs_lost * 0.4
    net          = jobs_lost - jobs_created
    
    return {
        'sectors':  SECTORS,
        'net_jobs': [round(float(-v), 3) for v in net],  # âm = mất việc
        'total_lost':    round(float(jobs_lost.sum()), 2),
        'total_created': round(float(jobs_created.sum()), 2),
        'net_total':     round(float(-net.sum()), 2),
    }
