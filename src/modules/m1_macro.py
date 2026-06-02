"""
Module 1: Macroeconomic simulation and GDP forecast using Cobb-Douglas function.
"""

def solve_m1_macro(budget, scenario_cfg):
    """
    Dự báo GDP qua hàm sản xuất Cobb-Douglas.
    
    Args:
        budget: Ngân sách đầu tư
        scenario_cfg: Dictionary cấu hình kịch bản (chứa tfp_growth, alloc_weights)
        
    Returns:
        dict: Chứa list 'years' và 'gdp'
    """
    a, b, g, d, t = 0.33, 0.42, 0.10, 0.08, 0.07
    s = a + b + g + d + t
    a, b, g, d, t = a/s, b/s, g/s, d/s, t/s

    K0, L0, D0, AI0, H0 = 25900, 53.4, 19.5, 80.1, 29.2
    tfpg = scenario_cfg['tfp_growth']
    A0 = 1.0

    years = list(range(2026, 2031))
    gdp   = []
    K, L, D, AI, H, A = K0, L0, D0, AI0, H0, A0
    
    for i in range(5):
        g_val = A * (K**a) * (L**b) * (D**g) * (AI**d) * (H**t)
        gdp.append(round(float(g_val / 1000), 2))  # nghìn tỷ
        K  = K  * 1.06
        L  = L  * 1.005
        D  = D  + 1.5
        AI = min(AI * 1.08, 100)
        H  = H  + 0.5
        A  = A  * (1 + tfpg)
        
    return {'years': years, 'gdp': gdp}
