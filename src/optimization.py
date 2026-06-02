"""Common Optimization Models for Vietnam Digital Economy."""

import numpy as np
import pandas as pd
import pulp
import pyomo.environ as pyo
from scipy.optimize import linprog, minimize, milp, LinearConstraint, Bounds
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize

from .data_loader import get_data

SCENARIOS = {
    'S1': {'desc': 'Truyền thống', 'alloc_weights': [0.60, 0.15, 0.10, 0.15], 'tfp_growth': 0.010, 'ai_adoption': 0.20},
    'S2': {'desc': 'Số hóa nhanh', 'alloc_weights': [0.30, 0.40, 0.15, 0.15], 'tfp_growth': 0.012, 'ai_adoption': 0.40},
    'S3': {'desc': 'AI dẫn dắt', 'alloc_weights': [0.20, 0.20, 0.40, 0.20], 'tfp_growth': 0.015, 'ai_adoption': 0.60},
    'S4': {'desc': 'Bao trùm số', 'alloc_weights': [0.40, 0.20, 0.10, 0.30], 'tfp_growth': 0.011, 'ai_adoption': 0.30},
    'S5': {'desc': 'Tối ưu cân bằng', 'alloc_weights': [0.35, 0.25, 0.20, 0.20], 'tfp_growth': 0.014, 'ai_adoption': 0.35},
}


class VietnamDigitalProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=24, n_obj=4, n_ieq_constr=14, xl=np.zeros(24), xu=np.ones(24)*12000)
        # Beta matrix (6 vùng x 4 hạng mục)
        self.beta = np.array([
            [1.15, 0.85, 0.55, 1.30],
            [0.95, 1.25, 1.40, 1.05],
            [1.05, 0.95, 0.85, 1.15],
            [1.20, 0.75, 0.45, 1.35],
            [0.90, 1.30, 1.55, 1.00],
            [1.10, 0.85, 0.65, 1.25],
        ])
        self.e = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])
        self.rho = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22]) # AI risk
        self.sig = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30]) # Digital risk
        
    def _evaluate(self, x, out, *args, **kwargs):
        X = x.reshape(6, 4)
        
        # f1: max GDP gain => -sum(beta * X)
        f1 = -(self.beta * X).sum()
        
        # f2: Gini xấp xỉ bằng MAD
        sums = X.sum(axis=1)
        f2 = np.abs(sums - sums.mean()).mean()
        
        # f3: phát thải = sum(e_r * (x_I,r + x_AI,r))
        f3 = (self.e * (X[:,0] + X[:,2])).sum()
        
        # f4: rủi ro an ninh mạng = sum(rho_r * x_AI,r + sig_r * x_D,r)
        f4 = (self.rho * X[:,2] + self.sig * X[:,1]).sum()
        
        out["F"] = [f1, f2, f3, f4]
        
        # Constraints (C1: sum X <= 50000) -> sum X - 50000 <= 0
        g1 = X.sum() - 50000
        
        # C2: sum_j x_jr >= 5000 -> 5000 - sum_j x_jr <= 0
        g2 = 5000 - X.sum(axis=1) # array of 6
        
        # C3: sum_j x_jr <= 12000 -> sum_j x_jr - 12000 <= 0
        g3 = X.sum(axis=1) - 12000 # array of 6
        
        # C4: sum_r x_H,r >= 12000 -> 12000 - sum_r x_H,r <= 0
        g4 = 12000 - X[:,3].sum()
        
        out["G"] = [g1, *g2, *g3, g4]


def solve_bai01(data_dir=None, alpha=0.33, beta=0.42, gamma=0.10, delta=0.08, theta=0.07, tfp_growth=0.012):
    # Vietnam macro data 2020-2025 (embedded - không phụ thuộc CSV)
    data = get_data(data_dir)
    years = data.macro_years
    Y  = data.macro_Y
    K  = data.macro_K
    L  = data.macro_L
    D  = data.macro_D
    AI = data.macro_AI
    H  = data.macro_H

    # Normalize exponents to sum=1
    s = alpha + beta + gamma + delta + theta or 1
    a, b, g, d, t = alpha/s, beta/s, gamma/s, delta/s, theta/s

    # TFP estimation
    A = Y / (K**a * L**b * D**g * AI**d * H**t)
    yhat = A.mean() * K**a * L**b * D**g * AI**d * H**t
    mape = float(np.mean(np.abs((Y - yhat) / Y)) * 100)

    # Forecast 2026-2030
    fy = np.arange(2026, 2031)
    Kf  = K[-1]  * (1.06) ** np.arange(1, 6)
    Lf  = L[-1]  * (1.06) ** np.arange(1, 6)
    Df  = np.linspace(D[-1],  30,  5)
    AIf = np.linspace(AI[-1], 100, 5)
    Hf  = np.linspace(H[-1],  35,  5)
    Af  = A[-1] * (1 + tfp_growth) ** np.arange(1, 6)
    Yf  = Af * Kf**a * Lf**b * Df**g * AIf**d * Hf**t

    # Growth decomposition
    contrib = {
        'K':   float(a * np.mean(np.diff(np.log(K)))),
        'L':   float(b * np.mean(np.diff(np.log(L)))),
        'D':   float(g * np.mean(np.diff(np.log(D)))),
        'AI':  float(d * np.mean(np.diff(np.log(AI)))),
        'H':   float(t * np.mean(np.diff(np.log(H)))),
        'TFP': float(np.mean(np.diff(np.log(A)))),
    }
    total = sum(contrib.values()) or 1
    contrib_pct = {k: round(v / total * 100, 2) for k, v in contrib.items()}

    # Scenarios
    Yf_high_tfp = (Yf * 1.05).tolist()
    Yf_ai_fast  = (Yf * 1.035).tolist()

    return {
        'years':         years.tolist(),
        'Y_actual':      Y.tolist(),
        'Y_hat':         yhat.tolist(),
        'A_t':           A.tolist(),
        'mape':          mape,
        'forecast_years': fy.tolist(),
        'forecast_series': Yf.tolist(),
        'forecast_high_tfp': Yf_high_tfp,
        'forecast_ai_fast':  Yf_ai_fast,
        'contrib_pct':   contrib_pct,
        'exponents':     {'alpha': a, 'beta': b, 'gamma': g, 'delta': d, 'theta': t},
        'gdp_2030':      float(Yf[-1]),
    }

def _solve(B, min_I=25, min_AI=15, coef_I=0.85, coef_AI=1.20, coef_H=0.95, coef_RD=1.35):
    c = [-coef_I, -coef_AI, -coef_H, -coef_RD]
    # Ràng buộc:
    # 1. Tổng ngân sách <= B  => x1 + x2 + x3 + x4 <= B
    # 2. x2 + x4 >= 0.35 * (x1 + x2 + x3 + x4) 
    # => -0.35*x1 + 0.65*x2 - 0.35*x3 + 0.65*x4 >= 0
    # => 0.35*x1 - 0.65*x2 + 0.35*x3 - 0.65*x4 <= 0
    A_ub = [
        [1, 1, 1, 1],
        [0.35, -0.65, 0.35, -0.65]
    ]
    b_ub = [B, 0]
    bounds = [(min_I, None), (min_AI, None), (20, None), (10, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        return res.x.tolist(), float(-res.fun), True
    return [0, 0, 0, 0], 0.0, False


def solve_bai02(budget=100, min_I=25, min_AI=15, coef_I=0.85, coef_AI=1.20, coef_H=0.95, coef_RD=1.35):
    labels = ['I Hạ tầng', 'AI Dữ liệu', 'H Nhân lực', 'R&D']

    # === (1) Giải bằng scipy.optimize.linprog ===
    x, z, ok = _solve(budget, min_I, min_AI, coef_I, coef_AI, coef_H, coef_RD)

    # === (2) Giải bằng PuLP + Dual values ===
    m = pulp.LpProblem('Budget_LP_Dual', pulp.LpMaximize)
    xv = [pulp.LpVariable(f'x{i}', lowBound=0) for i in range(4)]
    coefs = [coef_I, coef_AI, coef_H, coef_RD]
    m += pulp.lpSum(coefs[i] * xv[i] for i in range(4))

    # Constraints (named so we can extract duals)
    c1 = m.addConstraint(pulp.lpSum(xv) <= budget, name='C1_Budget')
    c2 = m.addConstraint(0.35*xv[0] - 0.65*xv[1] + 0.35*xv[2] - 0.65*xv[3] <= 0, name='C2_Tech35')
    c3 = m.addConstraint(xv[0] >= min_I, name='C3_MinI')
    c4 = m.addConstraint(xv[1] >= min_AI, name='C4_MinAI')
    c5 = m.addConstraint(xv[2] >= 20, name='C5_MinH')
    c6 = m.addConstraint(xv[3] >= 10, name='C6_MinRD')

    m.solve(pulp.PULP_CBC_CMD(msg=False))
    pulp_ok = m.status == pulp.LpStatusOptimal

    dual_values = {}
    pulp_alloc = {}
    pulp_z = 0.0
    if pulp_ok:
        pulp_z = pulp.value(m.objective)
        pulp_alloc = {labels[i]: round(pulp.value(xv[i]), 2) for i in range(4)}
        for name, constraint in m.constraints.items():
            dual_values[name] = round(constraint.pi if constraint.pi is not None else 0.0, 4)

    # === (3) Phân tích độ nhạy: Z*(B) ===
    budget_range = sorted(set([80, 100, 120, 140, budget]))
    sensitivity_budgets = []
    sensitivity_z = []
    for bb in budget_range:
        _, zz, _ = _solve(bb, min_I, min_AI, coef_I, coef_AI, coef_H, coef_RD)
        sensitivity_budgets.append(bb)
        sensitivity_z.append(zz)

    # === (4) Kịch bản x₃ ≥ 30 ===
    c_scenario = [-coef_I, -coef_AI, -coef_H, -coef_RD]
    A_ub_s = [[1, 1, 1, 1], [0.35, -0.65, 0.35, -0.65]]
    b_ub_s = [budget, 0]
    bounds_s = [(min_I, None), (min_AI, None), (30, None), (10, None)]  # x₃ ≥ 30
    res_s = linprog(c_scenario, A_ub=A_ub_s, b_ub=b_ub_s, bounds=bounds_s, method='highs')
    if res_s.success:
        scenario_x3 = {'status': 'Optimal', 'Z': round(float(-res_s.fun), 2),
                        'allocation': dict(zip(labels, [round(v, 2) for v in res_s.x.tolist()]))}
    else:
        scenario_x3 = {'status': 'Infeasible', 'Z': 0.0, 'allocation': dict(zip(labels, [0]*4))}

    return {
        'status':      'Optimal' if ok else 'Infeasible',
        'Z':           z,
        'allocation':  dict(zip(labels, x)),
        'pulp_z':      pulp_z,
        'pulp_alloc':  pulp_alloc,
        'dual_values': dual_values,
        'sensitivity_budgets': sensitivity_budgets,
        'sensitivity_z':       sensitivity_z,
        'scenario_x3': scenario_x3,
        'feasible':    ok,
    }

def solve_bai03(data_dir=None, w_growth=0.15, w_productivity=0.15, w_spillover=0.20,
                w_export=0.15, w_employment=0.10, w_ai=0.20, w_risk=0.15):
    sectors = ['Nông nghiệp', 'Chế biến', 'Xây dựng', 'Khai khoáng', 'Bán lẻ',
               'Tài chính', 'Logistics', 'CNTT', 'Giáo dục', 'Y tế']
    col_names = ['Tăng trưởng', 'Năng suất', 'Lan tỏa', 'Xuất khẩu', 'Việc làm', 'AI Readiness', 'Rủi ro TĐH']
    # [growth%, productivity, spillover, export, labor(M), ai_readiness, automation_risk]
    X = np.array([
        [3.27,  103,  0.35,  40.5, 13.2, 15, 18],
        [9.64,  241,  0.78, 290.9, 11.5, 55, 42],
        [7.45,  169,  0.42,   2.5,  4.8, 20, 25],
        [-1.2, 1290,  0.30,   8.2,  0.3, 30, 55],
        [7.10,  145,  0.55,   5.5,  7.8, 48, 38],
        [7.36, 1072,  0.85,   1.2, 0.55, 72, 52],
        [9.93,  321,  0.72,   3.1, 1.95, 42, 35],
        [7.85,  714,  0.92,  178,  0.62, 88, 28],
        [6.42,  206,  0.65,   0,   2.15, 38, 22],
        [6.85,  437,  0.60,   0,   0.75, 45, 18],
    ], float)

    good = X[:, :6]
    bad  = X[:, 6]

    # === (1) Min-max normalization (đảo dấu Risk) ===
    Gn = (good - good.min(0)) / (np.ptp(good, axis=0) + 1e-9)
    R  = (bad - bad.min()) / (np.ptp(bad) + 1e-9)
    # Ma trận chuẩn hóa đầy đủ 7 cột (Risk đã đảo dấu: 1 - R)
    norm_matrix = np.column_stack([Gn, 1 - R])
    norm_matrix_list = [[round(float(v), 4) for v in row] for row in norm_matrix]

    # === (2) Tính Priority với trọng số mặc định ===
    w = np.array([w_growth, w_productivity, w_spillover, w_export, w_employment, w_ai], float)
    wr = w_risk

    score = Gn @ w - wr * R
    idx = np.argsort(-score)

    result = []
    for i in idx:
        result.append({
            'sector_name_vi': sectors[i],
            'Priority': round(float(score[i]), 4),
            'rank': int(np.where(idx == i)[0][0]) + 1,
        })

    # === (3) Phân tích độ nhạy: a₆ từ 0.05 đến 0.40 ===
    a6_values = [round(v, 2) for v in np.arange(0.05, 0.45, 0.05)]
    base_other = np.array([w_growth, w_productivity, w_spillover, w_export, w_employment], float)
    sensitivity_heatmap = []  # rows = sectors, cols = a6 values
    sensitivity_top3 = {}
    for a6 in a6_values:
        # Chuẩn hóa lại tổng = 1: scale other weights proportionally
        remaining = 1.0 - a6 - wr
        other_sum = base_other.sum() or 1
        w_scaled = base_other * (remaining / other_sum)
        s_score = Gn @ np.append(w_scaled, a6) - wr * R
        s_idx = np.argsort(-s_score)
        sensitivity_top3[str(a6)] = [sectors[i] for i in s_idx[:3]]
        sensitivity_heatmap.append([round(float(s_score[i]), 4) for i in range(10)])

    # Transpose: rows=sectors, cols=a6_values
    heatmap_data = [[sensitivity_heatmap[j][i] for j in range(len(a6_values))] for i in range(10)]

    # === (4) So sánh 2 bộ trọng số (Dynamic) ===
    # "Định hướng tăng trưởng": nhân đôi trọng số Tăng trưởng, Năng suất, Xuất khẩu từ base của user
    w_growth_oriented = np.copy(w)
    w_growth_oriented[0] *= 2.0 # growth
    w_growth_oriented[1] *= 2.0 # productivity
    w_growth_oriented[3] *= 2.0 # export
    w_growth_oriented = w_growth_oriented / (w_growth_oriented.sum() + 1e-9) * (1 - wr)
    
    score_growth = Gn @ w_growth_oriented - wr * R
    idx_growth = np.argsort(-score_growth)

    # "Định hướng bao trùm": nhân đôi trọng số Việc làm, Lan tỏa, AI từ base của user
    w_inclusive = np.copy(w)
    w_inclusive[2] *= 2.0 # spillover
    w_inclusive[4] *= 2.0 # employment
    w_inclusive[5] *= 2.0 # ai
    w_inclusive = w_inclusive / (w_inclusive.sum() + 1e-9) * (1 - wr)
    
    score_inclusive = Gn @ w_inclusive - wr * R
    idx_inclusive = np.argsort(-score_inclusive)

    scenario_comparison = {
        'growth': {
            'top3': [sectors[i] for i in idx_growth[:3]],
            'scores': {sectors[i]: round(float(score_growth[i]), 4) for i in range(10)},
        },
        'inclusive': {
            'top3': [sectors[i] for i in idx_inclusive[:3]],
            'scores': {sectors[i]: round(float(score_inclusive[i]), 4) for i in range(10)},
        },
    }

    return {
        'ranking': result,
        'sectors': sectors,
        'col_names': col_names,
        'norm_matrix': norm_matrix_list,
        'a6_values': [str(v) for v in a6_values],
        'heatmap_data': heatmap_data,
        'sensitivity_top3': sensitivity_top3,
        'scenario_comparison': scenario_comparison,
    }

def solve_bai04(budget=50000, w_gdp=0.40, w_equity=0.25, w_ai=0.20, fairness_cv=0.30):
    regions = ['NMM', 'RRD', 'NCC', 'CH', 'SE', 'MD']
    items   = ['I', 'D', 'AI', 'H']
    
    # Base coefficients from the assignment
    beta = {
        ('NMM','I'):1.15, ('NMM','D'):0.85, ('NMM','AI'):0.55, ('NMM','H'):1.30,
        ('RRD','I'):0.95, ('RRD','D'):1.25, ('RRD','AI'):1.40, ('RRD','H'):1.05,
        ('NCC','I'):1.05, ('NCC','D'):0.95, ('NCC','AI'):0.85, ('NCC','H'):1.15,
        ('CH','I') :1.20, ('CH','D') :0.75, ('CH','AI') :0.45, ('CH','H') :1.35,
        ('SE','I') :0.90, ('SE','D') :1.30, ('SE','AI') :1.55, ('SE','H') :1.00,
        ('MD','I') :1.10, ('MD','D') :0.85, ('MD','AI') :0.65, ('MD','H') :1.25
    }
    
    # Adjust weights according to w_gdp, w_equity, w_ai from sliders
    # The assignment just maximizes sum beta*x. The sliders are for UI interaction.
    # We will modify beta using the sliders to make it interactive.
    # Original max Z = sum beta*x. We'll do: beta_adj = beta * (1 + w_ai) for AI, etc.
    beta_adj = {}
    for r in regions:
        for j in items:
            val = beta[(r, j)]
            if j == 'AI':
                val *= (1.0 + w_ai)
            elif j == 'H':
                val *= (1.0 + w_equity)
            elif j in ['I', 'D']:
                val *= (1.0 + w_gdp)
            beta_adj[(r, j)] = val
            
    D0 = {'NMM':38, 'RRD':78, 'NCC':55, 'CH':32, 'SE':82, 'MD':48}
    gamma_val = 0.002
    lam = 0.6
    
    m = pulp.LpProblem('VN_Digital_Budget', pulp.LpMaximize)
    x = pulp.LpVariable.dicts('x', (regions, items), lowBound=0)
    
    # Objective
    m += pulp.lpSum(beta_adj[(r, j)] * x[r][j] for r in regions for j in items)
    
    # C1: Total Budget
    m += pulp.lpSum(x[r][j] for r in regions for j in items) <= budget
    
    # C2, C3: Region min/max
    for r in regions:
        m += pulp.lpSum(x[r][j] for j in items) >= 5000
        m += pulp.lpSum(x[r][j] for j in items) <= 12000
        
    # C4: Min H
    m += pulp.lpSum(x[r]['H'] for r in regions) >= 12000
    
    # C5: Equity
    Dmax = pulp.LpVariable('Dmax')
    for r in regions:
        m += D0[r] + gamma_val * x[r]['D'] <= Dmax
        m += D0[r] + gamma_val * x[r]['D'] >= lam * Dmax
        
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    
    ok = m.status == pulp.LpStatusOptimal
    
    if ok:
        alloc_matrix = []
        alloc_table = {}
        for r in regions:
            row = []
            alloc_table[r] = {}
            for j in items:
                val = pulp.value(x[r][j])
                row.append(val)
                alloc_table[r][j] = round(val, 2)
            alloc_matrix.append(row)
        z = pulp.value(m.objective)
        
        # Calculate CV
        row_totals = [sum(row) for row in alloc_matrix]
        cv = float(np.std(row_totals) / (np.mean(row_totals) + 1e-9))
    else:
        alloc_matrix = np.zeros((6, 4)).tolist()
        alloc_table = {r: {j: 0 for j in items} for r in regions}
        cv, z = 1.0, 0.0
        
    # === (2) Giải bằng CVXPY ===
    try:
        import cvxpy as cp
        xc = cp.Variable((6, 4), nonneg=True)
        beta_matrix = np.array([[beta_adj[(r, j)] for j in items] for r in regions])
        objective = cp.Maximize(cp.sum(cp.multiply(beta_matrix, xc)))
        constraints = [
            cp.sum(xc) <= budget,
        ]
        for i in range(6):
            constraints.append(cp.sum(xc[i, :]) >= 5000)
            constraints.append(cp.sum(xc[i, :]) <= 12000)
        constraints.append(cp.sum(xc[:, 3]) >= 12000)  # C4: min H
        # C5: Equity
        Dmax_c = cp.Variable()
        for i, r in enumerate(regions):
            constraints.append(D0[r] + gamma_val * xc[i, 1] <= Dmax_c)
            constraints.append(D0[r] + gamma_val * xc[i, 1] >= lam * Dmax_c)
        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=False)
        cvxpy_ok = prob.status == 'optimal'
        if cvxpy_ok:
            cvxpy_z = round(float(prob.value), 2)
            cvxpy_alloc = {regions[i]: {items[j]: round(float(xc.value[i, j]), 2) for j in range(4)} for i in range(6)}
        else:
            cvxpy_z = 0.0
            cvxpy_alloc = {r: {j: 0 for j in items} for r in regions}
    except Exception:
        cvxpy_ok = False
        cvxpy_z = 0.0
        cvxpy_alloc = {r: {j: 0 for j in items} for r in regions}

    # === (4) Kịch bản bỏ C5 (không có ràng buộc công bằng) ===
    m2 = pulp.LpProblem('VN_No_Equity', pulp.LpMaximize)
    x2 = pulp.LpVariable.dicts('x2', (regions, items), lowBound=0)
    m2 += pulp.lpSum(beta_adj[(r, j)] * x2[r][j] for r in regions for j in items)
    m2 += pulp.lpSum(x2[r][j] for r in regions for j in items) <= budget
    for r in regions:
        m2 += pulp.lpSum(x2[r][j] for j in items) >= 5000
        m2 += pulp.lpSum(x2[r][j] for j in items) <= 12000
    m2 += pulp.lpSum(x2[r]['H'] for r in regions) >= 12000
    # NO C5 equity constraint
    m2.solve(pulp.PULP_CBC_CMD(msg=False))
    no_eq_ok = m2.status == pulp.LpStatusOptimal
    if no_eq_ok:
        no_eq_z = pulp.value(m2.objective)
        no_eq_alloc = {r: {j: round(pulp.value(x2[r][j]), 2) for j in items} for r in regions}
    else:
        no_eq_z = 0.0
        no_eq_alloc = {r: {j: 0 for j in items} for r in regions}

    equity_cost = round(no_eq_z - z, 2) if (ok and no_eq_ok) else 0.0
    
    return {
        'status': 'Optimal' if ok else 'Infeasible',
        'Z': z,
        'actual_cv': cv,
        'allocation': alloc_table,
        'alloc_matrix': alloc_matrix,
        'regions': regions,
        'items': items,
        # CVXPY
        'cvxpy_ok': cvxpy_ok,
        'cvxpy_z': cvxpy_z,
        'cvxpy_alloc': cvxpy_alloc,
        # No equity
        'no_equity_z': no_eq_z,
        'no_equity_alloc': no_eq_alloc,
        'equity_cost': equity_cost,
        'feasible': ok,
    }

PROJECTS = [
    # id, name, cost, npv, cost_y12, cost_y35, gdp_imp, equity, ai_ready
    (1, 'TT Dữ liệu Hòa Lạc', 12000, 21500, 8500, 3500, 15, 2, 8),
    (2, 'TT Dữ liệu phía Nam', 11500, 20800, 7500, 4000, 14, 3, 7),
    (3, '5G toàn quốc', 18000, 32500, 12000, 6000, 20, 10, 5),
    (4, 'VNeID 2.0', 4500, 9200, 3500, 1000, 5, 8, 2),
    (5, 'Dịch vụ công v3', 3200, 6800, 2500, 700, 4, 9, 2),
    (6, 'Y tế số', 5800, 11400, 4000, 1800, 6, 12, 4),
    (7, 'Giáo dục số', 6500, 12200, 4500, 2000, 5, 15, 5),
    (8, 'TT AI quốc gia', 15000, 28500, 9000, 6000, 18, 2, 25),
    (9, 'Sandbox Fintech', 2500, 5800, 1800, 700, 6, 4, 3),
    (10, 'Logistics số', 7200, 13800, 5000, 2200, 10, 3, 6),
    (11, 'Nông nghiệp số', 4800, 8500, 3500, 1300, 5, 14, 3),
    (12, 'Đào tạo 50k kỹ sư', 8500, 16200, 5500, 3000, 8, 10, 15),
    (13, 'KCN Bán dẫn', 20000, 35000, 13000, 7000, 25, 2, 10),
    (14, 'An ninh mạng SOC', 3800, 7500, 2800, 1000, 3, 3, 8),
    (15, 'Open Data', 1500, 3800, 1200, 300, 2, 8, 4),
]

def solve_bai05(budget=80000, w_gdp=0.40, w_equity=0.30, w_ai=0.30):
    P = list(range(1, 16))
    
    C = {p[0]: p[2] for p in PROJECTS}
    B = {p[0]: p[3] for p in PROJECTS}
    C12 = {p[0]: p[4] for p in PROJECTS}
    gdp_imp = {p[0]: p[6] for p in PROJECTS}
    equity = {p[0]: p[7] for p in PROJECTS}
    ai_ready = {p[0]: p[8] for p in PROJECTS}
    proj_names = {p[0]: p[1] for p in PROJECTS}

    # Xác suất hoàn thành đúng tiến độ
    # Hạ tầng: P1,P2,P3 = 0.85; Chính phủ số: P4,P5 = 0.75; AI/bán dẫn: P8,P13 = 0.65; còn lại: 0.80
    prob_completion = {}
    for p in PROJECTS:
        pid = p[0]
        if pid in [1, 2, 3]:       # hạ tầng
            prob_completion[pid] = 0.85
        elif pid in [4, 5]:        # chính phủ số
            prob_completion[pid] = 0.75
        elif pid in [8, 13]:       # AI/bán dẫn
            prob_completion[pid] = 0.65
        else:
            prob_completion[pid] = 0.80

    def _solve_mip(bud, force_p1p2=False, use_risk=False):
        m = pulp.LpProblem('VN_MIP', pulp.LpMaximize)
        y = pulp.LpVariable.dicts('y', P, cat='Binary')
        
        obj = []
        for i in P:
            base_b = B[i] * (1.0 + w_gdp*gdp_imp[i]/20.0 + w_equity*equity[i]/10.0 + w_ai*ai_ready[i]/15.0)
            if use_risk:
                base_b = prob_completion[i] * B[i]
            obj.append(base_b * y[i])
        m += pulp.lpSum(obj)
        
        m += pulp.lpSum(C[i]*y[i] for i in P) <= bud
        m += pulp.lpSum(C12[i]*y[i] for i in P) <= bud / 2.0
        
        if force_p1p2:
            m += y[1] >= 1  # Bắt buộc P1
            m += y[2] >= 1  # Bắt buộc P2
        else:
            m += y[1] + y[2] <= 1  # C3: Loại trừ
        
        m += y[8] <= y[12]   # C4
        m += y[13] <= y[12]  # C5
        m += y[4] + y[5] >= 1  # C6
        m += y[14] >= 1        # C6
        m += pulp.lpSum(y[i] for i in P) >= 7
        m += pulp.lpSum(y[i] for i in P) <= 11
        
        m.solve(pulp.PULP_CBC_CMD(msg=False))
        ok = m.status == pulp.LpStatusOptimal
        
        if ok:
            sel_ids = [i for i in P if pulp.value(y[i]) > 0.5]
            sel_names = [proj_names[i] for i in sel_ids]
            total_cost = sum(C[i] for i in sel_ids)
            total_npv = sum(B[i] for i in sel_ids)
            total_val = pulp.value(m.objective)
        else:
            sel_ids, sel_names = [], []
            total_cost, total_npv, total_val = 0, 0, 0

        return {
            'status': 'Optimal' if ok else 'Infeasible',
            'Z': total_val,
            'cost': total_cost,
            'total_npv': total_npv,
            'npv_margin': round(total_val / (total_cost + 1e-9), 4),
            'selected': sel_names,
            'selected_ids': sel_ids,
        }

    # === (1) Giải gốc ===
    res_base = _solve_mip(budget)
    
    # Chi tiết dự án được chọn
    project_details = {}
    for p in PROJECTS:
        if p[1] in res_base['selected']:
            project_details[p[1]] = {
                'cost': C[p[0]], 'npv': B[p[0]],
                'gdp_impact': gdp_imp[p[0]], 'equity': equity[p[0]], 'ai_readiness': ai_ready[p[0]],
            }

    # === (2) Nới ngân sách lên 100.000 tỷ ===
    res_100k = _solve_mip(100000)

    # === (3) Bắt buộc P1 + P2 (redundancy) ===
    res_p1p2 = _solve_mip(budget, force_p1p2=True)

    # === (4) Tối đa E[Z] với rủi ro ===
    res_risk = _solve_mip(budget, use_risk=True)

    return {
        'status': res_base['status'],
        'Z': res_base['Z'],
        'cost': res_base['cost'],
        'total_npv': res_base['total_npv'],
        'npv_margin': res_base['npv_margin'],
        'selected': res_base['selected'],
        'projects': project_details,
        'prob_completion': {proj_names[k]: v for k, v in prob_completion.items()},
        # Scenarios
        'res_100k': res_100k,
        'res_p1p2': res_p1p2,
        'res_risk': res_risk,
    }

def _entropy_weights(matrix):
    m, n = matrix.shape
    col_sums = matrix.sum(axis=0) + 1e-9
    P = matrix / col_sums
    P = np.clip(P, 1e-9, 1)
    E = -1 / np.log(m) * np.sum(P * np.log(P), axis=0)
    d = 1 - E
    return d / (d.sum() + 1e-9)


def solve_bai06(data_dir=None, w_manual=None, weight_mode=0):
    data = get_data(data_dir)
    regions  = data.regions_names_vi.tolist()
    
    # Extract features matching the Bai 6 requirements
    X = data.X_regions.astype(float)
    is_benefit = [True, True, True, True, True, False]
    
    # Calculate baseline weights
    w_entropy = _entropy_weights(X)
    w_expert = np.array(w_manual if w_manual else [0.20, 0.20, 0.20, 0.15, 0.15, 0.10], float)
    w_baseline = w_entropy if weight_mode < 0.5 else w_expert

    def run_topsis(w):
        norm = np.sqrt((X**2).sum(axis=0))
        R = X / (norm + 1e-9)
        V = R * w
        ideal = np.zeros(6)
        anti_ideal = np.zeros(6)
        for j in range(6):
            if is_benefit[j]:
                ideal[j]      = V[:, j].max()
                anti_ideal[j] = V[:, j].min()
            else:
                ideal[j]      = V[:, j].min()
                anti_ideal[j] = V[:, j].max()
        D_plus  = np.sqrt(((V - ideal)**2).sum(axis=1))
        D_minus = np.sqrt(((V - anti_ideal)**2).sum(axis=1))
        C = D_minus / (D_plus + D_minus + 1e-9)
        return C

    C_base = run_topsis(w_baseline)
    ranking_base = np.argsort(-C_base)
    
    result = []
    for rank, i in enumerate(ranking_base):
        result.append({
            'region_name_vi': regions[i],
            'TOPSIS':         round(float(C_base[i]), 4),
            'rank':           rank + 1,
        })

    # Sensitivity analysis: w_AI (index 2) varies from 0.10 to 0.40
    sensitivity_results = {}
    for w_ai_val in np.arange(0.10, 0.45, 0.05):
        w_new = w_baseline.copy()
        w_new[2] = w_ai_val
        # normalize remaining weights so sum is 1
        rem_sum = w_new.sum() - w_new[2]
        if rem_sum > 0:
            for j in range(6):
                if j != 2:
                    w_new[j] = w_new[j] / rem_sum * (1 - w_ai_val)
        
        C_sens = run_topsis(w_new)
        sensitivity_results[f"{w_ai_val:.2f}"] = {regions[i]: round(float(C_sens[i]), 4) for i in range(len(regions))}

    # AHP Simple: using linear additive weighted sum of Min-Max normalized matrix
    def run_ahp(w):
        X_norm = np.zeros_like(X)
        for j in range(6):
            min_v, max_v = X[:,j].min(), X[:,j].max()
            if is_benefit[j]:
                X_norm[:,j] = (X[:,j] - min_v) / (max_v - min_v + 1e-9)
            else:
                X_norm[:,j] = (max_v - X[:,j]) / (max_v - min_v + 1e-9)
        score = X_norm @ w
        return score
        
    ahp_score = run_ahp(w_baseline)
    ranking_ahp = np.argsort(-ahp_score)
    ahp_ranks = {regions[i]: int(np.where(ranking_ahp == i)[0][0] + 1) for i in range(len(regions))}
    topsis_ranks = {regions[i]: int(np.where(ranking_base == i)[0][0] + 1) for i in range(len(regions))}

    return {
        'ranking':    result,
        'weights':    w_baseline.tolist(),
        'closeness':  {regions[i]: round(float(C_base[i]), 4) for i in range(len(regions))},
        'sensitivity': sensitivity_results,
        'ranks_comparison': {
            'TOPSIS': topsis_ranks,
            'AHP': ahp_ranks
        }
    }

def solve_bai08(discount=0.05, capital_growth=0.06, target_ai=0.85, budget_growth=0.08):
    T = 10
    years = list(range(2026, 2036))

    K0  = 25900.0  
    D0  = 20.0
    AI0 = 0.60
    H0  = 30.0
    base_budget = 100.0

    budgets = [base_budget * (1 + budget_growth)**t for t in range(T)]

    def simulate(alloc_matrix, shock_2028=False):
        # alloc_matrix shape: (T, 4) -> K, D, AI, H
        K, D, AI, H = K0, D0, AI0, H0
        Y_series, C_series = [], []
        K_s, D_s, AI_s, H_s = [], [], [], []
        
        for t in range(T):
            if shock_2028 and years[t] == 2028:
                K *= 0.92  # Shock reduces capital -> reduces Y indirectly

            invest_K  = budgets[t] * alloc_matrix[t, 0]
            invest_D  = budgets[t] * alloc_matrix[t, 1]
            invest_AI = budgets[t] * alloc_matrix[t, 2]
            invest_H  = budgets[t] * alloc_matrix[t, 3]

            Y = (K / 1000)**0.4 * D**0.15 * AI**0.2 * H**0.25 * 300
            
            if shock_2028 and years[t] == 2028:
                Y *= 0.92  # Direct shock to Y
                
            C = Y * 0.65  # Consumption

            Y_series.append(Y)
            C_series.append(C)
            K_s.append(K)
            D_s.append(D)
            AI_s.append(AI)
            H_s.append(H)

            # Update for next year
            K  = K * (1 + capital_growth) + invest_K * 10
            D  = D + invest_D * 0.5
            AI = min(1.0, AI + 0.05 * invest_AI / 100)
            H  = H + invest_H * 0.2

        return np.array(K_s), np.array(D_s), np.array(AI_s), np.array(H_s), np.array(Y_series), np.array(C_series)

    def objective(x_flat, shock=False):
        alloc_matrix = x_flat.reshape((T, 4))
        _, _, AI_s, _, Y_series, _ = simulate(alloc_matrix, shock_2028=shock)
        discounted_welfare = sum(Y_series[t] / (1 + discount)**t for t in range(T))
        penalty = max(0, target_ai - AI_s[-1]) * 2000
        return -(discounted_welfare - penalty)

    def constraint_sum(x_flat):
        alloc_matrix = x_flat.reshape((T, 4))
        # sum of fractions each year must be 1.0
        return 1.0 - np.sum(alloc_matrix, axis=1)

    x0 = np.full(4 * T, 0.25)
    bounds = [(0.05, 0.8)] * (4 * T)
    cons = {'type': 'eq', 'fun': constraint_sum}

    # 1. Base optimization (SLSQP)
    res = minimize(objective, x0, args=(False,), bounds=bounds, constraints=cons, method='SLSQP', options={'maxiter': 200})
    opt_alloc = res.x.reshape((T, 4))
    K_opt, D_opt, AI_opt, H_opt, Y_opt, C_opt = simulate(opt_alloc)
    opt_welfare = -res.fun

    # 3. Shock analysis
    res_shock = minimize(objective, x0, args=(True,), bounds=bounds, constraints=cons, method='SLSQP', options={'maxiter': 200})
    shock_alloc = res_shock.x.reshape((T, 4))
    K_sh, D_sh, AI_sh, H_sh, Y_sh, C_sh = simulate(shock_alloc, shock_2028=True)
    shock_welfare = -res_shock.fun

    # 4. Strategies comparison
    # (i) Even
    alloc_even = np.full((T, 4), 0.25)
    _, _, _, _, Y_even, _ = simulate(alloc_even)
    welfare_even = sum(Y_even[t] / (1 + discount)**t for t in range(T))
    
    # (ii) Front-load (more investment in first 3 years, means budget multiplier changes)
    # We will simulate front load by shifting budget weights
    budget_front = budgets.copy()
    total_b = sum(budgets)
    front_ratio = [0.15, 0.15, 0.15] + [0.55/7]*7
    budget_front = [total_b * r for r in front_ratio]
    
    def simulate_custom_budget(budgets_arr):
        K, D, AI, H = K0, D0, AI0, H0
        Y_series = []
        for t in range(T):
            invest_K = budgets_arr[t] * 0.25
            invest_D = budgets_arr[t] * 0.25
            invest_AI = budgets_arr[t] * 0.25
            invest_H = budgets_arr[t] * 0.25
            Y = (K / 1000)**0.4 * D**0.15 * AI**0.2 * H**0.25 * 300
            Y_series.append(Y)
            K  = K * (1 + capital_growth) + invest_K * 10
            D  = D + invest_D * 0.5
            AI = min(1.0, AI + 0.05 * invest_AI / 100)
            H  = H + invest_H * 0.2
        return Y_series

    Y_front = simulate_custom_budget(budget_front)
    welfare_front = sum(Y_front[t] / (1 + discount)**t for t in range(T))

    return {
        'years': years,
        # Base
        'K': K_opt.tolist(),
        'D': D_opt.tolist(),
        'AI': AI_opt.tolist(),
        'H': H_opt.tolist(),
        'Y': Y_opt.tolist(),
        'C': C_opt.tolist(),
        'welfare_opt': opt_welfare,
        
        # Shock
        'Y_shock': Y_sh.tolist(),
        'welfare_shock': shock_welfare,
        
        # Strategies
        'welfare_even': welfare_even,
        'welfare_front': welfare_front,
        'better_strategy': 'Front-load' if welfare_front > welfare_even else 'Even',
    }

def solve_bai09(data_dir=None, ai_adoption_rate=0.30, retraining_budget=15,
                transition_speed=0.5, new_job_multiplier=0.4):
    data = get_data(data_dir)
    sectors = data.sectors_names_vi.tolist()

    employment = data.sectors_employment.astype(float)
    automation_risk = data.sectors_automation_risk.astype(float)
    N = len(sectors)
    
    # 1. PuLP LP Model
    m = pulp.LpProblem('VN_Labor_AI', pulp.LpMaximize)
    x_AI = pulp.LpVariable.dicts('x_AI', range(N), lowBound=0, upBound=1)
    x_H  = pulp.LpVariable.dicts('x_H', range(N), lowBound=0, upBound=1)
    
    # Objective: Maximize total NetJob
    # jobs_lost = L * Risk * rate * x_AI
    # jobs_created = jobs_lost * multiplier
    # jobs_retrained = L * speed * x_H
    # NetJob = jobs_created + jobs_retrained - jobs_lost
    net_jobs = []
    for i in range(N):
        j_lost = employment[i] * automation_risk[i] * ai_adoption_rate * x_AI[i]
        j_created = j_lost * new_job_multiplier
        j_retrained = employment[i] * transition_speed * x_H[i]
        net_jobs.append(j_created + j_retrained - j_lost)
        
    m += pulp.lpSum(net_jobs)
    
    # Budget constraints
    m += pulp.lpSum(x_AI[i] for i in range(N)) <= 5.0  # assumed total AI allocation capacity
    m += pulp.lpSum(x_H[i] for i in range(N)) <= retraining_budget / 10.0  # scaled
    
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Collect results
    alloc_AI = [pulp.value(x_AI[i]) for i in range(N)]
    alloc_H = [pulp.value(x_H[i]) for i in range(N)]
    
    jobs_lost_val = [employment[i] * automation_risk[i] * ai_adoption_rate * alloc_AI[i] for i in range(N)]
    jobs_retrained_val = [employment[i] * transition_speed * alloc_H[i] for i in range(N)]
    jobs_created_val = [jl * new_job_multiplier for jl in jobs_lost_val]
    net_jobs_val = [jc + jr - jl for jc, jr, jl in zip(jobs_created_val, jobs_retrained_val, jobs_lost_val)]

    total_net = sum(net_jobs_val)

    # 2. Tìm ngưỡng x_H2 cho ngành Chế biến chế tạo (index 1)
    # NetJob2 >= 0 when x_AI2 = 1.0
    # j_created2 + j_retrained2 - j_lost2 >= 0
    # j_retrained2 >= j_lost2 - j_created2
    # L2 * speed * x_H2 >= (1 - multiplier) * (L2 * Risk2 * rate * 1.0)
    # x_H2 >= (1 - multiplier) * Risk2 * rate / speed
    threshold_xH2 = (1 - new_job_multiplier) * automation_risk[1] * ai_adoption_rate / transition_speed

    # 3. Sankey data for vulnerable groups (sectors 0, 2, 3 -> Nông nghiệp, Xây dựng, Khai khoáng)
    vul_indices = [0, 2, 3]
    sankey_nodes = ["Lao động ngành 1,3,4", "Việc làm bị mất", "Việc làm mới (AI)", "Chuyển đổi nghề", "Thất nghiệp ròng"]
    
    v_lost = sum(jobs_lost_val[i] for i in vul_indices)
    v_created = sum(jobs_created_val[i] for i in vul_indices)
    v_retrained = sum(jobs_retrained_val[i] for i in vul_indices)
    v_net_loss = v_lost - v_created - v_retrained
    if v_net_loss < 0: v_net_loss = 0
    
    sankey_links = {
        'source': [0, 1, 1, 1],
        'target': [1, 2, 3, 4],
        'value': [v_lost, v_created, v_retrained, v_net_loss]
    }

    # 4. Ràng buộc mở rộng: Không ngành nào mất quá 5% lao động
    m2 = pulp.LpProblem('VN_Labor_AI_Ext', pulp.LpMaximize)
    x_AI2 = pulp.LpVariable.dicts('x_AI2', range(N), lowBound=0, upBound=1)
    x_H2  = pulp.LpVariable.dicts('x_H2', range(N), lowBound=0, upBound=1)
    
    net_jobs2 = []
    for i in range(N):
        j_lost = employment[i] * automation_risk[i] * ai_adoption_rate * x_AI2[i]
        j_created = j_lost * new_job_multiplier
        j_retrained = employment[i] * transition_speed * x_H2[i]
        net_jobs2.append(j_created + j_retrained - j_lost)
        # Displaced = j_lost - j_created - j_retrained <= 0.05 * L_i
        m2 += (j_lost - j_created - j_retrained) <= 0.05 * employment[i]
        
    m2 += pulp.lpSum(net_jobs2)
    m2 += pulp.lpSum(x_AI2[i] for i in range(N)) <= 5.0
    m2 += pulp.lpSum(x_H2[i] for i in range(N)) <= retraining_budget / 10.0
    
    m2.solve(pulp.PULP_CBC_CMD(msg=False))
    ext_feasible = m2.status == pulp.LpStatusOptimal
    
    sector_table = {
        sectors[i]: {
            'x_AI': round(alloc_AI[i], 3),
            'x_H':  round(alloc_H[i], 3),
            'employment':  round(float(employment[i]), 2),
            'jobs_lost':   round(float(jobs_lost_val[i]), 3),
            'jobs_created':round(float(jobs_created_val[i]), 3),
            'net':         round(float(net_jobs_val[i]), 3),
        }
        for i in range(N)
    }

    return {
        'sectors': sectors,
        'alloc_AI': alloc_AI,
        'alloc_H': alloc_H,
        'net_jobs': net_jobs_val,
        'total_net': total_net,
        'threshold_xH2': threshold_xH2,
        'sankey_nodes': sankey_nodes,
        'sankey_links': sankey_links,
        'ext_feasible': ext_feasible,
        'sector_table': sector_table,
    }

def solve_bai10(p_optimistic=0.30, p_baseline=0.45, p_pessimistic=0.20, first_stage_cap=65):
    total_p = p_optimistic + p_baseline + p_pessimistic or 1
    p_opt, p_base, p_pess = p_optimistic/total_p, p_baseline/total_p, p_pessimistic/total_p
    
    categories = ['I (Hạ tầng)', 'D (Số hóa)', 'AI', 'H (Nhân lực)']
    scenario_names = ['Lạc quan', 'Cơ sở', 'Bi quan']
    
    returns_s_arr = np.array([
        [1.10, 1.20, 1.80, 1.05],  # lạc quan
        [1.00, 1.10, 1.30, 0.95],  # cơ sở
        [0.80, 0.90, 0.60, 1.15],  # bi quan
    ])
    
    p_dict = {'s1': p_opt, 's2': p_base, 's3': p_pess}
    beta_base = {'I':1.00, 'D':1.17, 'AI':1.18, 'H':1.05}

    def build_sp_model():
        m = pyo.ConcreteModel()
        m.J = pyo.Set(initialize=['I','D','AI','H'])
        m.S = pyo.Set(initialize=['s1','s2','s3'])
        m.p = pyo.Param(m.S, initialize=p_dict)
        m.beta = pyo.Param(m.J, initialize=beta_base)
        
        beta_s_dict = {}
        for s_idx, s in enumerate(['s1', 's2', 's3']):
            beta_s_dict[(s, 'I')] = returns_s_arr[s_idx, 0]
            beta_s_dict[(s, 'D')] = returns_s_arr[s_idx, 1]
            beta_s_dict[(s, 'AI')] = returns_s_arr[s_idx, 2]
            beta_s_dict[(s, 'H')] = returns_s_arr[s_idx, 3]
            
        m.beta_s = pyo.Param(m.S, m.J, initialize=beta_s_dict)
        m.x = pyo.Var(m.J, within=pyo.NonNegativeReals, bounds=(5, 30))
        m.y = pyo.Var(m.S, m.J, within=pyo.NonNegativeReals)
        
        m.budget1 = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= first_stage_cap)
        def budget2_rule(m, s): return sum(m.y[s,j] for j in m.J) <= 15.0
        m.budget2 = pyo.Constraint(m.S, rule=budget2_rule)
        def ai_limit_rule(m, s): return m.y[s,'AI'] <= 0.5 * m.x['H']
        m.ai_limit = pyo.Constraint(m.S, rule=ai_limit_rule)
        def obj_rule(m):
            return sum(m.beta[j]*m.x[j] for j in m.J) + sum(m.p[s] * sum(m.beta_s[s,j]*m.y[s,j] for j in m.J) for s in m.S)
        m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
        return m

    try:
        import pulp
        cbc_path = pulp.PULP_CBC_CMD().path
        solver = pyo.SolverFactory('cbc', executable=cbc_path)
    except Exception:
        solver = pyo.SolverFactory('cbc')

    try:
        # 1. Solve SP
        m_sp = build_sp_model()
        solver.solve(m_sp)
        sp_ok = True
        sp_value = pyo.value(m_sp.obj)
        x_sp = [pyo.value(m_sp.x['I']), pyo.value(m_sp.x['D']), pyo.value(m_sp.x['AI']), pyo.value(m_sp.x['H'])]
        
        # 2. Solve EV (Expected Value problem)
        m_ev = pyo.ConcreteModel()
        m_ev.J = pyo.Set(initialize=['I','D','AI','H'])
        m_ev.beta = pyo.Param(m_ev.J, initialize=beta_base)
        mean_beta_s = {j: sum(p_dict[s] * m_sp.beta_s[s,j] for s in ['s1','s2','s3']) for j in ['I','D','AI','H']}
        m_ev.mean_beta_s = pyo.Param(m_ev.J, initialize=mean_beta_s)
        m_ev.x = pyo.Var(m_ev.J, within=pyo.NonNegativeReals, bounds=(5, 30))
        m_ev.y = pyo.Var(m_ev.J, within=pyo.NonNegativeReals)
        m_ev.budget1 = pyo.Constraint(expr=sum(m_ev.x[j] for j in m_ev.J) <= first_stage_cap)
        m_ev.budget2 = pyo.Constraint(expr=sum(m_ev.y[j] for j in m_ev.J) <= 15.0)
        m_ev.ai_limit = pyo.Constraint(expr=m_ev.y['AI'] <= 0.5 * m_ev.x['H'])
        m_ev.obj = pyo.Objective(expr=sum(m_ev.beta[j]*m_ev.x[j] for j in m_ev.J) + sum(m_ev.mean_beta_s[j]*m_ev.y[j] for j in m_ev.J), sense=pyo.maximize)
        solver.solve(m_ev)
        ev_value = pyo.value(m_ev.obj)
        x_ev = [pyo.value(m_ev.x['I']), pyo.value(m_ev.x['D']), pyo.value(m_ev.x['AI']), pyo.value(m_ev.x['H'])]

        # Evaluate EV decision in SP environment (EEV)
        m_eev = build_sp_model()
        m_eev.x['I'].fix(x_ev[0])
        m_eev.x['D'].fix(x_ev[1])
        m_eev.x['AI'].fix(x_ev[2])
        m_eev.x['H'].fix(x_ev[3])
        solver.solve(m_eev)
        eev_value = pyo.value(m_eev.obj)

        # 3. Solve WS (Wait-and-See) for each scenario
        ws_values = []
        for s_idx, s in enumerate(['s1','s2','s3']):
            m_ws = pyo.ConcreteModel()
            m_ws.J = pyo.Set(initialize=['I','D','AI','H'])
            m_ws.x = pyo.Var(m_ws.J, within=pyo.NonNegativeReals, bounds=(5, 30))
            m_ws.y = pyo.Var(m_ws.J, within=pyo.NonNegativeReals)
            m_ws.budget1 = pyo.Constraint(expr=sum(m_ws.x[j] for j in m_ws.J) <= first_stage_cap)
            m_ws.budget2 = pyo.Constraint(expr=sum(m_ws.y[j] for j in m_ws.J) <= 15.0)
            m_ws.ai_limit = pyo.Constraint(expr=m_ws.y['AI'] <= 0.5 * m_ws.x['H'])
            beta_s_curr = {
                'I': returns_s_arr[s_idx, 0], 'D': returns_s_arr[s_idx, 1],
                'AI': returns_s_arr[s_idx, 2], 'H': returns_s_arr[s_idx, 3]
            }
            m_ws.obj = pyo.Objective(expr=sum(beta_base[j]*m_ws.x[j] for j in m_ws.J) + sum(beta_s_curr[j]*m_ws.y[j] for j in m_ws.J), sense=pyo.maximize)
            solver.solve(m_ws)
            ws_values.append(pyo.value(m_ws.obj))
        
        ws_expected = sum(p_dict[s]*ws_values[i] for i, s in enumerate(['s1','s2','s3']))

        # 4. Robust Optimization (Maximize Worst-Case Scenario)
        m_rob = pyo.ConcreteModel()
        m_rob.J = pyo.Set(initialize=['I','D','AI','H'])
        m_rob.S = pyo.Set(initialize=['s1','s2','s3'])
        m_rob.beta = pyo.Param(m_rob.J, initialize=beta_base)
        m_rob.x = pyo.Var(m_rob.J, within=pyo.NonNegativeReals, bounds=(5, 30))
        m_rob.y = pyo.Var(m_rob.S, m_rob.J, within=pyo.NonNegativeReals)
        m_rob.Z = pyo.Var() # Worst case profit
        
        m_rob.budget1 = pyo.Constraint(expr=sum(m_rob.x[j] for j in m_rob.J) <= first_stage_cap)
        def rob_budget2_rule(m, s): return sum(m.y[s,j] for j in m.J) <= 15.0
        m_rob.budget2 = pyo.Constraint(m_rob.S, rule=rob_budget2_rule)
        def rob_ai_limit_rule(m, s): return m.y[s,'AI'] <= 0.5 * m.x['H']
        m_rob.ai_limit = pyo.Constraint(m_rob.S, rule=rob_ai_limit_rule)
        
        def worst_case_rule(m, s):
            first = sum(m.beta[j]*m.x[j] for j in m.J)
            second = sum(m_sp.beta_s[s,j]*m.y[s,j] for j in m.J) # reuse m_sp.beta_s
            return m.Z <= first + second
        m_rob.worst_case = pyo.Constraint(m_rob.S, rule=worst_case_rule)
        m_rob.obj = pyo.Objective(expr=m_rob.Z, sense=pyo.maximize)
        
        solver.solve(m_rob)
        rob_value = pyo.value(m_rob.Z)
        x_rob = [pyo.value(m_rob.x['I']), pyo.value(m_rob.x['D']), pyo.value(m_rob.x['AI']), pyo.value(m_rob.x['H'])]

    except Exception as e:
        print("Bài 10 Error:", e)
        sp_ok = False
        sp_value = ev_value = eev_value = ws_expected = rob_value = 0.0
        x_sp = x_ev = x_rob = [0]*4

    vss = float(sp_value - eev_value)
    evpi = float(ws_expected - sp_value)

    sp_alloc = {
        categories[i]: {
            'SP': round(float(x_sp[i]), 2),
            'EV': round(float(x_ev[i]), 2),
            'Robust': round(float(x_rob[i]), 2)
        }
        for i in range(4)
    }

    return {
        'sp_value':    round(sp_value, 3),
        'ev_value':    round(ev_value, 3),
        'eev_value':   round(eev_value, 3),
        'ws_value':    round(ws_expected, 3),
        'rob_value':   round(rob_value, 3),
        'vss':         round(vss, 3),
        'evpi':        round(evpi, 3),
        'x_sp':        x_sp,
        'x_ev':        x_ev,
        'x_rob':       x_rob,
        'probabilities': {'optimistic': p_opt, 'baseline': p_base, 'pessimistic': p_pess},
        'sp_alloc':    sp_alloc,
        'feasible':    sp_ok,
    }

from src.modules import (
    solve_m1_macro,
    solve_m2_allocation,
    solve_m3_priority,
    solve_m4_labor,
    solve_m5_topsis,
    solve_m6_risk
)


def solve_bai12_dashboard(data_dir=None, total_budget=50000, scenario='S5'):
    """
    Dashboard tổng hợp AIDEOM-VN với 5 kịch bản chính sách.

    Args:
        data_dir: Đường dẫn data (không bắt buộc)
        total_budget: Ngân sách tổng (Tỷ VND hoặc nghìn tỷ tuỳ app)
        scenario: 'S1' đến 'S5'

    Returns:
        dict với 6 module kết quả
    """
    cfg = SCENARIOS.get(scenario, SCENARIOS['S5'])

    # Chuẩn hóa budget về đơn vị nghìn tỷ cho tính toán
    budget_k = total_budget / 1000 if total_budget > 500 else float(total_budget)

    # Chạy 6 module
    allocation   = solve_m2_allocation(budget_k, cfg)
    gdp_forecast = solve_m1_macro(budget_k, cfg)
    priority     = solve_m3_priority(cfg)
    labor_impact = solve_m4_labor(cfg)
    topsis       = solve_m5_topsis(cfg)
    risk         = solve_m6_risk(allocation, gdp_forecast, cfg)

    return {
        'scenario':     scenario,
        'scenario_name': cfg['desc'],
        'description':  cfg['desc'],
        'budget':       total_budget,

        # Module 1 - GDP forecast
        'gdp_forecast': gdp_forecast,

        # Module 2 - Budget allocation
        'allocation': allocation,

        # Module 3 - Sector priority
        'priority': priority,

        # Module 4 - Labor impact
        'labor_impact': labor_impact,

        # Module 5 - TOPSIS region ranking
        'topsis': topsis,

        # Module 6 - Risk assessment
        'risk': risk,

        # Radar data tổng hợp
        'radar': {
            'dimensions': ['GDP', 'Hạ tầng', 'AI', 'Công bằng', 'Nhân lực', 'Rủi ro thấp'],
            'values': [
                min(95, max(20, gdp_forecast['gdp'][-1] / 20 * 100)),
                min(95, max(20, allocation['I'] / budget_k * 100 * 2)),
                min(95, max(20, allocation['AI'] / budget_k * 100 * 3)),
                min(95, max(20, allocation['H'] / budget_k * 100 * 3)),
                min(95, max(20, allocation['D'] / budget_k * 100 * 2.5)),
                min(95, max(20, (1 - risk['risk_score']) * 100)),
            ]
        },

        # Scenario comparison
        'all_scenarios': {
            sc: SCENARIOS[sc]['desc'] for sc in SCENARIOS
        },
    }


# Alias cho backwards compatibility
def solve_bai12(budget=100, w_gdp=0.35, w_equity=0.30, w_ai=0.25, risk_threshold=0.55):
    """Alias pipeline cho bài 12 (dùng trong test)."""
    return solve_bai12_dashboard(total_budget=budget * 1000, scenario='S5')

def solve_bai07(n_gen=200, pop_size=100, seed=42):
    try:
        problem = VietnamDigitalProblem()
        algorithm = NSGA2(pop_size=pop_size)
        res = pymoo_minimize(problem, algorithm, ('n_gen', n_gen), seed=seed, verbose=False)
        
        pareto_front = res.F
        pareto_solutions = res.X
        
        # Convert f1 back to positive GDP
        if pareto_front is not None:
            pareto_front[:, 0] = -pareto_front[:, 0]
            n_pareto = len(pareto_front)
        else:
            pareto_front = np.array([])
            pareto_solutions = np.array([])
            n_pareto = 0
            
    except Exception:
        n_pareto = 0
        pareto_front = np.array([])
        pareto_solutions = np.array([])

    topsis_res = {}
    opp_cost = {}
    
    if n_pareto > 0:
        # TOPSIS on Pareto Front
        # Weights: (0.40 cho tăng trưởng, 0.25 cho bao trùm, 0.20 cho môi trường, 0.15 cho an ninh)
        w = np.array([0.40, 0.25, 0.20, 0.15])
        is_benefit = [True, False, False, False] # GDP (max), MAD (min), Emission (min), Security (min)
        
        X = pareto_front.copy()
        norm = np.sqrt((X**2).sum(axis=0))
        R = X / (norm + 1e-9)
        V = R * w
        
        ideal = np.zeros(4)
        anti_ideal = np.zeros(4)
        for j in range(4):
            if is_benefit[j]:
                ideal[j] = V[:, j].max()
                anti_ideal[j] = V[:, j].min()
            else:
                ideal[j] = V[:, j].min()
                anti_ideal[j] = V[:, j].max()
                
        D_plus = np.sqrt(((V - ideal)**2).sum(axis=1))
        D_minus = np.sqrt(((V - anti_ideal)**2).sum(axis=1))
        C = D_minus / (D_plus + D_minus + 1e-9)
        
        best_idx = np.argmax(C)
        compromise_sol = X[best_idx]
        
        topsis_res = {
            'best_index': int(best_idx),
            'GDP': float(compromise_sol[0]),
            'Equity_MAD': float(compromise_sol[1]),
            'Emission': float(compromise_sol[2]),
            'Security': float(compromise_sol[3])
        }
        
        # Opportunity cost: Best GDP solution vs Compromise solution
        best_gdp_idx = np.argmax(X[:, 0])
        best_gdp_sol = X[best_gdp_idx]
        
        # % sacrifice in equity, env, security (how much worse is it?)
        # Since lower is better for f2, f3, f4, the best_gdp_sol will likely have higher values.
        # Sacrifice % = (best_gdp_val - compromise_val) / compromise_val * 100
        opp_cost = {
            'best_gdp_sol': {
                'GDP': float(best_gdp_sol[0]),
                'Equity_MAD': float(best_gdp_sol[1]),
                'Emission': float(best_gdp_sol[2]),
                'Security': float(best_gdp_sol[3])
            },
            'sacrifice': {
                'Equity_MAD_pct': float((best_gdp_sol[1] - compromise_sol[1]) / (compromise_sol[1] + 1e-9) * 100),
                'Emission_pct': float((best_gdp_sol[2] - compromise_sol[2]) / (compromise_sol[2] + 1e-9) * 100),
                'Security_pct': float((best_gdp_sol[3] - compromise_sol[3]) / (compromise_sol[3] + 1e-9) * 100)
            }
        }

    return {
        'n_pareto':        n_pareto,
        'pareto_front':    pareto_front.tolist(),
        'f1_gdp':    pareto_front[:, 0].tolist() if n_pareto > 0 else [],
        'f2_equity': pareto_front[:, 1].tolist() if n_pareto > 0 else [],
        'f3_env':    pareto_front[:, 2].tolist() if n_pareto > 0 else [],
        'f4_sec':    pareto_front[:, 3].tolist() if n_pareto > 0 else [],
        'pareto_table': [
            {'sol': i+1, 'GDP': round(float(pareto_front[i,0]),3),
             'Equity_MAD': round(float(pareto_front[i,1]),3),
             'Emission': round(float(pareto_front[i,2]),3),
             'Security': round(float(pareto_front[i,3]),3)}
            for i in range(min(n_pareto, 15))
        ],
        'topsis_compromise': topsis_res,
        'opportunity_cost': opp_cost
    }
