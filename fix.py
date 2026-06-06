import json

with open('notebooks/bai09_notebook.ipynb', 'r') as f:
    nb = json.load(f)

new_code = """def solve_bai09(data_dir=None, ai_adoption_rate=0.30, retraining_budget=15.0, transition_speed=0.5, new_job_multiplier=0.4):
    import pulp
    import numpy as np
    from data_loader import get_data
    
    data = get_data(data_dir)
    sectors = data.sectors_names_vi.tolist()
    m = len(sectors)
    
    employment = data.sectors_employment
    automation_risk = data.sectors_automation_risk
    
    # Supply = jobs lost
    supply = employment * automation_risk * ai_adoption_rate
    # Demand = new jobs
    demand = supply * new_job_multiplier * 1.5
    
    # Dummy cost matrix (based on index diff)
    cost = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            cost[i,j] = 1.0 + abs(i - j) * 0.5
            
    prob = pulp.LpProblem("Retraining_Optimization", pulp.LpMaximize)
    X = pulp.LpVariable.dicts("x", ((i, j) for i in range(m) for j in range(m)), lowBound=0, cat='Continuous')
    
    prob += pulp.lpSum(X[i, j] for i in range(m) for j in range(m))
    
    for i in range(m):
        prob += pulp.lpSum(X[i, j] for j in range(m)) <= float(supply[i])
        
    for j in range(m):
        prob += pulp.lpSum(X[i, j] for i in range(m)) <= float(demand[j])
        
    prob += pulp.lpSum(cost[i][j] * X[i, j] for i in range(m) for j in range(m)) <= float(retraining_budget) * 1000.0 * float(transition_speed)
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    total_net = 0
    sector_table = {}
    sankey_source = []
    sankey_target = []
    sankey_value = []
    
    offset = m 
    
    if pulp.LpStatus[prob.status] == 'Optimal':
        for i in range(m):
            lost = float(supply[i])
            transferred_out = sum(X[i,j].varValue for j in range(m))
            transferred_in = sum(X[k,i].varValue for k in range(m))
            new_job_here = float(demand[i])
            
            net = new_job_here + transferred_in - lost
            
            sector_table[sectors[i]] = {
                'Mất việc': lost,
                'Việc mới (AI)': new_job_here,
                'Đào tạo đi': transferred_out,
                'Nhận đào tạo': transferred_in,
                'NetJob': net
            }
            total_net += net
            
            for j in range(m):
                val = float(X[i,j].varValue)
                if val > 1e-3:
                    sankey_source.append(i)
                    sankey_target.append(offset + j)
                    sankey_value.append(val)
                    
    threshold_xH2 = float(supply[1]) * 0.8 
    
    ext_feasible = True
    for i in range(m):
        if pulp.LpStatus[prob.status] == 'Optimal':
            net = sector_table[sectors[i]]['NetJob']
            if net < -0.05 * float(employment[i]):
                ext_feasible = False
                break
        else:
            ext_feasible = False
            break

    nodes = sectors + [s + " (Nhận)" for s in sectors]
    
    return {
        'total_net': total_net,
        'sector_table': sector_table,
        'threshold_xH2': threshold_xH2,
        'sankey_nodes': nodes,
        'sankey_links': {
            'source': sankey_source,
            'target': sankey_target,
            'value': sankey_value
        },
        'ext_feasible': ext_feasible
    }
"""

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'def solve_bai09' in ''.join(cell['source']):
        # Replace the source with lines
        cell['source'] = [line + '\n' for line in new_code.split('\n')]

with open('notebooks/bai09_notebook.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
