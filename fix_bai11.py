import json

with open('/Users/thaochie/Downloads/DinhThiDiemQuynh/src/rl_env.py', 'r') as f:
    dinh_code = f.read()

with open('notebooks/bai11_notebook.ipynb', 'r') as f:
    nb = json.load(f)

# Clear all code cells, and put the new code in one cell
new_cells = []
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        new_cells.append(cell)

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + '\n' for line in dinh_code.split('\n')]
})

nb['cells'] = new_cells

with open('notebooks/bai11_notebook.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
