"""
Mô-đun tối ưu hóa đã được tái cấu trúc.
Giờ đây, logic giải quyết các bài toán được thực thi ĐẦY ĐỦ trong các Jupyter Notebooks (.ipynb)
nhằm đáp ứng yêu cầu tài liệu học thuật. File này chỉ đóng vai trò kế thừa (wrapper) 
để Dashboard gọi lại các hàm từ Notebook mà không phá vỡ UI.
"""

from ipynb.fs.full.notebooks.bai01_notebook import solve_bai01
from ipynb.fs.full.notebooks.bai02_notebook import solve_bai02
from ipynb.fs.full.notebooks.bai03_notebook import solve_bai03
from ipynb.fs.full.notebooks.bai04_notebook import solve_bai04
from ipynb.fs.full.notebooks.bai05_notebook import solve_bai05
from ipynb.fs.full.notebooks.bai06_notebook import solve_bai06
from ipynb.fs.full.notebooks.bai07_notebook import solve_bai07
from ipynb.fs.full.notebooks.bai08_notebook import solve_bai08
from ipynb.fs.full.notebooks.bai09_notebook import solve_bai09
from ipynb.fs.full.notebooks.bai10_notebook import solve_bai10
from ipynb.fs.full.notebooks.bai12_notebook import solve_bai12, solve_bai12_dashboard
