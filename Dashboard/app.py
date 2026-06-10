import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.optimization import solve_bai01, solve_bai02, solve_bai03, solve_bai04, solve_bai05, solve_bai06, solve_bai07, solve_bai08, solve_bai09, solve_bai10, solve_bai12, solve_bai12_dashboard
from src.rl_env import solve_bai11

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

st.set_page_config(page_title="AIDEOM-VN Streamlit", layout="wide", initial_sidebar_state="collapsed")

try:
    with open("src/style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except Exception:
    pass

from streamlit_option_menu import option_menu

pages = [
    "Bài 1", "Bài 2", "Bài 3",
    "Bài 4", "Bài 5", "Bài 6",
    "Bài 7", "Bài 8", "Bài 9",
    "Bài 10", "Bài 11", "Bài 12"
]

st.markdown("""
<div style='text-align: center; margin-bottom: 20px;'>
    <h4 style='margin-bottom: 0px; color: #553C9A;'>Trường Đại học Kinh tế - ĐHQGHN</h4>
    <h5 style='margin-bottom: 0px; color: #4A5568;'>Khoa Kinh tế Phát triển</h5>
    <p style='font-style: italic; color: #718096;'>Sinh viên Nguyễn Yến Nhi</p>
</div>
""", unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=pages,
    icons=['1-circle', '2-circle', '3-circle', '4-circle', '5-circle', '6-circle', '7-circle', '8-circle', '9-circle', '10-circle', '11-circle', '12-circle'],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#FFFFFF", "border-radius": "12px", "box-shadow": "0 2px 4px rgba(0,0,0,0.05)", "margin-bottom": "24px"},
        "icon": {"color": "#CDB4DB", "font-size": "16px"}, 
        "nav-link": {"font-size": "14px", "text-align": "center", "margin":"0px", "--hover-color": "#F4F7FC"},
        "nav-link-selected": {"background-color": "#CDB4DB", "color": "white"},
    }
)

page = selected

LINE_COLOR = '#CDB4DB'

# ─── Bài 1 ───
if page == pages[0]:
    st.title("Bài 1: Hàm sản xuất Cobb-Douglas mở rộng")
    with st.expander("⚙️ Bảng điều khiển thông số (Topbar)", expanded=True):
        c1, c2, c3 = st.columns(3)
        alpha = c1.slider("α (Vốn K)", 0.10, 0.50, 0.33, 0.01)
        beta  = c2.slider("β (Lao động L)", 0.10, 0.60, 0.42, 0.01)
        gamma = c3.slider("γ (Số hóa D)", 0.01, 0.30, 0.10, 0.01)
        c1, c2, c3 = st.columns(3)
        delta = c1.slider("δ (AI)", 0.01, 0.20, 0.08, 0.01)
        theta = c2.slider("θ (Nhân lực H)", 0.01, 0.20, 0.07, 0.01)
    res = solve_bai01(DATA_DIR, alpha, beta, gamma, delta, theta)

    # --- (1) Đồ thị A_t theo năm ---
    with st.container(border=True):
        st.subheader("1. Xu hướng TFP (A_t) theo năm")
        fig_at = px.line(x=res['years'], y=res['A_t'], title="Năng suất nhân tố tổng hợp A_t (2020-2025)", markers=True,
                         labels={'x': 'Năm', 'y': 'A_t'})
        fig_at.update_traces(line_color=LINE_COLOR)
        st.plotly_chart(fig_at, use_container_width=True)

        # --- (2) So sánh Ŷ vs Y thực tế + MAPE ---
    with st.container(border=True):
        st.subheader("2. So sánh Ŷ dự báo vs Y thực tế")
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(fill='tozeroy', x=res['years'], y=res['Y_actual'], name='Y thực tế', mode='lines+markers', line=dict(color='#E53E3E')))
        fig_compare.add_trace(go.Scatter(fill='tozeroy', x=res['years'], y=res['Y_hat'], name='Ŷ dự báo (A̅ trung bình)', mode='lines+markers', line=dict(color=LINE_COLOR, dash='dash')))
        fig_compare.update_layout(title="Y thực tế vs Ŷ dự báo (Cobb-Douglas)", xaxis_title="Năm", yaxis_title="GDP (Nghìn tỷ VND)")
        st.plotly_chart(fig_compare, use_container_width=True)
        st.metric("MAPE (Mean Absolute Percentage Error)", f"{res['mape']:.2f}%")

        # --- (3) Phân rã tăng trưởng: bảng + biểu đồ cột ---
    with st.container(border=True):
        st.subheader("3. Phân rã tăng trưởng GDP 2020-2025")
        contrib = res['contrib_pct']
        df_contrib = pd.DataFrame({'Yếu tố': list(contrib.keys()), 'Đóng góp (%)': list(contrib.values())})
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(df_contrib, use_container_width=True, hide_index=True)
        with col2:
            fig_decomp = px.bar(df_contrib, x='Yếu tố', y='Đóng góp (%)',
                                title="Đóng góp vào tăng trưởng GDP (%)", color='Yếu tố',
                                color_discrete_sequence=px.colors.sequential.Blues_r)
            fig_decomp.update_layout(showlegend=False)
            st.plotly_chart(fig_decomp, use_container_width=True)

        # --- (4) Dự báo GDP đến 2030 ---
    with st.container(border=True):
        st.subheader("4. Dự báo GDP Việt Nam đến 2030")
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(fill='tozeroy', x=res['years'], y=res['Y_actual'], name='Y thực tế (2020-2025)', mode='lines+markers', line=dict(color='#E53E3E')))
        fig_forecast.add_trace(go.Scatter(fill='tozeroy', x=res['forecast_years'], y=res['forecast_series'], name='Dự báo cơ sở', mode='lines+markers', line=dict(color=LINE_COLOR)))
        fig_forecast.add_trace(go.Scatter(fill='tozeroy', x=res['forecast_years'], y=res['forecast_high_tfp'], name='Kịch bản TFP cao', mode='lines+markers', line=dict(color='#48BB78', dash='dash')))
        fig_forecast.add_trace(go.Scatter(fill='tozeroy', x=res['forecast_years'], y=res['forecast_ai_fast'], name='Kịch bản AI nhanh', mode='lines+markers', line=dict(color='#ED8936', dash='dot')))
        fig_forecast.update_layout(title="Dự báo GDP Việt Nam 2026-2030", xaxis_title="Năm", yaxis_title="GDP (Nghìn tỷ VND)")
        st.plotly_chart(fig_forecast, use_container_width=True)
        st.metric("GDP dự báo 2030 (Kịch bản cơ sở)", f"{res['gdp_2030']:,.0f} nghìn tỷ VND")

# ─── Bài 2 ───
elif page == pages[1]:
    st.title("Bài 2: Quy hoạch tuyến tính phân bổ ngân sách")
    with st.expander("⚙️ Bảng điều khiển thông số (Topbar)", expanded=True):
        c1, c2, c3 = st.columns(3)
        tb  = c1.slider("Tổng ngân sách", 50, 200, 100, 5)
        mi  = c2.slider("Min I (Hạ tầng)", 5, 50, 25)
        mai = c3.slider("Min AI", 5, 50, 15)
    res = solve_bai02(budget=tb, min_I=mi, min_AI=mai)

    # --- (1) Giải bằng linprog ---
    with st.container(border=True):
        st.subheader("1. Giải bằng scipy.optimize.linprog")
        if res['status'] == 'Optimal':
            st.success(f"Khả thi. Z* = {res['Z']:.2f}")
            fig = px.bar(x=list(res['allocation'].keys()), y=list(res['allocation'].values()),
                         title="Phân bổ tối ưu (linprog)", color=list(res['allocation'].keys()),
                         color_discrete_sequence=px.colors.sequential.Blues_r)
            fig.update_layout(showlegend=False, xaxis_title="Hạng mục", yaxis_title="Nghìn tỷ VND")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Không khả thi với các ràng buộc này.")

        # --- (2) Giải bằng PuLP + Dual values ---
    with st.container(border=True):
        st.subheader("2. Giải bằng PuLP – Giá đối ngẫu (Shadow Prices)")
        if res['dual_values']:
            dual_labels = {
                'C1_Budget': 'C1: Ngân sách tổng',
                'C2_Tech35': 'C2: Tỷ lệ công nghệ ≥ 35%',
                'C3_MinI': 'C3: Min Hạ tầng (I)',
                'C4_MinAI': 'C4: Min AI',
                'C5_MinH': 'C5: Min Nhân lực (H)',
                'C6_MinRD': 'C6: Min R&D',
            }
            df_dual = pd.DataFrame({
                'Ràng buộc': [dual_labels.get(k, k) for k in res['dual_values'].keys()],
                'Shadow Price (π)': list(res['dual_values'].values())
            })
            st.dataframe(df_dual, use_container_width=True, hide_index=True)
            budget_sp = res['dual_values'].get('C1_Budget', 0)
            st.info(f"**Ý nghĩa chính sách:** Shadow price của ràng buộc ngân sách tổng = **{budget_sp}**. "
                    f"Điều này có nghĩa: nếu tăng thêm 1 đơn vị ngân sách, giá trị hàm mục tiêu Z* sẽ tăng thêm {budget_sp} đơn vị. "
                    f"Đây là cơ sở để Chính phủ đánh giá hiệu quả biên của việc tăng ngân sách.")
        else:
            st.warning("Không có dual values (bài toán không khả thi hoặc solver không hỗ trợ).")

        # --- (3) Phân tích độ nhạy: đường cong Z*(B) ---
    with st.container(border=True):
        st.subheader("3. Phân tích độ nhạy – Đường cong Z*(B)")
        fig_sens = px.line(x=res['sensitivity_budgets'], y=res['sensitivity_z'],
                           title="Z* theo Ngân sách tổng B", markers=True,
                           labels={'x': 'Ngân sách B (Nghìn tỷ VND)', 'y': 'Z*'})
        fig_sens.update_traces(line_color=LINE_COLOR)
        st.plotly_chart(fig_sens, use_container_width=True)

        # --- (4) Kịch bản x₃ ≥ 30 ---
    with st.container(border=True):
        st.subheader("4. Kịch bản: Ưu tiên nhân lực số (x₃ ≥ 30)")
        sc = res['scenario_x3']
        if sc['status'] == 'Optimal':
            col1, col2 = st.columns(2)
            col1.metric("Z* gốc (x₃ ≥ 20)", f"{res['Z']:.2f}")
            col2.metric("Z* mới (x₃ ≥ 30)", f"{sc['Z']:.2f}", delta=f"{sc['Z'] - res['Z']:.2f}")
            df_compare = pd.DataFrame({
                'Hạng mục': list(res['allocation'].keys()),
                'Gốc (x₃≥20)': list(res['allocation'].values()),
                'Mới (x₃≥30)': list(sc['allocation'].values()),
            })
            st.dataframe(df_compare, use_container_width=True, hide_index=True)
            st.success(f"Bài toán vẫn **khả thi**. Z* thay đổi: {res['Z']:.2f} → {sc['Z']:.2f} (Δ = {sc['Z'] - res['Z']:.2f}).")
        else:
            st.error("Bài toán **không còn khả thi** khi thêm ràng buộc x₃ ≥ 30.")

# ─── Bài 3 ───
elif page == pages[2]:
    st.title("Bài 3: Xây dựng chỉ số ưu tiên ngành")
    with st.expander("⚙️ Bảng điều khiển thông số (Topbar)", expanded=True):
        c1, c2, c3 = st.columns(3)
        w_growth = c1.slider("w Tăng trưởng", 0.0, 0.5, 0.15, 0.01)
        w_prod = c2.slider("w Năng suất", 0.0, 0.5, 0.15, 0.01)
        w_spill = c3.slider("w Lan tỏa", 0.0, 0.5, 0.20, 0.01)
        c1, c2, c3 = st.columns(3)
        w_exp = c1.slider("w Xuất khẩu", 0.0, 0.5, 0.15, 0.01)
        w_emp = c2.slider("w Việc làm", 0.0, 0.5, 0.10, 0.01)
        w_ai = c3.slider("w AI Readiness", 0.0, 0.5, 0.20, 0.01)
        c1, c2, c3 = st.columns(3)
        w_risk = c1.slider("w Rủi ro (Penalty)", 0.0, 0.5, 0.15, 0.01)
    res = solve_bai03(w_growth=w_growth, w_productivity=w_prod, w_spillover=w_spill, w_export=w_exp, w_employment=w_emp, w_ai=w_ai, w_risk=w_risk)

    # --- (1) Ma trận chuẩn hóa min-max ---
    with st.container(border=True):
        st.subheader("1. Ma trận chuẩn hóa Min-Max (đảo dấu Rủi ro)")
        df_norm = pd.DataFrame(res['norm_matrix'], columns=res['col_names'], index=res['sectors'])
        st.dataframe(df_norm.style.format("{:.4f}"), use_container_width=True)

        # --- (2) Xếp hạng Priority ---
    with st.container(border=True):
        st.subheader("2. Xếp hạng 10 ngành theo Priority")
        names = [r['sector_name_vi'] for r in res['ranking']]
        scores = [r['Priority'] for r in res['ranking']]
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.bar(x=names, y=scores, title="Chỉ số ưu tiên theo ngành (giảm dần)", color=scores, color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            df_rank = pd.DataFrame(res['ranking'])[['rank', 'sector_name_vi', 'Priority']]
            df_rank.columns = ['Hạng', 'Ngành', 'Priority']
            st.dataframe(df_rank, use_container_width=True, hide_index=True)

        # --- (3) Phân tích độ nhạy a₆ - Heatmap ---
    with st.container(border=True):
        st.subheader("3. Phân tích độ nhạy: Trọng số AI Readiness (a₆)")
        df_heatmap = pd.DataFrame(res['heatmap_data'], columns=[f"a₆={v}" for v in res['a6_values']], index=res['sectors'])
        fig_hm = px.imshow(df_heatmap, title="Heatmap Priority theo a₆ (AI Readiness)",
                           labels=dict(x="Trọng số a₆", y="Ngành", color="Priority"),
                           color_continuous_scale="Blues", aspect="auto", text_auto=".3f")
        st.plotly_chart(fig_hm, use_container_width=True)
        st.info("**Top-3 theo từng a₆:** " + " | ".join([f"a₆={k}: {', '.join(v)}" for k, v in res['sensitivity_top3'].items()]))

        # --- (4) So sánh 2 bộ trọng số ---
    with st.container(border=True):
        st.subheader("4. So sánh: Tăng trưởng vs Bao trùm")
        sc = res['scenario_comparison']
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🚀 Định hướng Tăng trưởng**")
            st.markdown(f"Top 3: **{', '.join(sc['growth']['top3'])}**")
            fig_g = px.bar(x=list(sc['growth']['scores'].keys()), y=list(sc['growth']['scores'].values()),
                           title="Điểm ưu tiên – Tăng trưởng", color=list(sc['growth']['scores'].values()),
                           color_continuous_scale='Oranges')
            st.plotly_chart(fig_g, use_container_width=True)
        with col2:
            st.markdown("**🤝 Định hướng Bao trùm**")
            st.markdown(f"Top 3: **{', '.join(sc['inclusive']['top3'])}**")
            fig_i = px.bar(x=list(sc['inclusive']['scores'].keys()), y=list(sc['inclusive']['scores'].values()),
                           title="Điểm ưu tiên – Bao trùm", color=list(sc['inclusive']['scores'].values()),
                           color_continuous_scale='Greens')
            st.plotly_chart(fig_i, use_container_width=True)

# ─── Bài 4 ───
elif page == pages[3]:
    st.title("Bài 4: LP Phân bổ ngân sách ngành - vùng")
    with st.expander("⚙️ Bảng điều khiển thông số (Topbar)", expanded=True):
        c1, c2, c3 = st.columns(3)
        budget = c1.slider("Tổng ngân sách (Tỷ VND)", 20000, 80000, 50000, 5000)
        w_gdp = c2.slider("w GDP", 0.0, 1.0, 0.40, 0.05)
        w_equity = c3.slider("w Công bằng", 0.0, 1.0, 0.25, 0.05)
        c1, c2, c3 = st.columns(3)
        w_ai = c1.slider("w AI", 0.0, 1.0, 0.20, 0.05)
    res = solve_bai04(budget, w_gdp=w_gdp, w_equity=w_equity, w_ai=w_ai)

    # --- (1) PuLP + CBC: Ma trận 6×4 + Z* ---
    with st.container(border=True):
        st.subheader("1. Giải bằng PuLP (CBC) – Ma trận phân bổ 6×4")
        if res['status'] == 'Optimal':
            st.success(f"Z* (PuLP) = {res['Z']:,.1f}")
            df_pulp = pd.DataFrame(res['allocation']).T
            st.dataframe(df_pulp.style.format("{:,.1f}"), use_container_width=True)
        else:
            st.error("Không khả thi.")

        # --- (2) CVXPY: So sánh ---
    with st.container(border=True):
        st.subheader("2. Giải bằng CVXPY – So sánh kết quả")
        if res['cvxpy_ok']:
            st.success(f"Z* (CVXPY) = {res['cvxpy_z']:,.1f}")
            df_cvxpy = pd.DataFrame(res['cvxpy_alloc']).T
            st.dataframe(df_cvxpy.style.format("{:,.1f}"), use_container_width=True)
            diff_z = abs(res['Z'] - res['cvxpy_z'])
            if diff_z < 1.0:
                st.info(f"✅ Hai phương pháp cho kết quả **giống nhau** (chênh lệch Z*: {diff_z:,.2f} ≈ 0).")
            else:
                st.warning(f"⚠️ Chênh lệch Z*: {diff_z:,.2f}. Sự khác biệt nhỏ do solver SCS (CVXPY) là solver xấp xỉ, trong khi CBC (PuLP) giải chính xác.")
        else:
            st.warning("CVXPY không khả dụng hoặc không tìm được nghiệm tối ưu. Đảm bảo đã cài `cvxpy`.")

        # --- (3) Heatmap phân bổ tối ưu ---
    with st.container(border=True):
        st.subheader("3. Heatmap phân bổ tối ưu")
        if res['status'] == 'Optimal':
            df_hm = pd.DataFrame(res['allocation']).T
            fig = px.imshow(df_hm,
                            labels=dict(x="Hạng mục", y="Vùng", color="Ngân sách (Tỷ VND)"),
                            title="Heatmap Phân bổ Ngân sách theo Vùng và Hạng mục",
                            color_continuous_scale="Blues", aspect="auto", text_auto=",.0f")
            st.plotly_chart(fig, use_container_width=True)
            # Nhận xét
            row_sums = df_hm.sum(axis=1)
            max_region = row_sums.idxmax()
            st.info(f"**Nhận xét:** Vùng nhận ngân sách nhiều nhất: **{max_region}** ({row_sums[max_region]:,.0f} tỷ VND). "
                    f"Mỗi vùng có hạng mục ưu tiên riêng dựa trên hệ số β tương ứng.")

        # --- (4) Bỏ C5: Chi phí kinh tế của công bằng ---
    with st.container(border=True):
        st.subheader("4. Chi phí kinh tế của công bằng vùng miền (bỏ C5)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Z* có C5 (Công bằng)", f"{res['Z']:,.1f}")
        col2.metric("Z* không C5", f"{res['no_equity_z']:,.1f}")
        col3.metric("Chi phí công bằng (ΔZ)", f"{res['equity_cost']:,.1f} tỷ VND", delta=f"-{res['equity_cost']:,.1f}")
        if res['equity_cost'] > 0:
            st.warning(f"Ràng buộc công bằng vùng miền (C5) làm giảm GDP gain **{res['equity_cost']:,.1f} tỷ VND**. "
                       f"Đây là 'chi phí' kinh tế mà xã hội trả để đảm bảo phát triển đồng đều giữa các vùng.")
        else:
            st.info("Ràng buộc công bằng không ảnh hưởng đến Z* (không có chi phí công bằng).")
        # Heatmap so sánh
        df_noeq = pd.DataFrame(res['no_equity_alloc']).T
        fig_noeq = px.imshow(df_noeq,
                             labels=dict(x="Hạng mục", y="Vùng", color="Ngân sách (Tỷ VND)"),
                             title="Heatmap KHÔNG có ràng buộc công bằng (bỏ C5)",
                             color_continuous_scale="Reds", aspect="auto", text_auto=",.0f")
        st.plotly_chart(fig_noeq, use_container_width=True)

        # --- (5) Bỏ C3: Chi phí kinh tế của giới hạn trần ngân sách vùng ---
    with st.container(border=True):
        st.subheader("5. Chi phí kinh tế của công bằng vùng miền (bỏ C3)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Z* có C3 (Trần 12,000 tỷ)", f"{res['Z']:,.1f}")
        col2.metric("Z* không C3", f"{res.get('no_c3_z', 0):,.1f}")
        c3_cost = res.get('c3_cost', 0)
        col3.metric("Chi phí giới hạn trần (ΔZ)", f"{c3_cost:,.1f} tỷ VND", delta=f"-{c3_cost:,.1f}")
        if c3_cost > 0:
            st.warning(f"Ràng buộc trần ngân sách mỗi vùng (C3 <= 12,000 tỷ) làm giảm GDP gain **{c3_cost:,.1f} tỷ VND**. "
                       f"Việc bỏ C3 sẽ dẫn đến ngân sách tập trung toàn bộ vào các vùng có hiệu suất cao, gây mất cân đối vùng miền.")
        else:
            st.info("Ràng buộc C3 không ảnh hưởng đến Z* hiện tại.")
            
        # Heatmap so sánh bỏ C3
        df_no_c3 = pd.DataFrame(res.get('no_c3_alloc', {})).T
        if not df_no_c3.empty:
            fig_no_c3 = px.imshow(df_no_c3,
                                 labels=dict(x="Hạng mục", y="Vùng", color="Ngân sách (Tỷ VND)"),
                                 title="Heatmap KHÔNG có ràng buộc giới hạn trần (bỏ C3)",
                                 color_continuous_scale="Purples", aspect="auto", text_auto=",.0f")
            st.plotly_chart(fig_no_c3, use_container_width=True)

# ─── Bài 5 ───
elif page == pages[4]:
    st.title("Bài 5: Tối ưu hóa danh mục dự án đầu tư công")
    with st.expander("⚙️ Bảng điều khiển thông số (Topbar)", expanded=True):
        c1, c2, c3 = st.columns(3)
        budget = c1.slider("Ngân sách tổng (Tỷ VND)", 40000, 120000, 80000, 5000)
        w_gdp = c2.slider("w GDP (b5)", 0.0, 1.0, 0.40, 0.05)
        w_equity = c3.slider("w Công bằng (b5)", 0.0, 1.0, 0.30, 0.05)
        c1, c2, c3 = st.columns(3)
        w_ai = c1.slider("w AI (b5)", 0.0, 1.0, 0.30, 0.05)
    res = solve_bai05(budget, w_gdp=w_gdp, w_equity=w_equity, w_ai=w_ai)
    
    with st.container(border=True):
        st.subheader("1. Kết quả giải gốc (PuLP - CBC)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tổng Lợi ích (Z*)", f"{res['Z']:,.2f}")
        c2.metric("Tổng Chi phí", f"{res['cost']:,.2f}")
        c3.metric("Tổng NPV", f"{res['total_npv']:,.2f}")
        c4.metric("NPV Biên (Z*/Cost)", f"{res['npv_margin']:.4f}")
        if res['selected']:
            df_proj = pd.DataFrame(res['projects']).T
            st.dataframe(df_proj.style.format("{:.2f}"), use_container_width=True)
        else:
            st.warning("Infeasible - Không có dự án nào được chọn!")

    with st.container(border=True):
        st.subheader("2. Phân tích: Nới ngân sách lên 100.000 tỷ")
        r100 = res['res_100k']
        c1, c2 = st.columns(2)
        c1.metric("Lợi ích Z* (100k)", f"{r100['Z']:,.2f}", delta=f"{r100['Z'] - res['Z']:,.2f}")
        c2.write(f"**Các dự án được chọn:** {', '.join(r100['selected'])}")

    with st.container(border=True):
        st.subheader("3. Phân tích: Bắt buộc chọn P1 và P2 (Redundancy)")
        rP = res['res_p1p2']
        if rP['status'] == 'Optimal':
            c1, c2 = st.columns(2)
            c1.metric("Lợi ích Z* (P1+P2)", f"{rP['Z']:,.2f}", delta=f"{rP['Z'] - res['Z']:,.2f}")
            c2.write(f"**Các dự án được chọn:** {', '.join(rP['selected'])}")
        else:
            st.error("❌ Không khả thi (Infeasible) do vi phạm ràng buộc ngân sách hoặc ràng buộc loại trừ C3.")

    with st.container(border=True):
        st.subheader("4. Mở rộng: Rủi ro dự án (Tối đa hóa lợi ích kỳ vọng E[Z])")
        rR = res['res_risk']
        c1, c2 = st.columns(2)
        c1.metric("E[Z] Kỳ vọng", f"{rR['Z']:,.2f}")
        c2.write(f"**Tập dự án chọn an toàn:** {', '.join(rR['selected'])}")
        with st.expander("Bảng xác suất hoàn thành đúng tiến độ"):
            st.json(res['prob_completion'])
# ─── Bài 6 ───
elif page == pages[5]:
    st.title("Bài 6: Đánh giá đa tiêu chí xếp hạng Vùng")
    with st.expander("⚙️ Bảng điều khiển thông số (Topbar)", expanded=True):
        mode = st.radio("Chế độ trọng số:", ["Chuyên gia (Manual)", "Entropy (Tự động)"], index=0)
        weight_mode = 0 if mode == "Entropy (Tự động)" else 1
        w_expert = None
        if weight_mode == 1:
            st.write("Cấu hình trọng số chuyên gia:")
            c1, c2, c3 = st.columns(3)
            w_grdp = c1.slider("w GRDP/capita", 0.0, 0.5, 0.10, 0.01)
            w_digi = c2.slider("w Digital", 0.0, 0.5, 0.10, 0.01)
            w_ai = c3.slider("w AI ready", 0.0, 0.5, 0.15, 0.01)
            c1, c2, c3 = st.columns(3)
            w_labor = c1.slider("w Lao động", 0.0, 0.5, 0.20, 0.01)
            w_rd = c2.slider("w R&D", 0.0, 0.5, 0.15, 0.01)
            w_gini = c3.slider("w Gini (cost)", 0.0, 0.5, 0.15, 0.01)
            # Dữ liệu chỉ có 6 biến thay vì 8, do đó lấy 6 trọng số đầu và chuẩn hóa nếu cần
            w_expert = [w_grdp, w_digi, w_ai, w_labor, w_rd, w_gini]
            s = sum(w_expert)
            if s > 0: w_expert = [x/s for x in w_expert]
            
    res = solve_bai06(w_manual=w_expert, weight_mode=weight_mode)
    
    with st.container(border=True):
        st.subheader("1 & 2. Xếp hạng vùng theo TOPSIS")
        names = [r['region_name_vi'] for r in res['ranking']]
        scores = [r['TOPSIS'] for r in res['ranking']]
        fig = px.bar(x=names, y=scores, title=f"Điểm TOPSIS (Trọng số: {mode})", color=scores, color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.subheader("3. Phân tích độ nhạy: Thay đổi trọng số $w_{AI}$ (0.10 - 0.40)")
        # Prepare data for heatmap
        sens_data = []
        for w_val, ranks in res['sensitivity'].items():
            for region, score in ranks.items():
                sens_data.append({"w_AI": w_val, "Region": region, "Score": score})
        df_sens = pd.DataFrame(sens_data)
        df_pivot = df_sens.pivot(index="Region", columns="w_AI", values="Score")
    
        fig_heat = px.imshow(df_pivot, text_auto=".3f", aspect="auto", 
                             title="Biến động điểm TOPSIS khi thay đổi trọng số AI",
                             color_continuous_scale="Viridis")
        st.plotly_chart(fig_heat, use_container_width=True)

    with st.container(border=True):
        st.subheader("4. So sánh phương pháp: TOPSIS vs AHP đơn giản")
        df_comp = pd.DataFrame(res['ranks_comparison'])
        st.dataframe(df_comp, use_container_width=True)

# ─── Bài 7 ───
elif page == pages[6]:
    st.title("Bài 7: Tối ưu đa mục tiêu (NSGA-II)")
    with st.expander("⚙️ Bảng điều khiển thông số (Topbar)", expanded=True):
        c1, c2, c3 = st.columns(3)
        n_gen = c1.slider("Số thế hệ (generations)", 50, 300, 200, 50)
        pop_size = c2.slider("Kích thước quần thể", 50, 200, 100, 50)
    
    with st.spinner("Đang chạy NSGA-II tối ưu hóa 4 mục tiêu..."):
        res = solve_bai07(n_gen, pop_size)
    
    if res['n_pareto'] == 0:
        st.error("Không tìm thấy nghiệm Pareto hợp lệ. Vui lòng kiểm tra lại solver hoặc thông số.")
    else:
        with st.container(border=True):
            st.subheader("1. Tập Pareto: 3 mục tiêu đầu (Scatter 3D)")
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=res['f1_gdp'], y=res['f2_equity'], z=res['f3_env'],
                mode='markers', marker=dict(size=4, color=res['f1_gdp'], colorscale='Blues', opacity=0.8)
            )])
            fig_3d.update_layout(title='Pareto Front 3D', scene=dict(xaxis_title='GDP (f1)', yaxis_title='Equity_MAD (f2)', zaxis_title='Emission (f3)'))
            st.plotly_chart(fig_3d, use_container_width=True)

        with st.container(border=True):
            st.subheader("2. Biểu đồ Tọa độ song song (Parallel Coordinates) - 4 Mục tiêu")
            fig_parc = go.Figure(data=
                go.Parcoords(
                    line = dict(color = res['f1_gdp'], colorscale = 'Viridis', showscale = True),
                    dimensions = list([
                        dict(range = [min(res['f1_gdp']), max(res['f1_gdp'])], label = 'GDP (Max)', values = res['f1_gdp']),
                        dict(range = [max(res['f2_equity']), min(res['f2_equity'])], label = 'Equity_MAD (Min)', values = res['f2_equity']),
                        dict(range = [max(res['f3_env']), min(res['f3_env'])], label = 'Emission (Min)', values = res['f3_env']),
                        dict(range = [max(res['f4_sec']), min(res['f4_sec'])], label = 'Security (Min)', values = res['f4_sec'])
                    ])
                )
            )
            st.plotly_chart(fig_parc, use_container_width=True)

        with st.container(border=True):
            st.subheader("3. Nghiệm thỏa hiệp duy nhất (TOPSIS)")
            top = res['topsis_compromise']
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("GDP", f"{top['GDP']:,.2f}")
            c2.metric("Equity_MAD", f"{top['Equity_MAD']:,.2f}")
            c3.metric("Emission", f"{top['Emission']:,.2f}")
            c4.metric("Security", f"{top['Security']:,.2f}")
            st.write(f"*(Chọn từ {res['n_pareto']} nghiệm với trọng số: 0.40 GDP, 0.25 Bao trùm, 0.20 Môi trường, 0.15 An ninh)*")

        with st.container(border=True):
            st.subheader("4. Phân tích Chi phí cơ hội (Opportunity Cost)")
            opp = res['opportunity_cost']
            c1, c2 = st.columns(2)
            c1.info(f"**Nghiệm Tăng trưởng cao nhất:**\n- GDP: {opp['best_gdp_sol']['GDP']:,.2f}\n- Bao trùm: {opp['best_gdp_sol']['Equity_MAD']:,.2f}\n- Môi trường: {opp['best_gdp_sol']['Emission']:,.2f}\n- An ninh: {opp['best_gdp_sol']['Security']:,.2f}")
        
            c2.warning("**Chi phí cơ hội so với nghiệm thỏa hiệp:**\n\n" +
                       f"- **Bao trùm (MAD)**: Xấu hơn **{opp['sacrifice']['Equity_MAD_pct']:,.1f}%**\n" +
                       f"- **Môi trường**: Xấu hơn **{opp['sacrifice']['Emission_pct']:,.1f}%**\n" +
                       f"- **An ninh**: Xấu hơn **{opp['sacrifice']['Security_pct']:,.1f}%**\n\n" +
                       "*(Việc chạy theo tăng trưởng thuần túy làm hi sinh đáng kể các mục tiêu còn lại)*")

# ─── Bài 8 ───
elif page == pages[7]:
    st.title("Bài 8: Tối ưu động liên thời gian")
    with st.expander("⚙️ Bảng điều khiển thông số (Topbar)", expanded=True):
        c1, c2, c3 = st.columns(3)
        discount = c1.slider("Chiết khấu δ", 0.0, 0.15, 0.05, 0.01)
        cap_growth = c2.slider("Tăng vốn/năm", 0.02, 0.12, 0.06, 0.01)
        target_ai = c3.slider("Mục tiêu AI 2035", 0.5, 1.0, 0.85, 0.05)
        c1, c2, c3 = st.columns(3)
        bud_growth = c1.slider("Tăng NS/năm", 0.03, 0.15, 0.08, 0.01)
    res = solve_bai08(discount=discount, capital_growth=cap_growth, target_ai=target_ai, budget_growth=bud_growth)
    
    with st.container(border=True):
        st.subheader("1. Giải bằng scipy.optimize.minimize (SLSQP)")
        st.info(f"**Tổng phúc lợi tối ưu (Z*):** {res['welfare_opt']:,.1f} tỷ VND")

    with st.container(border=True):
        st.subheader("2. Quỹ đạo tối ưu K, D, AI, H, Y, C (2026-2035)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(fill='tozeroy', x=res['years'], y=res['K'], name='Vốn (K)', line=dict(color='#3182CE')))
        fig.add_trace(go.Scatter(fill='tozeroy', x=res['years'], y=res['Y'], name='Sản lượng (Y)', line=dict(color='#E53E3E')))
        fig.add_trace(go.Scatter(fill='tozeroy', x=res['years'], y=res['C'], name='Tiêu dùng (C)', line=dict(color='#38A169')))
        fig.add_trace(go.Scatter(fill='tozeroy', x=res['years'], y=res['D'], name='Số hóa (D)', yaxis='y2', line=dict(color='#D69E2E')))
        fig.add_trace(go.Scatter(fill='tozeroy', x=res['years'], y=res['H'], name='Nhân lực (H)', yaxis='y2', line=dict(color='#805AD5')))
        fig.add_trace(go.Scatter(fill='tozeroy', x=res['years'], y=res['AI'], name='AI', yaxis='y2', line=dict(color='#D53F8C', dash='dash')))
        fig.update_layout(title='Quỹ đạo tối ưu các biến số vĩ mô', yaxis2=dict(overlaying='y', side='right', title="Chỉ số phụ"))
        st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.subheader("3. Phân tích cú sốc 2028 (Giảm 8% Y)")
        col1, col2 = st.columns(2)
        col1.metric("Welfare (Không cú sốc)", f"{res['welfare_opt']:,.1f}")
        col2.metric("Welfare (Có cú sốc)", f"{res['welfare_shock']:,.1f}", delta=f"{res['welfare_shock'] - res['welfare_opt']:,.1f}")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(fill='tozeroy', x=res['years'], y=res['Y'], name='Y (Bình thường)', line=dict(color='#3182CE')))
        fig2.add_trace(go.Scatter(fill='tozeroy', x=res['years'], y=res['Y_shock'], name='Y (Cú sốc)', line=dict(color='#E53E3E', dash='dash')))
        fig2.update_layout(title='So sánh quỹ đạo Sản lượng (Y) khi có cú sốc 2028')
        st.plotly_chart(fig2, use_container_width=True)

    with st.container(border=True):
        st.subheader("4. So sánh chiến lược đầu tư: Trải đều vs Front-load")
        col1, col2 = st.columns(2)
        col1.metric("Welfare (Trải đều)", f"{res['welfare_even']:,.1f}")
        col2.metric("Welfare (Front-load)", f"{res['welfare_front']:,.1f}")
        st.success(f"Chiến lược tốt hơn: **{res['better_strategy']}**")

# ─── Bài 9 ───
elif page == pages[8]:
    st.title("Bài 9: Mô phỏng tác động AI lên lao động")
    with st.expander("⚙️ Bảng điều khiển thông số (Topbar)", expanded=True):
        c1, c2, c3 = st.columns(3)
        ai_rate = c1.slider("Tốc độ áp dụng AI", 0.10, 0.70, 0.30, 0.05)
        retrain = c2.slider("Ngân sách đào tạo (Nghìn tỷ)", 5, 50, 15, 5)
        speed = c3.slider("Tốc độ chuyển đổi", 0.1, 1.0, 0.5, 0.1)
        c1, c2, c3 = st.columns(3)
        new_job = c1.slider("Hệ số việc mới/AI", 0.1, 0.8, 0.4, 0.05)
    res = solve_bai09(ai_adoption_rate=ai_rate, retraining_budget=retrain, transition_speed=speed, new_job_multiplier=new_job)
    
    with st.container(border=True):
        st.subheader("1. Phân bổ tối ưu (PuLP)")
        st.metric("Tổng Việc làm ròng (NetJob)", f"{res['total_net']:,.2f} triệu")
        df_sectors = pd.DataFrame(res['sector_table']).T
        st.dataframe(df_sectors.style.format("{:.3f}"), use_container_width=True)

    with st.container(border=True):
        st.subheader("2. Ngưỡng đầu tư đào tạo ngành Chế biến chế tạo (Ngành 2)")
        st.info(f"**Ngưỡng $x_{{H, 2}}$ tối thiểu** để $NetJob_2 \\ge 0$ khi tối đa hóa AI ($x_{{AI, 2}}=1$): **{res['threshold_xH2']:.4f}**")

    with st.container(border=True):
        st.subheader("3. Luồng dịch chuyển lao động nhóm dễ tổn thương (Ngành 1, 3, 4)")
        fig_sankey = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15,
              thickness = 20,
              line = dict(color = "black", width = 0.5),
              label = res['sankey_nodes']
            ),
            link = dict(
              source = res['sankey_links']['source'],
              target = res['sankey_links']['target'],
              value = res['sankey_links']['value']
            ))])
        fig_sankey.update_layout(title_text="Biểu đồ Sankey: Dịch chuyển lao động phổ thông", font_size=12)
        st.plotly_chart(fig_sankey, use_container_width=True)

    with st.container(border=True):
        st.subheader("4. Ràng buộc mở rộng: Không ngành nào mất quá 5% lao động")
        if res['ext_feasible']:
            st.success("✅ **Khả thi:** Mô hình CÓ THỂ tìm được phân bổ thỏa mãn điều kiện không ngành nào mất >5% lao động.")
        else:
            st.error("❌ **Không khả thi:** Ngân sách đào tạo không đủ hoặc tốc độ chuyển đổi quá chậm để giữ mức mất việc <5% ở tất cả các ngành.")

# ─── Bài 10 ───
elif page == pages[9]:
    st.title("Bài 10: Quy hoạch ngẫu nhiên 2 giai đoạn")
    with st.expander("⚙️ Bảng điều khiển thông số (Topbar)", expanded=True):
        c1, c2, c3 = st.columns(3)
        p1 = c1.slider("P(Lạc quan)", 0.0, 1.0, 0.3, 0.05)
        p2 = c2.slider("P(Cơ sở)", 0.0, 1.0, 0.45, 0.05)
        p3 = c3.slider("P(Bi quan)", 0.0, 1.0, 0.2, 0.05)
        c1, c2, c3 = st.columns(3)
        budget = c1.slider("Ngân sách GĐ1 (Nghìn tỷ)", 30, 80, 65, 5)
    res = solve_bai10(p_optimistic=p1, p_baseline=p2, p_pessimistic=p3, first_stage_cap=budget)
    
    with st.container(border=True):
        st.subheader("1. So sánh Lợi nhuận kỳ vọng")
        col1, col2, col3 = st.columns(3)
        col1.metric("Giải pháp ngẫu nhiên (SP)", f"{res['sp_value']:,.1f}")
        col2.metric("Kỳ vọng giá trị hoàn hảo (EVPI)", f"{res['evpi']:,.1f}", help="Lợi ích thêm nếu biết trước tương lai")
        col3.metric("Giá trị của giải pháp ngẫu nhiên (VSS)", f"{res['vss']:,.1f}", help="Tổn thất nếu chỉ dùng trung bình (EEV)")

    with st.container(border=True):
        st.subheader("2. Quyết định phân bổ Giai đoạn 1 (First-stage)")
        df_alloc = pd.DataFrame(res['sp_alloc']).T
        st.dataframe(df_alloc.style.format("{:.2f}"), use_container_width=True)

    with st.container(border=True):
        st.subheader("3. Robust Optimization (Cực đại hóa kịch bản xấu nhất)")
        st.info(f"Giá trị lợi ích tồi tệ nhất được đảm bảo (Worst-case Z): **{res['rob_value']:,.1f}**")
        categories = ['I (Hạ tầng)', 'D (Số hóa)', 'AI', 'H (Nhân lực)']
    
        colA, colB = st.columns(2)
        with colA:
            fig_sp = px.pie(names=categories, values=res['x_sp'], title="Phân bổ SP (Tối đa hóa kỳ vọng)")
            st.plotly_chart(fig_sp, use_container_width=True)
        with colB:
            fig_rob = px.pie(names=categories, values=res['x_rob'], title="Phân bổ Robust (An toàn nhất)")
            st.plotly_chart(fig_rob, use_container_width=True)

# ─── Bài 11 ───
elif page == pages[10]:
    st.title("Bài 11: Học tăng cường (Q-learning & DQN)")
    with st.expander("⚙️ Bảng điều khiển thông số (Topbar)", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        alpha = c1.slider("Learning rate α", 0.01, 0.5, 0.1, 0.01)
        gamma = c2.slider("Discount γ", 0.5, 0.99, 0.95, 0.01)
        episodes = c3.slider("Số episodes", 1000, 20000, 10000, 1000)
        use_dqn = c4.checkbox("Huấn luyện DQN (stable-baselines3)", value=True)
        
    with st.spinner("Đang huấn luyện Agent (Vui lòng đợi vài giây)..."):
        res = solve_bai11(learning_rate=alpha, discount_factor=gamma, episodes=episodes, use_dqn=use_dqn)
        
    with st.container(border=True):
        st.subheader("1. Đánh giá Learning Curve")
        fig = go.Figure()
        x_q = list(range(0, res['episodes'], max(1, res['episodes']//100)))
        fig.add_trace(go.Scatter(fill='tozeroy', x=x_q, y=res['q_smoothed'], mode='lines', name='Q-Learning (Tabular)', line=dict(color='#3182CE')))
        if len(res['dqn_smoothed']) > 0:
            x_dqn = list(range(0, 20000, max(1, 20000//100)))
            # Adjust x_dqn length if needed
            x_dqn = x_dqn[:len(res['dqn_smoothed'])]
            fig.add_trace(go.Scatter(fill='tozeroy', x=x_dqn, y=res['dqn_smoothed'], mode='lines', name='DQN (Neural Net)', line=dict(color='#E53E3E')))
        fig.update_layout(title="Tổng phần thưởng trung bình theo Episodes", xaxis_title="Episodes", yaxis_title="Reward")
        st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.subheader("2. Chính sách tối ưu $\\\\pi^*(s)$ tại các trạng thái khởi đầu")
        df_policies = pd.DataFrame(list(res['extracted_policies'].items()), columns=["Trạng thái giả định", "Hành động (Policy) được chọn"])
        st.dataframe(df_policies, use_container_width=True)

    with st.container(border=True):
        st.subheader("3. So sánh phần thưởng tích lũy trung bình")
        df_rules = pd.DataFrame(list(res['rules_perf'].items()), columns=["Chính sách", "Phần thưởng trung bình"])
        fig_bar = px.bar(df_rules, x="Chính sách", y="Phần thưởng trung bình", color="Chính sách", title="Hiệu suất so với Rule-based Policies")
        st.plotly_chart(fig_bar, use_container_width=True)

# ─── Bài 12 ───
elif page == pages[11]:
    st.title("Bài 12: Đồ án tổng hợp AIDEOM-VN")
    with st.expander("⚙️ Bảng điều khiển thông số (Topbar)", expanded=True):
        c1, c2, c3 = st.columns(3)
        scenario = c1.selectbox("Kịch bản:", ['S1','S2','S3','S4','S5'], index=4,
            format_func=lambda s: {'S1':'S1. Truyền thống','S2':'S2. Số hóa nhanh','S3':'S3. AI dẫn dắt','S4':'S4. Bao trùm số','S5':'S5. Tối ưu cân bằng'}[s])
        c1, c2, c3 = st.columns(3)
        budget = c1.slider("Ngân sách tổng (Tỷ VND)", 10000, 100000, 50000, 5000)
    res = solve_bai12_dashboard(DATA_DIR, budget, scenario)
    
    st.info(f"**Mô tả kịch bản:** {res['description']}")
    
    tab1, tab2, tab3, tab4 = st.tabs([" Tổng quan & Phân bổ", " Kịch bản so sánh", " Tăng trưởng GDP & Việc làm", " Rủi ro & Vùng miền"])
    
    with tab1:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Hạ tầng (I)", f"{res['allocation']['I']:,.0f}")
            c2.metric("Số hóa (D)", f"{res['allocation']['D']:,.0f}")
            c3.metric("AI", f"{res['allocation']['AI']:,.0f}")
            c4.metric("Nhân lực (H)", f"{res['allocation']['H']:,.0f}")
        
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=res['radar']['values'],
                theta=res['radar']['dimensions'],
                fill='toself', line_color=LINE_COLOR
            ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title="Đánh giá tổng hợp đa chiều", height=400)
            st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab2:
        with st.container(border=True):
            st.subheader("So sánh 3 kịch bản chính (Năm 2030)")
            # Calculate comparison for S1, S3, S5
            s1 = solve_bai12_dashboard(DATA_DIR, budget, 'S1')
            s3 = solve_bai12_dashboard(DATA_DIR, budget, 'S3')
            s5 = solve_bai12_dashboard(DATA_DIR, budget, 'S5')
        
            comp_data = {
                "Chỉ số (2030)": ["GDP (Nghìn tỷ VND)", "Tỷ trọng AI (%)", "Điểm Rủi ro", "Mất việc ròng (Ngàn người)"],
                "S1 (Truyền thống)": [
                    f"{s1['gdp_forecast']['gdp'][-1]:,.2f}",
                    f"{s1['risk']['ai_budget_share']}%",
                    f"{s1['risk']['risk_score']} ({s1['risk']['level']})",
                    f"{s1['labor_impact']['net_total']:,.1f}"
                ],
                "S3 (AI dẫn dắt)": [
                    f"{s3['gdp_forecast']['gdp'][-1]:,.2f}",
                    f"{s3['risk']['ai_budget_share']}%",
                    f"{s3['risk']['risk_score']} ({s3['risk']['level']})",
                    f"{s3['labor_impact']['net_total']:,.1f}"
                ],
                "S5 (Tối ưu cân bằng)": [
                    f"{s5['gdp_forecast']['gdp'][-1]:,.2f}",
                    f"{s5['risk']['ai_budget_share']}%",
                    f"{s5['risk']['risk_score']} ({s5['risk']['level']})",
                    f"{s5['labor_impact']['net_total']:,.1f}"
                ]
            }
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True)
            st.info("💡 **Phân tích:** Kịch bản S3 (AI dẫn dắt) cho GDP cao nhất nhưng rủi ro việc làm và an toàn hệ thống lớn. S5 (Cân bằng) hy sinh một phần GDP để duy trì an sinh xã hội và giảm rủi ro.")

    with tab3:
        with st.container(border=True):
            st.subheader("Dự báo GDP & Tác động Việc làm")
            colA, colB = st.columns(2)
            fig = px.line(x=res['gdp_forecast']['years'], y=res['gdp_forecast']['gdp'], title="Dự báo GDP (Nghìn tỷ VND)", markers=True)
            fig.update_traces(line_color=LINE_COLOR)
            colA.plotly_chart(fig, use_container_width=True)
        
            fig2 = px.bar(x=res['labor_impact']['sectors'], y=res['labor_impact']['net_jobs'], title="Tác động việc làm ròng",
                color=res['labor_impact']['net_jobs'], color_continuous_scale='Blues')
            colB.plotly_chart(fig2, use_container_width=True)
        
        with st.container(border=True):
            st.subheader("Xếp hạng Ưu tiên Ngành")
            df_prio = pd.DataFrame(res['priority'])
            fig3 = px.bar(df_prio, x='sector', y='score', title="Điểm ưu tiên theo ngành (Đa tiêu chí)", color='score', color_continuous_scale='Blues')
            st.plotly_chart(fig3, use_container_width=True)
    
    with tab4:
        with st.container(border=True):
            st.subheader("Rủi ro & Xếp hạng Vùng (TOPSIS)")
            col1, col2 = st.columns([1, 2])
            col1.metric("Điểm rủi ro", f"{res['risk']['risk_score']}", delta=res['risk']['level'], delta_color="inverse")
            col1.metric("Tỷ trọng vốn AI", f"{res['risk']['ai_budget_share']}%")
        
            df_topsis = pd.DataFrame(res['topsis'])
            fig4 = px.bar(df_topsis, x='region', y='score', title="Điểm TOPSIS các vùng", text='rank', color='score', color_continuous_scale='Blues')
            col2.plotly_chart(fig4, use_container_width=True)
