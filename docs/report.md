# Báo cáo Đồ án Tổng hợp AIDEOM-VN

## 1. Giới thiệu
AIDEOM-VN (Artificial Intelligence and Digital Economy Optimization Model for Vietnam) là một hệ thống tối ưu hóa đa mục tiêu và mô phỏng chính sách nhằm hỗ trợ việc ra quyết định đầu tư công vào Trí tuệ nhân tạo (AI) và Chuyển đổi số (Digital Economy). Đồ án tổng hợp này là sự kết tinh của các bài toán tối ưu (Tuyến tính, Đa mục tiêu, Ngẫu nhiên) và Học tăng cường (Reinforcement Learning) vào một Dashboard thống nhất.

## 2. Kiến trúc Hệ thống (Bài 12)
Hệ thống được thiết kế theo kiến trúc hướng mô-đun (Modular Architecture) bao gồm:
- **Module 1 (Macro):** Mô phỏng kinh tế vĩ mô thông qua hàm sản xuất Cobb-Douglas, dự báo tăng trưởng GDP từ 2026-2030 dựa trên các biến số vốn (K), lao động (L), số hóa (D), AI và nhân lực (H).
- **Module 2 (Allocation):** Tối ưu hóa phân bổ ngân sách bằng quy hoạch tuyến tính (Linear Programming - SciPy/PuLP) đảm bảo các ràng buộc đầu tư tối thiểu và tối đa.
- **Module 3 (Priority):** Đánh giá đa tiêu chí (TOPSIS/Entropy) để xếp hạng mức độ ưu tiên của 21 ngành kinh tế.
- **Module 4 (Labor Impact):** Mô phỏng dịch chuyển lao động và nguy cơ tự động hóa. Đưa ra chỉ số mất việc ròng (Net Job Loss).
- **Module 5 (Region Topsis):** Xếp hạng 6 vùng kinh tế - xã hội dựa trên các chỉ số lợi ích và chi phí.
- **Module 6 (Risk Assessment):** Đánh giá rủi ro hệ thống thông qua tỷ trọng phân bổ vốn AI và dự báo tăng trưởng GDP.

## 3. Ứng dụng Học Tăng Cường (Bài 11)
Hệ thống sử dụng Học tăng cường để tự động hóa chiến lược đầu tư:
- **Môi trường (Environment):** Xây dựng môi trường `VietnamEconomyEnv` bằng Gymnasium. Môi trường mô phỏng chu kỳ 10 năm với state bao gồm 4 biến trạng thái (Tăng trưởng GDP, Chuyển đổi số, Năng lực AI, Rủi ro thất nghiệp) và action là 5 kịch bản phân bổ vốn.
- **Thuật toán Q-Learning:** Sử dụng epsilon-greedy với cơ chế decay epsilon từ 1.0 xuống 0.05 qua 10,000 episodes. Kết quả thu được hàm giá trị tối ưu (Optimal Q-Table).
- **Thuật toán DQN:** Triển khai Deep Q-Network bằng thư viện `stable-baselines3`, sử dụng mạng Neural Network với 2 hidden layers (64 units) để xử lý hàm xấp xỉ giá trị (Value Function Approximation). DQN hội tụ nhanh và đưa ra phần thưởng trung bình cao hơn các chiến lược rule-based tĩnh.

## 4. Giao diện Dashboard (Streamlit)
Toàn bộ hệ thống được tích hợp vào một ứng dụng Streamlit duy nhất, hỗ trợ 4 Tabs tương tác cho Đồ án (Bài 12):
1. **Tổng quan & Phân bổ:** Biểu đồ Radar đánh giá đa chiều.
2. **Kịch bản so sánh:** So sánh trực quan giữa 3 kịch bản cốt lõi (S1 - Truyền thống, S3 - AI dẫn dắt, S5 - Cân bằng).
3. **GDP & Việc làm:** Dự báo biểu đồ tăng trưởng và tác động mất việc theo ngành.
4. **Rủi ro & Vùng miền:** Chỉ số rủi ro và đánh giá phân bổ vùng.

## 5. Kết luận
Kịch bản S5 (Tối ưu cân bằng) được đánh giá là lựa chọn ưu việt nhất khi tối đa hóa được GDP trong khi kiểm soát được rủi ro mất việc (<5% ở đa số ngành). Đồ án đã minh chứng khả năng áp dụng linh hoạt của các kỹ thuật Vận trù học và Trí tuệ nhân tạo vào quy hoạch chính sách vĩ mô.
