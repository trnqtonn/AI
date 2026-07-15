import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# =====================================================================
# --- PHẦN 1: THIẾT LẬP MÔI TRƯỜNG MÊ CUNG NÂNG CAO 8x8 ---
# =====================================================================
# 0: Trống, 1: Tường W, 2: Đích G, 4: Chìa khóa K, 5: Portal P
maze_advanced = np.array([
    [0, 0, 0, 0, 0, 1, 0, 4],
    [0, 1, 1, 1, 5, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 1, 0],
    [0, 5, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 0, 1, 2]
])
start = (0, 0)
goal = (7, 7)
key_pos = (0, 7)
portal_1 = (1, 4)
portal_2 = (6, 1)
episodes = 1000
max_steps = 150

def state_to_index_adv(x, y, has_key):
    base_idx = x * 8 + y
    return base_idx * 2 + (1 if has_key else 0)

def take_action_adv(state, action, monster_pos, has_key):
    x, y = state
    if action == 0:  # Lên
        next_x, next_y = x - 1, y
    elif action == 1:  # Xuống
        next_x, next_y = x + 1, y
    elif action == 2:  # Trái
        next_x, next_y = x, y - 1
    else:  # Phải
        next_x, next_y = x, y + 1

    # 1. Kiểm tra va chạm biên hoặc va tường W
    if (next_x < 0 or next_x >= 8 or next_y < 0 or next_y >= 8 or maze_advanced[next_x, next_y] == 1):
        return state, -5, has_key, False

    next_state = (next_x, next_y)

    # 2. Kiểm tra va chạm quái vật di động
    if next_state == monster_pos:
        return next_state, -100, has_key, True  # Phạt cực nặng và kết thúc trò chơi

    # 3. Xử lý logic Cổng dịch chuyển (Portal)
    if next_state == portal_1:
        next_state = portal_2
    elif next_state == portal_2:
        next_state = portal_1

    # 4. Xử lý logic nhặt Chìa khóa
    new_has_key = has_key
    reward = -1  # Phạt di chuyển mặc định

    if next_state == key_pos and not has_key:
        new_has_key = True
        reward = 50  # Thưởng lớn khi tìm thấy chìa khóa để định hướng hành vi

    # 5. Kiểm tra điều kiện về đích
    if next_state == goal:
        if new_has_key:
            return next_state, 100, new_has_key, True  # Về đích thành công khi có khóa
        else:
            return state, -10, has_key, False  # Bị chặn lại nếu chưa có khóa và phạt -10

    return next_state, reward, new_has_key, False


# Hàm mô phỏng quái vật di chuyển tuần tra ngẫu nhiên xung quanh khu vực trung tâm
def move_monster(current_pos):
    mx, my = current_pos
    # Quái vật chỉ có thể di chuyển trong phạm vi trống hoặc đứng yên quanh sàn giữa
    possible_moves = [(mx, my), (mx + 1, my), (mx - 1, my), (mx, my + 1), (mx, my - 1)]
    valid_moves = []
    for nx, ny in possible_moves:
        if 2 <= nx <= 5 and 2 <= ny <= 5 and maze_advanced[nx, ny] == 0:
            valid_moves.append((nx, ny))
    if len(valid_moves) > 0:
        return valid_moves[np.random.randint(len(valid_moves))]
    return current_pos


# =====================================================================
# --- PHẦN 2: HUẤN LUYỆN Q-LEARNING CHO MÔ TRƯỜNG NÂNG CAO ---
# =====================================================================
alpha = 0.1
gamma = 0.95
epsilon = 0.2
np.random.seed(42)

# Khởi tạo bảng Q cho 128 trạng thái và 4 hành động
q_table_adv = np.zeros((128, 4))
rewards_history = []

print("Đang tiến hành huấn luyện robot trên mê cung nâng cao 8x8...")
for episode in range(episodes):
    state = start
    has_key = False
    monster_pos = (2, 3)
    total_reward = 0

    for step in range(max_steps):
        s_idx = state_to_index_adv(state[0], state[1], has_key)

        # Chọn hành động bằng Epsilon-Greedy
        if np.random.rand() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(q_table_adv[s_idx])

        next_state, reward, next_has_key, done = take_action_adv(state, action, monster_pos, has_key)
        total_reward += reward

        # Di chuyển quái vật tuần tra sau mỗi bước đi của robot
        monster_pos = move_monster(monster_pos)

        # Cập nhật giá trị vào Q-Table
        next_s_idx = state_to_index_adv(next_state[0], next_state[1], next_has_key)
        best_next_action = np.argmax(q_table_adv[next_s_idx])
        q_table_adv[s_idx, action] += alpha * (
                    reward + gamma * q_table_adv[next_s_idx, best_next_action] - q_table_adv[s_idx, action])

        state = next_state
        has_key = next_has_key
        if done:
            break

    rewards_history.append(total_reward)

    # Cơ chế giảm dần epsilon để mô hình ổn định dần ở giai đoạn cuối
    if episode > 600:
        epsilon = max(0.02, epsilon * 0.99)

print("Huấn luyện hoàn tất!")

# =====================================================================
# --- PHẦN 3: KIỂM TRA LỘ TRÌNH TỐI ƯU ĐÃ HỌC ---
# =====================================================================
state = start
has_key = False
monster_pos = (2, 3)
path_adv = [(state, monster_pos, has_key)]

for _ in range(max_steps):
    if state == goal and has_key:
        break
    s_idx = state_to_index_adv(state[0], state[1], has_key)
    action = np.argmax(q_table_adv[s_idx])

    # Ở lượt test, quái vật tạm thời cố định hoặc di chuyển theo hạt giống cố định để trích xuất path mẫu
    next_state, _, next_has_key, done = take_action_adv(state, action, monster_pos, has_key)
    monster_pos = move_monster(monster_pos)

    state = next_state
    has_key = next_has_key
    path_adv.append((state, monster_pos, has_key))
    if done and state == goal:
        break

# In danh sách các bước đi ra Terminal
print("\n" + "=" * 15 + " LỘ TRÌNH TRÊN MÊ CUNG NÂNG CAO " + "=" * 15)
print("Đường đi chi tiết của Robot (Tọa độ Robot, Tọa độ Quái vật, Trạng thái Khóa):")
for step_idx, (r_pos, m_pos, k_status) in enumerate(path_adv):
    print(f"Bước {step_idx:02d}: Robot {r_pos} | Quái vật {m_pos} | Có chìa khóa: {k_status}")

# =====================================================================
# --- PHẦN 4: TRỰC QUAN HÓA SƠ ĐỒ SÁNG TẠO BẰNG ANIMATION ---
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle("Mô Phỏng Tính Năng Gợi Ý Phát Triển Thêm (Mê Cung Nâng Cao 8x8)", fontsize=14, fontweight='bold')

# Đồ thị 1: Tiến trình hội tụ phần thưởng
ax1.plot(rewards_history, color='tab:purple', alpha=0.6)
ax1.set_title("Đồ thị tổng phần thưởng qua các Episode", fontweight='bold')
ax1.set_xlabel("Episodes")
ax1.set_ylabel("Total Reward")
ax1.grid(True, linestyle='--')


# Đồ thị 2: Bản đồ mê cung động
def draw_base_maze(ax):
    ax.clear()
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 7.5)
    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.grid(True, color='black', alpha=0.15)

    for i in range(8):
        for j in range(8):
            if maze_advanced[i, j] == 1:  # Tường W
                ax.add_patch(patches.Rectangle((j - 0.5, 7 - i - 0.5), 1, 1, facecolor="gray", edgecolor='black'))
            elif maze_advanced[i, j] == 2:  # Đích G
                ax.add_patch(patches.Rectangle((j - 0.5, 7 - i - 0.5), 1, 1, facecolor="yellow", edgecolor='black'))
                ax.text(j, 7 - i, 'GOAL', ha='center', va='center', fontweight='bold', fontsize=9)
            elif maze_advanced[i, j] == 4:  # Chìa khóa K
                ax.add_patch(patches.Rectangle((j - 0.5, 7 - i - 0.5), 1, 1, facecolor="gold", edgecolor='black'))
                ax.text(j, 7 - i, '🔑\nKEY', ha='center', va='center', fontweight='bold', fontsize=8,
                        color='darkgoldenrod')
            elif maze_advanced[i, j] == 5:  # Cổng dịch chuyển P
                ax.add_patch(
                    patches.Rectangle((j - 0.5, 7 - i - 0.5), 1, 1, facecolor="mediumpurple", edgecolor='black',
                                      alpha=0.7))
                ax.text(j, 7 - i, '🌀\nPORTAL', ha='center', va='center', fontweight='bold', fontsize=8, color='white')
            elif (i, j) == start:  # Xuất phát S
                ax.add_patch(patches.Rectangle((j - 0.5, 7 - i - 0.5), 1, 1, facecolor="green", edgecolor='black'))
                ax.text(j, 7 - i, 'START', ha='center', va='center', fontweight='bold', fontsize=9, color='white')


draw_base_maze(ax2)


def update_animation(frame):
    if frame >= len(path_adv):
        frame = len(path_adv) - 1

    r_pos, m_pos, k_status = path_adv[frame]
    draw_base_maze(ax2)

    # Nếu robot nhặt được chìa khóa, vẽ dấu vết đường đi màu vàng nhạt, ngược lại vẽ màu xanh lam
    path_color = "lightgoldenrodyellow" if k_status else "lightcyan"
    for idx in range(frame + 1):
        past_r, _, _ = path_adv[idx]
        if past_r != start and past_r != goal:
            ax2.add_patch(
                patches.Rectangle((past_r[1] - 0.5, 7 - past_r[0] - 0.5), 1, 1, facecolor=path_color, alpha=0.4))

    # Vẽ vị trí hiện tại của Robot (Dấu chấm đỏ bự)
    ax2.plot(r_pos[1], 7 - r_pos[0], 'ro', markersize=14, label='Robot', edgecolors='black', zorder=6)
    ax2.text(r_pos[1], 7 - r_pos[0], '🤖', ha='center', va='center', fontsize=12, zorder=7)

    # Vẽ vị trí hiện tại của Quái vật tuần tra (Dấu x đen)
    ax2.plot(m_pos[1], 7 - m_pos[0], 'kx', markersize=14, markeredgewidth=3, label='Monster', zorder=6)
    ax2.text(m_pos[1], 7 - m_pos[0] + 0.3, '👾 M', ha='center', va='center', color='black', fontweight='bold',
             fontsize=9, zorder=7)

    ax2.set_title(f"Lộ trình mô phỏng - Bước {frame}\nTrạng thái chìa khóa: {'ĐÃ LẤY 🔑' if k_status else 'CHƯA CÓ ❌'}",
                  fontweight='bold', fontsize=11, color='darkblue')


ani = FuncAnimation(fig, update_animation, frames=len(path_adv) + 5, interval=600, repeat=False)
plt.tight_layout()
plt.show()