import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- THIẾT LẬP MÔI TRƯỜNG MÊ CUNG CHUẨN XÁC ---
maze_base = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [1, 0, 1, 0, 2]
])

start = (0, 0)
goal = (4, 4)
episodes = 200
max_steps = 100


def state_to_index(x, y):
    return x * 5 + y


def take_action(state, action, current_maze, trap_cell=None):
    x, y = state
    if action == 0:  # Lên
        next_x, next_y = x - 1, y
    elif action == 1:  # Xuống
        next_x, next_y = x + 1, y
    elif action == 2:  # Trái
        next_x, next_y = x, y - 1
    else:  # Phải
        next_x, next_y = x, y + 1

    if (next_x < 0 or next_x >= 5 or next_y < 0 or next_y >= 5 or current_maze[next_x, next_y] == 1):
        return state, -5

    next_state = (next_x, next_y)
    if trap_cell and next_state == trap_cell:
        return next_state, -30
    if next_state == goal:
        return next_state, 10
    else:
        return next_state, -1


def train_q_learning(alpha, gamma, epsilon, current_maze, trap_cell=None):
    np.random.seed(42)
    q_table = np.zeros((25, 4))
    rewards_history = []

    for episode in range(episodes):
        state = start
        total_reward = 0
        for step in range(max_steps):
            s_idx = state_to_index(state[0], state[1])
            if np.random.rand() < epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(q_table[s_idx])

            next_state, reward = take_action(state, action, current_maze, trap_cell)
            total_reward += reward

            next_s_idx = state_to_index(next_state[0], next_state[1])
            best_next_action = np.argmax(q_table[next_s_idx])
            q_table[s_idx, action] += alpha * (
                        reward + gamma * q_table[next_s_idx, best_next_action] - q_table[s_idx, action])

            state = next_state
            if state == goal:
                break
        rewards_history.append(total_reward)

    state = start
    path = [state]
    loop_check = set([state])
    for _ in range(max_steps):
        if state == goal:
            break
        action = np.argmax(q_table[state_to_index(state[0], state[1])])
        state, _ = take_action(state, action, current_maze, trap_cell)
        if state in loop_check and state != goal:
            path.append(state)
            break
        path.append(state)
        loop_check.add(state)

    return rewards_history, path


# --- CHẠY HUẤN LUYỆN 4 TRƯỜNG HỢP ---
configs = {
    "a": {"alpha": 0.1, "gamma": 0.9, "epsilon": 0.1, "color": "tab:blue"},
    "b": {"alpha": 0.05, "gamma": 0.8, "epsilon": 0.1, "color": "tab:orange"},
    "c": {"alpha": 0.1, "gamma": 0.7, "epsilon": 0.15, "color": "tab:green"},
    "d": {"alpha": 0.15, "gamma": 0.9, "epsilon": 0.15, "color": "tab:red"}
}

results = {}
for key, cfg in configs.items():
    history, path = train_q_learning(cfg["alpha"], cfg["gamma"], cfg["epsilon"], maze_base)
    results[key] = {"history": history, "path": path}

try:
    current_file_path = os.path.abspath(__file__)
except NameError:
    current_file_path = r"C:\Users\DELL\OneDrive\Documents\python\RL.py"
print(f"{sys.executable} {current_file_path}")
for key in ["a", "b", "c", "d"]:
    print(f"Trường hợp {key.upper()} - Đường đi tối ưu: {results[key]['path']}")

# --- FIGURE 1 ---
fig1, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
fig1.suptitle("Đồ thị tổng phần thưởng qua từng Episode của từng trường hợp", fontsize=12, fontweight='bold')
subplot_mapping = {"a": (0, 0), "b": (0, 1), "c": (1, 0), "d": (1, 1)}
for key, data in results.items():
    cfg = configs[key]
    row, col = subplot_mapping[key]
    ax = axs[row, col]
    ax.plot(data["history"], color=cfg["color"], linewidth=1.2)
    ax.set_title(f"Trường hợp {key.upper()} (α={cfg['alpha']}, γ={cfg['gamma']}, ε={cfg['epsilon']})",
                 fontweight='bold', fontsize=10, color=cfg["color"])
    ax.set_xlabel("Episodes", fontsize=8)
    ax.set_ylabel("Total Reward", fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout(pad=2.5)
fig1.subplots_adjust(top=0.88, hspace=0.4, wspace=0.3)

# --- FIGURE 2 ---
fig2, axs2 = plt.subplots(2, 2, figsize=(11, 11))
fig2.suptitle("Sơ đồ lộ trình đường đi tối ưu trong mê cung của từng trường hợp", fontsize=13, fontweight='bold',
              color='darkblue')
for key, data in results.items():
    row, col = subplot_mapping[key]
    ax2 = axs2[row, col]
    ax2.set_xlim(-0.5, 4.5)
    ax2.set_ylim(-0.5, 4.5)
    ax2.set_xticks(np.arange(5))
    ax2.set_yticks(np.arange(5))
    ax2.grid(True, color='black', linestyle='-', alpha=0.15)
    for i in range(5):
        for j in range(5):
            if maze_base[i, j] == 1:
                ax2.add_patch(
                    patches.Rectangle((j - 0.5, 4 - i - 0.5), 1, 1, facecolor="gray", edgecolor='black', alpha=0.8))
            elif maze_base[i, j] == 2:
                ax2.add_patch(patches.Rectangle((j - 0.5, 4 - i - 0.5), 1, 1, facecolor="yellow", edgecolor='black'))
                ax2.text(j, 4 - i, 'GOAL', ha='center', va='center', fontweight='bold', color='black', fontsize=8)
            elif (i, j) == start:
                ax2.add_patch(patches.Rectangle((j - 0.5, 4 - i - 0.5), 1, 1, facecolor="green", edgecolor='black'))
                ax2.text(j, 4 - i, 'START', ha='center', va='center', fontweight='bold', color='white', fontsize=8)
    current_path = data["path"]
    for (r, c) in current_path:
        if (r, c) != start and (r, c) != goal:
            ax2.add_patch(
                patches.Rectangle((c - 0.5, 4 - r - 0.5), 1, 1, facecolor="skyblue", alpha=0.5, edgecolor='blue',
                                  linestyle='--'))
    for idx in range(len(current_path) - 1):
        curr_s = current_path[idx]
        next_s = current_path[idx + 1]
        x_curr, y_curr = curr_s[1], 4 - curr_s[0]
        x_next, y_next = next_s[1], 4 - next_s[0]
        dx = (x_next - x_curr) * 0.35
        dy = (y_next - y_curr) * 0.35
        ax2.arrow(x_curr, y_curr, dx, dy, head_width=0.12, head_length=0.12, fc='darkblue', ec='darkblue', zorder=5)
    cfg = configs[key]
    ax2.set_title(f"Đường đi TH {key.upper()} (α={cfg['alpha']}, γ={cfg['gamma']}, ε={cfg['epsilon']})",
                  fontsize=10, fontweight='bold', color=cfg["color"])
plt.tight_layout(pad=3.0)
fig2.subplots_adjust(top=0.90, hspace=0.3, wspace=0.3)
plt.show()

# =====================================================================
# --- CÂU F: TẠO Ô BẪY TRÊN ĐƯỜNG ĐI VÀ HUẤN LUYỆN LẠI ---
# =====================================================================

# 1. Định nghĩa ô bẫy tại (2, 1) và tạo ma trận mê cung mới có bẫy
trap_cell = (2, 1)
maze_with_trap = maze_base.copy()
maze_with_trap[trap_cell[0], trap_cell[1]] = 3  # Số 3 đại diện cho ô bẫy


# Định nghĩa lại hàm dịch chuyển riêng cho câu f để áp dụng mức phạt -30
def take_action_f(state, action, current_maze):
    x, y = state
    if action == 0:  # Lên
        next_x, next_y = x - 1, y
    elif action == 1:  # Xuống
        next_x, next_y = x + 1, y
    elif action == 2:  # Trái
        next_x, next_y = x, y - 1
    else:  # Phải
        next_x, next_y = x, y + 1

    # Kiểm tra va chạm tường hoặc biên
    if (next_x < 0 or next_x >= 5 or next_y < 0 or next_y >= 5 or current_maze[next_x, next_y] == 1):
        return state, -5

    next_state = (next_x, next_y)

    # Kiểm tra nếu đạp trúng bẫy
    if next_state == trap_cell:
        return next_state, -30
    elif next_state == goal:
        return next_state, 10
    else:
        return next_state, -1


# 2. Huấn luyện lại từ đầu với cấu hình chuẩn (Trường hợp A: α=0.1, γ=0.9, ε=0.1)
np.random.seed(42)
q_table_f = np.zeros((25, 4))

for episode in range(episodes):
    state = start
    for step in range(max_steps):
        s_idx = state_to_index(state[0], state[1])

        if np.random.rand() < 0.1:  # epsilon = 0.1
            action = np.random.randint(4)
        else:
            action = np.argmax(q_table_f[s_idx])

        next_state, reward = take_action_f(state, action, maze_with_trap)

        next_s_idx = state_to_index(next_state[0], next_state[1])
        best_next_action = np.argmax(q_table_f[next_s_idx])

        # Cập nhật bảng Q
        q_table_f[s_idx, action] += 0.1 * (
                    reward + 0.9 * q_table_f[next_s_idx, best_next_action] - q_table_f[s_idx, action])

        state = next_state
        if state == goal:
            break

# 3. Trích xuất lộ trình tối ưu sau khi né bẫy
state = start
path_f = [state]
loop_check = set([state])
for _ in range(max_steps):
    if state == goal:
        break
    action = np.argmax(q_table_f[state_to_index(state[0], state[1])])
    state, _ = take_action_f(state, action, maze_with_trap)
    if state in loop_check and state != goal:
        path_f.append(state)
        break
    path_f.append(state)
    loop_check.add(state)

# 4. In thông tin mê cung và đường đi mới ra Terminal
print("\n" + "=" * 20 + " KẾT QUẢ CÂU F " + "=" * 20)
print(f"Vị trí ô bẫy nhận phạt -30: {trap_cell}")
print("\nMê cung đã tạo bẫy (S: Start, G: Goal, W: Wall, T: Trap, . : Empty):")
for i in range(5):
    row_str = ""
    for j in range(5):
        if (i, j) == start:
            row_str += "S "
        elif (i, j) == goal:
            row_str += "G "
        elif maze_with_trap[i, j] == 1:
            row_str += "W "
        elif maze_with_trap[i, j] == 3:
            row_str += "T "
        else:
            row_str += ". "
    print(row_str)

print(f"\nĐường đi tối ưu sau khi huấn luyện lại (Né bẫy):")
print(path_f)

# 5. --- FIGURE 3: TRỰC QUAN HÓA MÊ CUNG NÉ BẪY CỦA CÂU F ---
fig3, ax3 = plt.subplots(figsize=(6, 6))
ax3.set_xlim(-0.5, 4.5)
ax3.set_ylim(-0.5, 4.5)
ax3.set_xticks(np.arange(5))
ax3.set_yticks(np.arange(5))
ax3.grid(True, color='black', linestyle='-', alpha=0.15)

for i in range(5):
    for j in range(5):
        if maze_with_trap[i, j] == 1:  # Tường
            ax3.add_patch(patches.Rectangle((j - 0.5, 4 - i - 0.5), 1, 1, facecolor="gray", edgecolor='black'))
        elif maze_with_trap[i, j] == 2:  # Đích
            ax3.add_patch(patches.Rectangle((j - 0.5, 4 - i - 0.5), 1, 1, facecolor="yellow", edgecolor='black'))
            ax3.text(j, 4 - i, 'GOAL', ha='center', va='center', fontweight='bold', color='black')
        elif maze_with_trap[i, j] == 3:  # Bẫy
            ax3.add_patch(patches.Rectangle((j - 0.5, 4 - i - 0.5), 1, 1, facecolor="crimson", edgecolor='black'))
            ax3.text(j, 4 - i, 'TRAP\n(-30)', ha='center', va='center', fontweight='bold', color='white', fontsize=8)
        elif (i, j) == start:  # Xuất phát
            ax3.add_patch(patches.Rectangle((j - 0.5, 4 - i - 0.5), 1, 1, facecolor="green", edgecolor='black'))
            ax3.text(j, 4 - i, 'START', ha='center', va='center', fontweight='bold', color='white')

# Vẽ lộ trình mới né bẫy bằng màu xanh lam nhạt
for (r, c) in path_f:
    if (r, c) != start and (r, c) != goal:
        ax3.add_patch(patches.Rectangle((c - 0.5, 4 - r - 0.5), 1, 1, facecolor="skyblue", alpha=0.5, edgecolor='blue',
                                        linestyle='--'))

# Vẽ mũi tên định hướng hành động
for idx in range(len(path_f) - 1):
    curr_s = path_f[idx]
    next_s = path_f[idx + 1]
    x_curr, y_curr = curr_s[1], 4 - curr_s[0]
    x_next, y_next = next_s[1], 4 - next_s[0]
    dx = (x_next - x_curr) * 0.35
    dy = (y_next - y_curr) * 0.35
    ax3.arrow(x_curr, y_curr, dx, dy, head_width=0.12, head_length=0.12, fc='darkblue', ec='darkblue', zorder=5)

ax3.set_title("Sơ đồ lộ trình tối ưu sau khi tạo ô bẫy (Câu f)", fontsize=11, fontweight='bold', color='crimson')
plt.show()