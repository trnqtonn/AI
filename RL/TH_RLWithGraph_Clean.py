import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
plt.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\Downloads\\ffmpeg-7.1.1-full_build\\bin\\ffmpeg.exe'
import os

# Thiết lập môi trường
maze = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 1, 0, 2]
])  # 0: trống, 1: tường, 2: mục tiêu

# Tham số
alpha = 0.1  # Tốc độ học
gamma = 0.9  # Hệ số chiết khấu
epsilon = 0.1  # Xác suất khám phá
episodes = 200
max_steps = 100

start = (0, 0)
goal = (4, 4)

# Khởi tạo Q-table
q_table = np.zeros((25, 4))  # 25 trạng thái, 4 hành động (Lên, Xuống, Trái, Phải)

# Chuyển tọa độ thành chỉ số
def state_to_index(x, y):
    return x * 5 + y

# Kiểm tra và thực hiện hành động
def take_action(state, action):
    x, y = state
    if action == 0:  # Lên
        next_x, next_y = x - 1, y
    elif action == 1:  # Xuống
        next_x, next_y = x + 1, y
    elif action == 2:  # Trái
        next_x, next_y = x, y - 1
    else:  # Phải
        next_x, next_y = x, y + 1
    
    # Kiểm tra ranh giới và tường
    if (next_x < 0 or next_x >= 5 or next_y < 0 or next_y >= 5 or maze[next_x, next_y] == 1):
        return state, -5  # Va vào tường
    next_state = (next_x, next_y)
    if next_state == goal:
        reward = 10
    else:
        reward = -1
    return next_state, reward

# Huấn luyện
total_rewards = []  # Lưu tổng phần thưởng mỗi episode để theo dõi quá trình học
for episode in range(episodes):
    state = start
    total_reward = 0
    for step in range(max_steps):
        # Chọn hành động với epsilon-greedy
        if np.random.rand() < epsilon:
            action = np.random.randint(4)  # Khám phá
        else:
            action = np.argmax(q_table[state_to_index(state[0], state[1])])  # Khai thác
        
        # Thực hiện hành động, tích luỹ phần thưởng để vẽ hình và biểu đồ
        
        
        # Cập nhật Q-table
        
        # //q_value = q_table ... 
        
        
        state = next_state
        if state == goal:
            break
    total_rewards.append(total_reward)

# Tìm đường đi tối ưu
state = start
path = [state]
while state != goal:
    action = np.argmax(q_table[state_to_index(state[0], state[1])])
    state, _ = take_action(state, action)
    path.append(state)

print("Đường đi tối ưu:", path)

# Trực quan hóa quá trình học (biểu đồ phần thưởng)


# Trực quan hóa đường đi tối ưu
def plot_maze(ax):
    ax.clear()
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.grid(True)
    
    # Vẽ mê cung
    for i in range(5):
        for j in range(5):
            if maze[i, j] == 1:  # Tường
                ax.add_patch(patches.Rectangle((j-0.5, 4-i-0.5), 1, 1, facecolor="gray"))
            elif maze[i, j] == 2:  # Mục tiêu
                ax.add_patch(patches.Rectangle((j-0.5, 4-i-0.5), 1, 1, facecolor="yellow"))
            elif (i, j) == start:  # Điểm bắt đầu
                ax.add_patch(patches.Rectangle((j-0.5, 4-i-0.5), 1, 1, facecolor="green"))

# Tạo animation cho đường đi
fig, ax = plt.subplots(figsize=(6, 6))
plot_maze(ax)

def update(frame):
    plot_maze(ax)
    if frame < len(path):
        x, y = path[frame]
        ax.plot(y, 4-x, 'ro', markersize=10)  # Vẽ vị trí hiện tại của tác nhân
    ax.set_title(f"Đường đi tối ưu - Bước {frame+1}")

ani = FuncAnimation(fig, update, frames=len(path), interval=500, repeat=False)
plt.show()
#plt.savefig(os.path.expanduser("~\\Desktop\\learning_process.png"))  # Lưu vào Desktop
#Lưu animation thành video
#writer = FFMpegWriter(fps=30, bitrate=2000)  # fps: tốc độ khung hình, bitrate: chất lượng
#ani.save("c:\\ThucHanh\\optimal_path.mp4", writer=writer)

print("Đã lưu video thành 'optimal_pathNew.mp4'")
# Lưu animation thành video
output_path = 'c:\\ThucHanh\\optimal_pathNew.mp4' 
writer = FFMpegWriter(fps=2, bitrate=1800)
ani.save(output_path, writer=writer)

print(f"Đã lưu video thành '{output_path}'")

plt.close()

# Trực quan hóa Q-table trên mê cung
def draw_q_values(ax, q_table):
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.grid(True)

    # Vẽ mê cung
    for i in range(5):
        for j in range(5):
            if maze[i, j] == 1:  # Tường
                ax.add_patch(patches.Rectangle((j - 0.5, 4 - i - 0.5), 1, 1, color='gray'))
            elif maze[i, j] == 2:  # Mục tiêu
                ax.add_patch(patches.Rectangle((j - 0.5, 4 - i - 0.5), 1, 1, color='yellow'))
            elif (i, j) == start:
                ax.add_patch(patches.Rectangle((j - 0.5, 4 - i - 0.5), 1, 1, color='green'))

            if maze[i, j] != 1:  # Không vẽ Q tại ô là tường
                s = state_to_index(i, j)
                q = q_table[s]
                x, y = j, 4 - i

                # Vẽ mũi tên và giá trị Q
                if q[0] != 0:  # Lên
                    ax.arrow(x, y, 0, 0.2, head_width=0.1, color='red')
                    ax.text(x, y + 0.3, f'{q[0]:.1f}', ha='center', fontsize=8)
                if q[1] != 0:  # Xuống
                    ax.arrow(x, y, 0, -0.2, head_width=0.1, color='blue')
                    ax.text(x, y - 0.4, f'{q[1]:.1f}', ha='center', fontsize=8)
                if q[2] != 0:  # Trái
                    ax.arrow(x, y, -0.2, 0, head_width=0.1, color='green')
                    ax.text(x - 0.4, y, f'{q[2]:.1f}', va='center', fontsize=8)
                if q[3] != 0:  # Phải
                    ax.arrow(x, y, 0.2, 0, head_width=0.1, color='orange')
                    ax.text(x + 0.4, y, f'{q[3]:.1f}', va='center', fontsize=8)

# Vẽ Q-table
fig, ax = plt.subplots(figsize=(8, 8))
draw_q_values(ax, q_table)
ax.set_title("Giá trị Q-table theo từng hành động trên mê cung")
plt.show()