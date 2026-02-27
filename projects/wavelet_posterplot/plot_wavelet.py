import os
import mne
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
from ieeg.viz.parula import parula_map

# --- 1. 路径与全局配置 ---
HOME = os.path.expanduser("~")
Task_Tag = "LexicalDecRepDelay"
# 原始数据目录
save_dir = os.path.join(HOME, "Box", "CoganLab", "D_Data", Task_Tag, "Baishen_Figs", "LexicalDecRepDelay")
# 论文素材保存目录 (Fig1)
manuscript_save_dir = r"D:\lbs\Little_projects\Greg_LexDelay\materials\figs_elements\Fig1"

# 物理尺寸常量 (与之前的 Wave Plots 严格对齐)
# 宽度计算公式: Width = (Duration * unit_scale) + left_padding + right_padding
fig_scale = 0.5 # 全局缩放因子 (调整整体大小，保持宽高比不变)
unit_scale = 3.0*fig_scale        # 每 1 秒数据对应的物理长度 (inches/sec)
left_padding = 1.6*fig_scale      # 左侧固定留白 (inches)，用于对齐 Y 轴
right_padding = 0.4*fig_scale     # 右侧固定留白 (inches)
fig_height = 3*fig_scale        # TFR 图像固定高度
dpi = 300

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.linewidth'] = 2.5 

# --- 2. 核心绘图函数 ---
def plot_tfr_scaled(tfr_data, pick, x_limits, save_name, is_auditory=False):

    # 物理宽度算法
    x_duration = x_limits[1] - x_limits[0]
    fig_width = (x_duration * unit_scale) + left_padding + right_padding
    
    # 显式创建指定尺寸的 Figure，防止产生多余空图
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    
    # 手动控制边距，确保 Padding 物理距离恒定
    ax_left = left_padding / fig_width
    ax_right = 1.0 - (right_padding / fig_width)
    fig.subplots_adjust(left=ax_left, right=ax_right, bottom=0.2, top=0.9)
    ax = fig.add_subplot(111)

    # 绘图：show=False 防止弹出多余窗口
    tfr_data.plot(picks=pick, fmin=0, fmax=400, yscale='linear', vlim=(-2, 2), cmap=parula_map, axes=ax, colorbar=False, show=False)
    
    # 装饰线
    ax.axvline(x=0, linestyle='--', color='k', linewidth=2.5)
    if is_auditory:
        # 添加 Auditory 任务特有的刺激边界红线
        ax.axvline(x=0.65, color='red', linestyle='--', alpha=0.7, linewidth=3)
        ax.axvline(x=1.5, color='red', linestyle='--', alpha=0.7, linewidth=3)
        
    # --- 视觉标准化 (移除 Labels) ---
    ax.set_xlim(x_limits)
    ax.spines['bottom'].set_bounds(x_limits[0], x_limits[1])
    #ax.spines[['top', 'right']].set_visible(False)
    
    ax.set_xlabel('') 
    ax.set_ylabel('') 
    ax.set_title('')  
    
    # X 轴刻度：0.5s 步长，24号字体
    xticks = [0, 0.5, 1.0, 1.5]
    xticks = [t for t in xticks if x_limits[0] <= t <= x_limits[1]]
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    
    # Y 轴刻度逻辑
    if is_auditory:
        # ax.set_yticks([6, 19, 67, 300, 1065]) # Let MNE handle ticks for linear scale
        ax.tick_params(axis='y', labelsize=18)
    else:
        ax.set_yticks([])

    ax.tick_params(axis='x', labelsize=18)#, rotation=45)
    
    # 矢量化保存
    if not os.path.exists(manuscript_save_dir):
        os.makedirs(manuscript_save_dir)
        
    full_save_path = os.path.join(manuscript_save_dir, f"{save_name}.svg")
    fig.savefig(full_save_path, format='svg', bbox_inches=None)
    plt.close(fig) 

# --- 3. 自动化遍历执行 ---
electrode_configs = [
    ('D0096', 'LFPS14', 'SM_Del'),
    ('D0096', 'LFPS8',  'DelOnly'),
    ('D0102', 'RTAS5',  'Aud_Del'),
    ('D0102', 'RFPI9',  'Mtr_Del')
]

for fname in ('Auditory-tfr.h5', 'Go-tfr.h5', 'Resp-tfr.h5'):
    is_aud = (fname == 'Auditory-tfr.h5')
    x_lims = [-0.25, 1.6] if is_aud else [-0.25, 1.0]
    
    for sub, pick, tag in electrode_configs:
        file_path = os.path.join(save_dir, sub, 'wavelet', fname)
        if not os.path.exists(file_path): 
            continue
        
        # --- 核心修复逻辑 ---
        # 兼容处理：read_tfrs 可能返回列表或单个 AverageTFR 对象
        tfr_input = mne.time_frequency.read_tfrs(file_path)
        tfr = tfr_input[0] if isinstance(tfr_input, list) else tfr_input
        
        save_tag = f"{tag}_exmp_{fname.split('-')[0]}"
        plot_tfr_scaled(tfr, pick, x_lims, save_tag, is_auditory=is_aud)

print(f"所有 TFR 图像已更新并保存至: {manuscript_save_dir}")