import numpy as np
import matplotlib.pyplot as plt
import mne
from ieeg.timefreq import gamma # 確保您的環境中已安裝此包

# ==========================================
# 1. 參數設定與模擬數據生成
# ==========================================
n_trials = 100
n_channels = 50
fs = 1000.0  

# 時間設定：2s pre-onset, 1s post-onset (共 3s)
pre_onset_time = 2.0
post_onset_time = 1.0
n_samples = int((pre_onset_time + post_onset_time) * fs) # 共 3000 個 samples
onset_sample = int(pre_onset_time * fs)                  # 第 2000 個 sample 為 Onset

# 生成以 Onset 為 0 的時間向量 (單位: 秒)
time_vector = (np.arange(n_samples) - onset_sample) / fs

# 初始化背景噪音數據 (高斯白噪聲)
np.random.seed(42)
data = np.random.normal(0, 0.5, (n_trials, n_channels, n_samples))

# --- 模擬聽覺 High Gamma 反應 (語音長度 550ms) ---
speech_duration = 0.55 
gamma_burst = np.zeros(n_samples)

# 構建聽覺反應的 Envelope (包絡)
for i in range(onset_sample, n_samples):
    t_active = time_vector[i] # 刺激發生後的時間
    
    if t_active <= speech_duration:
        # 階段 1: 刺激期間 -> 快速升起 (tau=20ms) + 緩慢下降 (tau=300ms)
        env = (1 - np.exp(-t_active / 0.02)) * np.exp(-t_active / 0.3)
    else:
        # 階段 2: 刺激結束 -> 從結束點的能量開始快速衰落 (tau=20ms)
        env_at_offset = (1 - np.exp(-speech_duration / 0.02)) * np.exp(-speech_duration / 0.3)
        env = env_at_offset * np.exp(-(t_active - speech_duration) / 0.02)
        
    # 乘以 100 Hz 的載波 (Carrier Wave)
    gamma_burst[i] = env * np.sin(2 * np.pi * 100 * t_active)

# 正規化並放大信號強度 (峰值設為 5.0)
gamma_burst = (gamma_burst / np.max(np.abs(gamma_burst))) * 5.0

# 將模擬的聽覺刺激疊加到所有 trials 和 channels
data += gamma_burst

# 封裝成 MNE Epochs 格式
info = mne.create_info(ch_names=[f'CH{i}' for i in range(n_channels)], 
                       sfreq=fs, ch_types='eeg')
epochs_simulated = mne.EpochsArray(data, info, tmin=-pre_onset_time)

print(f"Simulated data shape: {data.shape}")

# ==========================================
# 2. 執行 Gamma 提取
# ==========================================
gamma_env = gamma.extract(data, fs=fs, passband=(70, 150), copy=True, n_jobs=1)

# ==========================================
# 3. 驗證 Pre-onset Leakage
# ==========================================
# 計算所有 trial 和 channel 的平均值
mean_raw = np.mean(data, axis=(0, 1))
mean_env = np.mean(gamma_env, axis=(0, 1))

# 定義 Baseline 區間 (-2.0s 到 -0.2s) 和 Leakage 檢測區間 (-0.1s 到 0s)
# 轉換為 sample 索引: 
# -2.0s = sample 0
# -0.2s = sample 1800
# -0.1s = sample 1900
#  0.0s = sample 2000
idx_baseline_end = int((pre_onset_time - 0.2) * fs)
idx_leakage_start = int((pre_onset_time - 0.1) * fs)

baseline_env = np.mean(mean_env[0:idx_baseline_end])
pre_onset_env = np.mean(mean_env[idx_leakage_start:onset_sample])

print(f"Baseline mean envelope (-2.0s to -0.2s): {baseline_env:.4f}")
print(f"Pre-onset mean envelope (-0.1s to 0.0s): {pre_onset_env:.4f}")

# 數學判定：如果在 onset 前 100ms 內，能量平均值超過基線 20%，則視為存在洩露
if pre_onset_env > baseline_env * 1.2:
    print("⚠️ 警告：檢測到顯著的 Pre-onset Leakage！")
else:
    print("✅ 未檢測到顯著的 Pre-onset Leakage。")

# ==========================================
# 4. 可視化結果
# ==========================================
plt.figure(figsize=(12, 6))

# 畫出原始模擬信號
plt.plot(time_vector, mean_raw, label="Raw Signal (Mean)", color='gray', alpha=0.5)
# 畫出提取的 Gamma Envelope
plt.plot(time_vector, mean_env, label="Extracted Gamma Envelope", color='red', linewidth=2)

# 標記 Onset 點 (0s)
plt.axvline(x=0, color='blue', linestyle='--', label='Stimulus Onset (0s)')

# 標記語音結束點 (0.55s)
plt.axvline(x=speech_duration, color='green', linestyle=':', label='Speech Offset (0.55s)')

# 標記泄露觀察區 (-100ms 到 0ms)
plt.axvspan(-0.1, 0, color='yellow', alpha=0.4, label='Leakage Window (-100ms pre-onset)')

plt.title("Pre-onset Leakage Verification (Auditory High Gamma, 550ms Speech)")
plt.xlabel("Time relative to onset (seconds)")
plt.ylabel("Amplitude")
plt.legend(loc="upper right")
plt.grid(True)
plt.xlim(-0.5, 1.0) # 只放大顯示 -0.5s 到 1.0s 的關鍵區間，讓 leakage 更明顯
plt.show()