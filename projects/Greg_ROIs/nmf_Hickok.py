import pandas as pd
import os
import numpy as np
from sklearn.decomposition import NMF
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import reconstruction_error
import matplotlib.pyplot as plt

# ---------------------------
# 1. LOAD DATA
# ---------------------------

lIFG_loc = 'D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\Greg_ROIs\\lIFG (vPCSA)'
lPMC_loc = 'D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\Greg_ROIs\\lPMC (dPCSA)'
Spt_loc = 'D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\Greg_ROIs\\Spt'

Spt_stim = pd.read_csv(os.path.join(Spt_loc, 'Spt_Stim_epoch.csv'))
lPMC_stim = pd.read_csv(os.path.join(lPMC_loc, 'lPMC (dPCSA)_Stim_epoch.csv'))
lIFG_stim = pd.read_csv(os.path.join(lIFG_loc, 'lIFG (vPCSA)_Stim_epoch.csv'))

Spt_go = pd.read_csv(os.path.join(Spt_loc, 'Spt_Go_epoch.csv'))
lPMC_go = pd.read_csv(os.path.join(lPMC_loc, 'lPMC (dPCSA)_Go_epoch.csv'))
lIFG_go = pd.read_csv(os.path.join(lIFG_loc, 'lIFG (vPCSA)_Go_epoch.csv'))

Spt_resp = pd.read_csv(os.path.join(Spt_loc, 'Spt_Resp_epoch.csv'))
lPMC_resp = pd.read_csv(os.path.join(lPMC_loc, 'lPMC (dPCSA)_Resp_epoch.csv'))
lIFG_resp = pd.read_csv(os.path.join(lIFG_loc, 'lIFG (vPCSA)_Resp_epoch.csv'))

# ---------------------------
# 2. BUILD FEATURE MATRIX
# ---------------------------

def build_matrix(stim, go, resp):
    stim_vals = stim.iloc[:,1:].values
    go_vals   = go.iloc[:,1:].values
    resp_vals = resp.iloc[:,1:].values

    # concatenate along time
    combined = np.concatenate([stim_vals, go_vals, resp_vals], axis=0)
    return combined.T  # electrodes x time

X_spt = build_matrix(Spt_stim, Spt_go, Spt_resp)
X_pmc = build_matrix(lPMC_stim, lPMC_go, lPMC_resp)
X_ifg = build_matrix(lIFG_stim, lIFG_go, lIFG_resp)

# stack all electrodes
X = np.vstack([X_spt, X_pmc, X_ifg])

# labels (optional)
labels = (["Spt"] * X_spt.shape[0] +
          ["lPMC"] * X_pmc.shape[0] +
          ["lIFG"] * X_ifg.shape[0])

# ---------------------------
# 3. PREPROCESSING
# ---------------------------

# NMF requires non-negative data
# so shift everything to be >= 0

X = X - X.min()

# optional: normalize each electrode
#X = X / (X.max(axis=1, keepdims=True) + 1e-8)

# ---------------------------
# 4. CHOOSE K
# ---------------------------

ks = range(2, 7)
errors = []

for k in ks:
    model = NMF(n_components=k, init='nndsvda', max_iter=1000, random_state=0)
    W = model.fit_transform(X)
    H = model.components_
    recon = np.dot(W, H)
    err = np.linalg.norm(X - recon)
    errors.append(err)

plt.figure()
plt.plot(ks, errors, marker='o')
plt.xlabel("k")
plt.ylabel("Reconstruction error")
plt.title("NMF model selection")
plt.show()

# ---------------------------
# 5. FINAL MODEL (k = 4)
# ---------------------------

k = 5
nmf = NMF(n_components=k, init='nndsvd', max_iter=2000, random_state=0)

W = nmf.fit_transform(X)   # electrodes x components
H = nmf.components_        # components x time

# ---------------------------
# 6. HARD ASSIGNMENT
# ---------------------------

# assign each electrode to its strongest component
subtypes = np.argmax(W, axis=1)

# ---------------------------
# 7. VISUALIZE COMPONENTS
# ---------------------------

plt.figure(figsize=(10,6))
for i in range(k):
    plt.plot(H[i], label=f"Component {i}")
plt.legend()
plt.title("NMF Components (time courses)")
plt.xlabel("Time (concatenated)")
plt.ylabel("Amplitude")
plt.show()

# ---------------------------
# 8. DISTRIBUTION BY ROI
# ---------------------------

import pandas as pd

df = pd.DataFrame({
    "ROI": labels,
    "Subtype": subtypes
})

print(pd.crosstab(df["ROI"], df["Subtype"]))

# ---------------------------
# 8. RAW DATA PLOT BY HARD ASSIGNMENT (修正版)
# ---------------------------
import matplotlib.pyplot as plt
import numpy as np

# 取得每顆電極所屬的最強 Component (硬分類)
subtypes = np.argmax(W, axis=1)

# 重建矩陣：加上 .T 將形狀轉置為 (電極數 x 時間點)，這樣才能垂直堆疊
X_stim_raw = np.vstack([Spt_stim.iloc[:, 1:].values.T, lPMC_stim.iloc[:, 1:].values.T, lIFG_stim.iloc[:, 1:].values.T])
X_go_raw   = np.vstack([Spt_go.iloc[:, 1:].values.T, lPMC_go.iloc[:, 1:].values.T, lIFG_go.iloc[:, 1:].values.T])
X_resp_raw = np.vstack([Spt_resp.iloc[:, 1:].values.T, lPMC_resp.iloc[:, 1:].values.T, lIFG_resp.iloc[:, 1:].values.T])

raw_epochs_data = [X_stim_raw, X_go_raw, X_resp_raw]
epochs_names = ['Stim', 'Go', 'Resp']

# 擷取三個階段的時間軸 (X軸)
# 既然 Rows 是時間，第一欄 (iloc[:, 0]) 就是物理時間的數值
t_stim = Spt_stim.iloc[:, 0].values.astype(float)
t_go   = Spt_go.iloc[:, 0].values.astype(float)
t_resp = Spt_resp.iloc[:, 0].values.astype(float)
time_axes = [t_stim, t_go, t_resp]

roi_names = ['Spt', 'lPMC', 'lIFG']
labels_array = np.array(labels)

# 建立 k 列 3 行的畫布
fig, axes = plt.subplots(k, 3, figsize=(12, 3 * k), sharey='row')

for comp_idx in range(k):
    for epoch_idx, epoch_name in enumerate(epochs_names):
        ax = axes[comp_idx, epoch_idx]
        t_vec = time_axes[epoch_idx]
        current_raw_data = raw_epochs_data[epoch_idx]
        
        for roi in roi_names:
            # 建立遮罩：找出同時符合「當前 Component」與「當前 ROI」的電極
            mask = (subtypes == comp_idx) & (labels_array == roi)
            
            if np.any(mask):
                # 取出這些電極的訊號並沿著電極維度 (axis=0) 取平均
                mean_signal = np.mean(current_raw_data[mask, :], axis=0)
                ax.plot(t_vec, mean_signal, label=roi, linewidth=1.5)
        
        # 畫上 x=0 的垂直對齊線
        ax.axvline(x=0, color='steelblue', linestyle='-', linewidth=1.5)
        
        
        # 設定標題與標籤
        if comp_idx == 0:
            ax.set_title(epoch_name, fontsize=12, fontweight='bold')
        if epoch_idx == 0:
            ax.set_ylabel(f'Subtype {comp_idx}', fontsize=12, fontweight='bold')
            
        # 只在第一張子圖顯示圖例
        if comp_idx == 0 and epoch_idx == 0:
            ax.legend(loc='upper left', frameon=False)
            
        # 美化邊框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[comp_idx, 0].set_ylim(-0.2, 5)
plt.tight_layout()
plt.show()