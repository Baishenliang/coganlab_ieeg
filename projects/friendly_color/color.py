#https://glasbey.readthedocs.io/en/latest/color_vision_deficiency.html
# conda activate glasbey-safe
import glasbey
import seaborn as sns

from matplotlib.colors import to_rgb  # 将 hex/命名色 转为 [0,1] RGB

sns.set()
palette = glasbey.create_palette(palette_size=5, colorblind_safe=True, cvd_severity=100, optimize_palette_search_radius=50)
sns.palplot(palette)


# 2) 输出 RGB 值
# 转为 [0,1] 浮点 RGB
rgb01 = [to_rgb(c) for c in palette]

# 转为 [0,255] 整数 RGB
rgb255 = [tuple(int(round(x*255)) for x in to_rgb(c)) for c in palette]

print("RGB [0,1]：")
for i, rgb in enumerate(rgb01, 1):
    print(f"{i:02d}: {rgb}")

print("\nRGB [0,255]：")
for i, rgb in enumerate(rgb255, 1):
    print(f"{i:02d}: {rgb}")
