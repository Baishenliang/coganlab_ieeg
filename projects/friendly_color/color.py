import glasbey
import seaborn as sns

from matplotlib.colors import to_hex, to_rgb

sns.set()
palette = glasbey.create_palette(palette_size=13, colorblind_safe=True, cvd_severity=100, optimize_palette_search_radius=50)
sns.palplot(palette)

print("Copy and paste this into your script:")
print("colors = [")
for i, c in enumerate(palette):
    comma = "," if i < len(palette) - 1 else ""
    print(f"    '{to_hex(c).upper()}'{comma}")
print("]")

rgb01 = [to_rgb(c) for c in palette]

rgb255 = [tuple(int(round(x*255)) for x in to_rgb(c)) for c in palette]

print("RGB [0,1]：")
for i, rgb in enumerate(rgb01, 1):
    print(f"{i:02d}: {rgb}")

print("\nRGB [0,255]:")
for i, rgb in enumerate(rgb255, 1):
    print(f"{i:02d}: {rgb}")
