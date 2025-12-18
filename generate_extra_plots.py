import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ================= 配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results_comparison")
CSV_PATH = os.path.join(RESULT_DIR, "final_comparison_results.csv")

# 检查 CSV 是否存在
if not os.path.exists(CSV_PATH):
    print(f"❌ 错误: 找不到数据文件 {CSV_PATH}")
    print("请先运行 evaluate_attack_enhanced.py 生成数据。")
    exit()

print(f"正在读取数据: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# 设置全局绘图风格 (学术风)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# ================= 图表 1: SSIM 画质分布对比曲线 (KDE Plot) =================
print("正在生成 SSIM 分布图...")
plt.figure(figsize=(10, 6))

# 使用 KDE (核密度估计) 画平滑曲线
sns.kdeplot(
    data=df, 
    x="final_ssim", 
    hue="Model", 
    fill=True, 
    common_norm=False, 
    palette="viridis",
    alpha=0.3,
    linewidth=2.5
)

plt.title("Distribution of Adversarial Image Quality (SSIM)", fontsize=14, fontweight='bold')
plt.xlabel("SSIM Score (1.0 = Original Quality, 0.0 = Destroyed)", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.xlim(0, 1.0)
plt.axvline(x=0.2, color='red', linestyle='--', label='Visibility Threshold (0.2)')
plt.legend(title='Model')

# 保存
save_path_ssim = os.path.join(RESULT_DIR, "extra_ssim_distribution.png")
plt.savefig(save_path_ssim, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ SSIM 曲线图已保存: {save_path_ssim}")


# ================= 图表 2: 置信度分布热力图 (Binning Heatmap) =================
print("正在生成置信度热力图...")
plt.figure(figsize=(12, 6))

# 1. 数据分箱 (Binning)
# 修正：在列表开头添加 0.0，使边界点变为 11 个，对应 10 个区间
bins = [0.0,0.025,0.05,0.075,0.1,0.5,0.9,0.95,0.975,1.0]

# 标签保持不变 (10个)
labels = ['[0.0-0.025)', '[0.025-0.05)', '[0.05-0.075)', '[0.075-0.1)', '[0.1-0.5)', '[0.5-0.9)', '[0.9-0.95)', '[0.95-0.975)', '[0.975-1.0)']

# 创建新列：Conf_Bin
# include_lowest=True 确保 0.0 这种极端情况也被包含在第一个区间内
df['Conf_Bin'] = pd.cut(df['final_conf'], bins=bins, labels=labels, include_lowest=True)

# 2. 计算每个模型在每个区间的样本数量
heatmap_data = df.groupby(['Model', 'Conf_Bin'], observed=False).size().unstack(fill_value=0)

# 3. 归一化 (转为百分比)
heatmap_data_pct = heatmap_data.div(heatmap_data.sum(axis=1), axis=0)

# 4. 绘制热力图
ax = sns.heatmap(
    heatmap_data_pct, 
    annot=True,       
    fmt=".1%",        
    cmap="YlGnBu",    
    cbar_kws={'label': 'Percentage of Samples'},
    linewidths=.5
)

plt.title("Confidence Distribution Heatmap (Adversarial Samples)", fontsize=14, fontweight='bold')
plt.xlabel("Confidence Range", fontsize=12)
plt.ylabel("Model Version", fontsize=12)
# 为了防止横坐标标签重叠，可以旋转一下
plt.xticks(rotation=45)

# 保存
save_path_heatmap = os.path.join(RESULT_DIR, "extra_confidence_heatmap.png")
plt.savefig(save_path_heatmap, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ 置信度热力图已保存: {save_path_heatmap}")

print("\n🎉 所有额外图表生成完毕！")