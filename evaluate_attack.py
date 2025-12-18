import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from attack_env import EmojiAttackEnv
from tqdm import tqdm

# ================= 配置区域 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 定义两个模型的路径 (Before & After)
MODELS_TO_EVALUATE = {
    "V1_Baseline": os.path.join(BASE_DIR, "runs/detect/emoji_defense_model/weights/best.pt"),
    "V2_Robust":   os.path.join(BASE_DIR, "runs_v2/detect/emoji_defense_model_v2/weights/best.pt")
}

VAL_DIR = os.path.join(BASE_DIR, "yolo_dataset/val/images")
LABEL_DIR = os.path.join(BASE_DIR, "yolo_dataset/val/labels")
PPO_MODEL_PATH = os.path.join(BASE_DIR, "emoji_attacker_ppo_v2.zip")
RESULT_DIR = os.path.join(BASE_DIR, "results_comparison")

# 攻击参数
MAX_STEPS = 20
IMAGE_LIMIT = 100 # 为了速度，限制测试50张，正式跑可以改大
# ===========================================

def get_target_id_from_label(label_path):
    if not os.path.exists(label_path): return 0
    with open(label_path, 'r') as f:
        line = f.readline()
        if not line: return 0
        return int(line.split()[0])

def generate_plots(df, output_dir):
    """
    根据评估结果生成对比图表
    """
    sns.set_theme(style="whitegrid")
    
    # 1. 攻击成功率对比 (Bar Chart)
    plt.figure(figsize=(8, 6))
    success_rates = df.groupby("Model")["success"].mean() * 100
    ax = sns.barplot(x=success_rates.index, y=success_rates.values, palette="viridis")
    plt.title("Attack Success Rate (Lower is Better)", fontsize=14, fontweight='bold')
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 100)
    for i in ax.containers:
        ax.bar_label(i, fmt='%.2f%%', padding=3)
    plt.savefig(os.path.join(output_dir, "comparison_success_rate.png"), dpi=300)
    plt.close()

    # 2. 鲁棒性分布 (Box Plot) - 核心图表
    # 展示模型最终的置信度分布。V1 应该很低(被攻破)，V2 应该很高(稳如泰山)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Model", y="final_conf", data=df, palette="Set2")
    plt.axhline(y=0.2, color='r', linestyle='--', label='Detection Threshold (0.2)')
    plt.title("Model Robustness Distribution under Attack", fontsize=14, fontweight='bold')
    plt.ylabel("Final Confidence Score")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "comparison_robustness_boxplot.png"), dpi=300)
    plt.close()

    # 3. 攻击权衡散点图 (Scatter Plot)
    # X轴: SSIM (画质), Y轴: Conf Drop (攻击效果)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="final_ssim", y="conf_drop", hue="Model", style="success", s=100, alpha=0.7)
    plt.title("Attack Impact: Visual Quality vs. Confidence Drop", fontsize=14)
    plt.xlabel("Image Quality (SSIM)")
    plt.ylabel("Confidence Drop (Attack Severity)")
    plt.savefig(os.path.join(output_dir, "comparison_attack_scatter.png"), dpi=300)
    plt.close()

    print(f"📊 图表已生成并保存至: {output_dir}")

def evaluate_single_model(model_name, model_path, ppo_agent, image_files):
    """
    针对单个模型运行完整的攻击评估循环
    """
    print(f"\n>>> 正在评估模型: {model_name} <<<")
    
    if not os.path.exists(model_path):
        print(f"⚠️ 跳过: 找不到模型文件 {model_path}")
        return []

    results = []
    # 为每个模型创建单独的图片保存目录
    img_save_dir = os.path.join(RESULT_DIR, f"images_{model_name}")
    if not os.path.exists(img_save_dir): os.makedirs(img_save_dir)

    for img_name in tqdm(image_files, desc=f"Attacking {model_name}"):
        img_path = os.path.join(VAL_DIR, img_name)
        label_path = os.path.join(LABEL_DIR, img_name.replace('.jpg', '.txt'))
        target_id = get_target_id_from_label(label_path)
        
        try:
            # 初始化环境 (加载对应的防御模型)
            env = EmojiAttackEnv(model_path, img_path, target_id)
        except Exception as e:
            continue

        obs, _ = env.reset()
        initial_conf = env.initial_conf
        
        # PPO 攻击循环
        done = False
        steps = 0
        final_conf = initial_conf
        final_ssim = 1.0
        
        while not done and steps < MAX_STEPS:
            action, _ = ppo_agent.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            
            final_conf = info['confidence']
            final_ssim = info['ssim']
            steps += 1
            if terminated or truncated:
                done = True
        
        # 判定攻击是否成功 (置信度是否低于 0.2)
        success = final_conf < 0.2
        
        results.append({
            "Model": model_name,
            "Image": img_name,
            "initial_conf": initial_conf,
            "final_conf": final_conf,
            "conf_drop": initial_conf - final_conf,
            "final_ssim": final_ssim,
            "success": success,
            "steps_taken": steps
        })

        # 保存成功的对抗样本 (用于报告中的 Case Study)
        if success:
            cv2.imwrite(os.path.join(img_save_dir, f"adv_{img_name}"), obs)
            
    return results

def main():
    if not os.path.exists(RESULT_DIR): os.makedirs(RESULT_DIR)
    
    # 检查 PPO
    if not os.path.exists(PPO_MODEL_PATH):
        print("❌ 错误: PPO 模型不存在，无法进行攻击评估。")
        return
    print("正在加载 PPO 攻击者...")
    ppo_agent = PPO.load(PPO_MODEL_PATH)

    # 获取图片列表
    all_images = [f for f in os.listdir(VAL_DIR) if f.endswith('.jpg')]
    if not all_images:
        print("❌ 错误: 验证集没有图片")
        return
    test_images = all_images[:IMAGE_LIMIT]

    all_results = []

    # --- 核心循环：分别评估 V1 和 V2 ---
    for name, path in MODELS_TO_EVALUATE.items():
        model_results = evaluate_single_model(name, path, ppo_agent, test_images)
        all_results.extend(model_results)

    if not all_results:
        print("❌ 没有产生任何结果，请检查模型路径。")
        return

    # --- 数据汇总与可视化 ---
    df = pd.DataFrame(all_results)
    
    # 1. 保存 CSV (包含两组数据)
    csv_path = os.path.join(RESULT_DIR, "final_comparison_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ 数据已保存: {csv_path}")
    
    # 2. 生成图表
    print("正在生成可视化图表...")
    try:
        generate_plots(df, RESULT_DIR)
    except Exception as e:
        print(f"⚠️ 绘图失败 (可能是缺少 seaborn): {e}")

    # 3. 打印最终摘要
    print("\n" + "="*40)
    print("🏆 FINAL SCOREBOARD (最终战绩)")
    print("="*40)
    summary = df.groupby("Model").agg(
        ASR=('success', lambda x: f"{x.mean()*100:.2f}%"),
        Avg_Conf_Drop=('conf_drop', 'mean'),
        Avg_SSIM=('final_ssim', 'mean')
    )
    print(summary)
    print("="*40)

if __name__ == "__main__":
    main()