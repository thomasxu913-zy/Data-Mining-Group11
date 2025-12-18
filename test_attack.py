from attack_env import EmojiAttackEnv
import cv2
import os
import random

# ================= 配置区域 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 模型路径
MODEL_PATH = os.path.join(BASE_DIR, "runs/detect/emoji_defense_model/weights/best.pt")

# 2. 你手动指定的测试图片与ID
# 注意：TARGET_ID 必须和你选的图片里的 Emoji 真实类别一致，否则初始置信度会很低
TEST_IMG_NAME = "train_13.jpg" 
TARGET_ID = 8  # 对应的 Emoji ID (比如 Hot_Face_🥵)

# ===========================================

def main():
    # --- 路径检查与自动修复逻辑 ---
    
    # 构建初始图片路径
    # 注意：这里假设你之前 split_data.py 把图片分到了 val/images
    # 如果找不到，脚本会自动去 train/images 或者 images 根目录找
    possible_paths = [
        os.path.join(BASE_DIR, "yolo_dataset/val/images", TEST_IMG_NAME),
        os.path.join(BASE_DIR, "yolo_dataset/train/images", TEST_IMG_NAME),
        os.path.join(BASE_DIR, "dataset/images", TEST_IMG_NAME)
    ]
    
    final_image_path = None
    
    # 1. 优先寻找你指定的图片
    for p in possible_paths:
        if os.path.exists(p):
            final_image_path = p
            print(f"✅ 找到指定图片: {final_image_path}")
            break
            
    # 2. 如果指定的图片找不到，自动随机挑一张作为保底
    if final_image_path is None:
        print(f"⚠️ 警告: 没找到 {TEST_IMG_NAME}，正在尝试自动选择一张...")
        val_dir = os.path.join(BASE_DIR, "yolo_dataset/val/images")
        if os.path.exists(val_dir) and len(os.listdir(val_dir)) > 0:
            random_file = random.choice([f for f in os.listdir(val_dir) if f.endswith('.jpg')])
            final_image_path = os.path.join(val_dir, random_file)
            
            
            
            print(f"🔄 自动切换为: {random_file}")
            print("❗注意: 自动切换图片的 TARGET_ID 可能不匹配，建议手动确认该图片的 Emoji 类别。")
        else:
            print("❌ 错误: 找不到任何图片，请检查 yolo_dataset 目录结构。")
            return

    # --- 检查模型是否存在 ---
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型文件: {MODEL_PATH}")
        print("请确认你已经成功运行了 train_defense.py 并且生成了 best.pt")
        return

    # --- 初始化环境 ---
    print("\n>>> 正在初始化对抗环境 (EmojiAttackEnv)...")
    try:
        env = EmojiAttackEnv(MODEL_PATH, final_image_path, TARGET_ID)
    except Exception as e:
        print(f"❌ 环境初始化失败: {e}")
        return
    
    # --- 开始测试 ---
    obs, _ = env.reset()
    
    print(f"\n🚀 开始随机攻击测试 (针对类别 ID: {TARGET_ID})...")
    print(f"{'Step':<5} | {'Action':<10} | {'Conf (置信度)':<12} | {'SSIM (画质)':<12} | {'Reward':<8}")
    print("-" * 60)

    for i in range(10):
        # 随机动作
        action = env.action_space.sample()
        
        # 执行
        obs, reward, terminated, truncated, info = env.step(action)
        
        action_name = ["Pass", "Blur", "Noise", "Darken", "Pixel"][action]
        
        # 打印状态
        # Conf 如果下降，说明攻击有效
        print(f"{i+1:<5} | {action_name:<10} | {info['confidence']:.4f}       | {info['ssim']:.4f}       | {reward:.2f}")
        
        # 保存过程图 
        save_name = f"attack_step_{i}.jpg"
        cv2.imwrite(save_name, obs)
        
        if terminated:
            print(f"\n🎉 终止条件触发！")
            if info['confidence'] < 0.1:
                print("   -> 攻击成功！模型已经认不出这是 Emoji 了。")
            else:
                print("   -> 图片画质损毁过高 (SSIM太低)，攻击判定失败。")
            break

    print(f"\n测试结束。请查看目录下的 {save_name} 观察图片变化。")

if __name__ == "__main__":
    main()