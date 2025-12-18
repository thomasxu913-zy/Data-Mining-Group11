import os
import shutil
import cv2
from tqdm import tqdm
from stable_baselines3 import PPO
from attack_env import EmojiAttackEnv

# ================= 配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 原始训练数据
SRC_IMG_DIR = os.path.join(BASE_DIR, "yolo_dataset/train/images")
SRC_LBL_DIR = os.path.join(BASE_DIR, "yolo_dataset/train/labels")

# 新生成的对抗数据存放位置
ADV_DATA_DIR = os.path.join(BASE_DIR, "yolo_dataset/adversarial")
ADV_IMG_DIR = os.path.join(ADV_DATA_DIR, "images")
ADV_LBL_DIR = os.path.join(ADV_DATA_DIR, "labels")

# 模型路径
YOLO_PATH = os.path.join(BASE_DIR, "runs/detect/emoji_defense_model/weights/best.pt")
PPO_PATH = os.path.join(BASE_DIR, "emoji_attacker_ppo_v2.zip")
# =======================================

def get_target_id(label_path):
    if not os.path.exists(label_path): return 0
    with open(label_path, 'r') as f:
        line = f.readline()
        if not line: return 0
        return int(line.split()[0])

def main():
    # 1. 创建新目录
    if os.path.exists(ADV_DATA_DIR):
        shutil.rmtree(ADV_DATA_DIR) # 清空旧数据
    os.makedirs(ADV_IMG_DIR)
    os.makedirs(ADV_LBL_DIR)

    # 2. 加载模型
    print("正在加载 PPO Agent...")
    ppo_model = PPO.load(PPO_PATH)
    
    # 3. 遍历训练集生成样本
    img_files = [f for f in os.listdir(SRC_IMG_DIR) if f.endswith('.jpg')]
    
    img_files = img_files[:1000] 
    
    print(f"🚀 开始生成 {len(img_files)} 张对抗样本...")
    
    for img_name in tqdm(img_files):
        src_img_path = os.path.join(SRC_IMG_DIR, img_name)
        src_lbl_path = os.path.join(SRC_LBL_DIR, img_name.replace('.jpg', '.txt'))
        
        target_id = get_target_id(src_lbl_path)
        
        # 初始化环境
        try:
            env = EmojiAttackEnv(YOLO_PATH, src_img_path, target_id)
        except:
            continue
            
        obs, _ = env.reset()
        done = False
        steps = 0
        
        # PPO 攻击 5-10 步 (不需要攻击到死，只要加上了干扰特征就行)
        while not done and steps < 10:
            action, _ = ppo_model.predict(obs, deterministic=False) # 随机一点，增加多样性
            obs, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated: done = True
            
        # 4. 保存生成的对抗图片
        save_name = f"adv_{img_name}"
        cv2.imwrite(os.path.join(ADV_IMG_DIR, save_name), obs)
        
        # 5. 复制标签 
        shutil.copy(src_lbl_path, os.path.join(ADV_LBL_DIR, save_name.replace('.jpg', '.txt')))

    print("✅ 对抗数据集生成完毕！")
    print(f"存放位置: {ADV_DATA_DIR}")

if __name__ == "__main__":
    main()