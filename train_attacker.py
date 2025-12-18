from stable_baselines3 import PPO  
from attack_env import EmojiAttackEnv
import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "runs/detect/emoji_defense_model/weights/best.pt")

TRAIN_IMG_NAME = "train_13.jpg" 
VAL_DIR = os.path.join(BASE_DIR, "yolo_dataset/val/images")
IMG_PATH = os.path.join(VAL_DIR, TRAIN_IMG_NAME)

def main():
    env = EmojiAttackEnv(MODEL_PATH, IMG_PATH, target_class_id=8)

    
    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-3,  # PPO 通常用较小的 LR
        batch_size=64,
        n_steps=128,         # PPO 每次更新前的步数
        ent_coef=0.05,       
        device='cpu'         
    )
    
    print("🚀 开始 PPO 训练...")
    model.learn(total_timesteps=3000) 
    model.save("emoji_attacker_ppo")

    
    print("\n>>> 展示 PPO 攻击策略 <<<")
    obs, _ = env.reset()
    for i in range(20):
        # deterministic=False 让它按概率随机选，增加多样性
        action, _ = model.predict(obs, deterministic=False) 
        
        obs, reward, terminated, _, info = env.step(action)
        act_name = ["Pass", "Blur", "Noise", "Darken", "Pixel","Random Patch"][action]
        print(f"Step {i}: [{act_name}] -> Conf: {info['confidence']:.2f}, Reward: {reward:.2f}")
        
        if terminated:
            if info['confidence'] < 0.2:
                print("🏆 攻击成功！")
                cv2.imwrite("ppo_success.jpg", obs)
            break

if __name__ == "__main__":
    main()