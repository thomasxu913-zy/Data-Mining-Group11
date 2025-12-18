from ultralytics import YOLO
import os

def main():
    # 1. 获取当前脚本所在的绝对路径 (即 Project_Chameleon 文件夹)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 拼接 data.yaml 的绝对路径
    # 这样无论你在终端哪里运行，它都会去 train_defense.py 同级目录下找文件
    yaml_path = os.path.join(BASE_DIR, "data.yaml")

    print(f"配置文件路径: {yaml_path}")
    
    # 检查一下文件到底在不在，不在就报错提示
    if not os.path.exists(yaml_path):
        print(f"❌ 错误: 找不到文件 {yaml_path}")
        print("请检查 data.yaml 是否就在 train_defense.py 旁边。")
        return

    # 3. 加载预训练模型
    model = YOLO('yolov8n.pt') 

    # 4. 开始训练
    print("开始训练防御模型...")
    results = model.train(
        data=yaml_path,   
        epochs=15,        
        imgsz=640,        
        batch=16,
        name='emoji_defense_model',
        exist_ok=True,
        device='cpu'     
    )

    # 5. 验证
    print("正在验证模型准确率...")
    metrics = model.val()
    
    print(f"训练完成！")

if __name__ == "__main__":
    main()