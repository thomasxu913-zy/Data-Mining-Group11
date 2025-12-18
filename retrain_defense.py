from ultralytics import YOLO
import os

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    YAML_PATH = os.path.join(BASE_DIR, "data.yaml")
    
    # 关键点：这次我们不加载 'yolov8n.pt'，而是加载我们之前训练好的 'best.pt'
    
    PREV_MODEL_PATH = os.path.join(BASE_DIR, "runs/detect/emoji_defense_model/weights/best.pt")
    
    print(f"加载上一代模型: {PREV_MODEL_PATH}")
    model = YOLO(PREV_MODEL_PATH)

    print("开始对抗训练 (Adversarial Training)...")
    results = model.train(
        data=YAML_PATH,
        epochs=10,        
        imgsz=640,
        batch=16,
        name='emoji_defense_model_v2', 
        exist_ok=True,
        device='cpu'
    )
    
    print("模型升级完成！新一代防御者已就位。")

if __name__ == "__main__":
    main()