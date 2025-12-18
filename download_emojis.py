import os
import requests
import time

# ================= 配置 =================
# 我们使用 iamcal/emoji-data 的 GitHub 镜像，这里有提取好的 Apple 风格 PNG (160x160px)

BASE_URL = "https://raw.githubusercontent.com/iamcal/emoji-data/master/img-apple-160/"

# 目标保存目录
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "emojis")

# 经典的“阴阳怪气” Emoji 列表 (Unicode 编码映射)

EMOJI_MAP = {
    "0.png": "1f605",  # 😅 流汗黄豆 (Sweat Smile) - 尴尬、敷衍
    "1.png": "1f643",  # 🙃 倒脸 (Upside-down Face) - 极度嘲讽、无奈
    "2.png": "1f349",  # 🍉 吃瓜 (Watermelon) - 看戏、不嫌事大
    "3.png": "1f921",  # 🤡 小丑 (Clown Face) - 讽刺对方或自嘲
    "4.png": "1f644",  # 🙄 翻白眼 (Rolling Eyes) - 无语、鄙视
    "5.png": "1f914",  # 🤔 思考 (Thinking Face) - 质疑、装傻
    "6.png": "1f31a",  # 🌚 黑脸月亮 (New Moon Face) - 阴险、滑稽
    "7.png": "1f44c",  # 👌 OK手势 (OK Hand) - 表面答应实则敷衍
    "8.png": "1f975",  # 🥵 脸红流汗 (Hot Face) - 甚至可以用来反串“急了”
    "9.png": "1f485",  # 💅涂指甲 (Nail Polish) - 傲娇、不在乎
}

# =======================================

def download_file(url, save_path):
    try:
        print(f"正在下载: {url} ...", end="")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(" [成功]")
            return True
        else:
            print(f" [失败] 状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f" [出错] {e}")
        return False

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"创建目录: {SAVE_DIR}")
    
    print(f"准备下载 {len(EMOJI_MAP)} 个高清 Apple 风格表情...")
    print("来源: iamcal/emoji-data (GitHub)")
    print("-" * 30)

    success_count = 0
    for filename, code in EMOJI_MAP.items():
        # iamcal 的文件名格式是纯小写 hex，例如 1f605.png
        url = f"{BASE_URL}{code}.png"
        save_path = os.path.join(SAVE_DIR, filename)
        
        if download_file(url, save_path):
            success_count += 1
        
        
        time.sleep(0.5)

    print("-" * 30)
    print(f"下载完成! 成功: {success_count}/{len(EMOJI_MAP)}")
    print(f"文件已保存在: {SAVE_DIR}")
    print("现在你可以重新运行 dataset_generator.py 了！")

if __name__ == "__main__":
    main()