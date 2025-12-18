#!/usr/bin/env python
# coding: utf-8

# ### I. Core Ideas
# ### Task Definition: Transform the problem into a sequence prediction task of "Text Context + Previous Emojis → Predict the Next Emoji".
# ### Data Modeling: Convert both the text and the Emoji sequence into vectors (text encoded with BERT, Emojis encoded with Embedding).
# ### Model Selection: Use LSTM/GRU (to capture sequence dependencies) or Transformer (to capture long-distance context), and combine text semantics and the historical sequence of Emojis to predict the next Emoji.
# ### Pattern Mining: Infer the "Emoji sequence rules in the text context" from the model's prediction results / attention weights.

# In[2]:


pip install transformers torch tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple


# ## Data preprocessing

# In[4]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Load Data
df = pd.read_csv(r"C:\Users\Thomas\Desktop\weibo_multi_emoji.csv", encoding="utf-8")

# 2. Data Cleaning: Split Text and Emoji Sequences, Construct Training Samples
train_data = []
for idx, row in df.iterrows():
    text = row["text"]
    emojis = eval(row["emojis"])  # Emoji sequence (e.g. ['[Love you]', '[Love you]', '[Haha]'])

   # Create a sample of "text + the first n Emojis → the (n + 1)th Emoji"
    for i in range(1, len(emojis)):
        context_emojis = emojis[:i]  # Introduction Emoji (Input)
        target_emoji = emojis[i]    # Target Emoji (Output)
        train_data.append({
            "text": text,
            "context_emojis": context_emojis,
            "target_emoji": target_emoji
        })

# Convert to DataFrame
train_df = pd.DataFrame(train_data)
print(f"Number of training samples：{len(train_df)}")

# 3. Build an Emoji dictionary (assign an ID to each Emoji)
all_emojis = []
for emojis in df["emojis"]:
    all_emojis.extend(eval(emojis))
emoji_vocab = list(set(all_emojis))
emoji2id = {emoji: idx for idx, emoji in enumerate(emoji_vocab)}
id2emoji = {idx: emoji for emoji, idx in emoji2id.items()}
vocab_size = len(emoji_vocab)
print(f"Emoji dictionary size：{vocab_size}")

# 4. Text encoding (extracting text semantic vectors using BERT)
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
bert_model = BertModel.from_pretrained("bert-base-chinese")

def encode_text(text):
    """Encode the text using BERT and output the vector at the [CLS] position (768 dimensions)）"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=50)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()[0]  # [CLS]向量

# Encode all the texts (takes a long time and can be processed in batches)
train_df["text_embedding"] = train_df["text"].apply(encode_text)

# 5. Emoji sequence encoding (converted to ID sequence, padding to the desired length)
max_emoji_len = max([len(emojis) for emojis in train_df["context_emojis"]])
def encode_emojis(emojis):
    """Convert the Emoji sequence to an ID sequence, padding with 0s if necessary."""
    ids = [emoji2id[emoji] for emoji in emojis]
    padding = [0] * (max_emoji_len - len(ids))
    return ids + padding

train_df["context_emoji_ids"] = train_df["context_emojis"].apply(encode_emojis)
train_df["target_id"] = train_df["target_emoji"].map(emoji2id)

# 6. Construct training data
X_text = np.array(train_df["text_embedding"].tolist())  
X_emojis = np.array(train_df["context_emoji_ids"].tolist())  
y = np.array(train_df["target_id"].tolist())

# Divide the training set/test set
X_text_train, X_text_test, X_emojis_train, X_emojis_test, y_train, y_test = train_test_split(
    X_text, X_emojis, y, test_size=0.2, random_state=42
)


# ## Build the Model (LSTM + Text Semantic Fusion)
# ## Use LSTM to capture the sequential dependencies of Emoji sequences, and simultaneously fuse the text semantic vectors, ultimately predicting the next Emoji:

# In[7]:


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Dropout

# 1. Emoji sequence input branch
emoji_input = Input(shape=(max_emoji_len,))
emoji_embedding = Embedding(input_dim=vocab_size, output_dim=64, mask_zero=True)(emoji_input)
lstm_out = LSTM(128)(emoji_embedding)  # Whether Emoji sequences is dependent

# 2. Text semantic input branch
text_input = Input(shape=(768,))
text_dense = Dense(128, activation="relu")(text_input)  # Reduce the dimension of the text vector

# 3. Merge the two branches
concat = Concatenate()([lstm_out, text_dense])
dropout = Dropout(0.3)(concat)
output = Dense(vocab_size, activation="softmax")(dropout)  # Predict the probability of the next Emoji

# 4. Compile model
model = Model(inputs=[emoji_input, text_input], outputs=output)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# 5. Train the model
history = model.fit(
    [X_emojis_train, X_text_train], y_train,
    validation_data=([X_emojis_test, X_text_test], y_test),
    epochs=10,
    batch_size=32
)


# In[8]:


# 新增：导入绘图库
import matplotlib.pyplot as plt

# ===================== 新增：绘制准确率和Loss曲线 =====================
# --------------- 图1：训练/验证准确率变化曲线 ---------------
plt.figure(figsize=(8, 6))  # 设置画布大小
# 绘制训练准确率
plt.plot(history.history['accuracy'], color='#1f77b4', label='Training Accuracy', linewidth=2)
# 绘制验证准确率
plt.plot(history.history['val_accuracy'], color='#ff7f0e', label='Validation Accuracy', linewidth=2)
# 图表美化
plt.title('Model Accuracy During Training', fontsize=12)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.legend(loc='lower right', fontsize=9)
plt.grid(alpha=0.3)  # 添加网格（透明度0.3）
plt.xticks(range(0, 10, 1))  # x轴刻度（对应10个epoch）
plt.tight_layout()  # 自动调整布局
plt.show()

# --------------- 图2：训练/验证Loss变化曲线 ---------------
plt.figure(figsize=(8, 6))
# 绘制训练损失
plt.plot(history.history['loss'], color='#1f77b4', label='Training Loss', linewidth=2)
# 绘制验证损失
plt.plot(history.history['val_loss'], color='#ff7f0e', label='Validation Loss', linewidth=2)
# 图表美化
plt.title('Model Loss During Training', fontsize=12)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.legend(loc='upper right', fontsize=9)
plt.grid(alpha=0.3)
plt.xticks(range(0, 10, 1))
plt.tight_layout()
plt.show()


# ## IV. Pattern Mining (Reconstructing Emoji Order Rules from the Model)
# ## 1. Prediction Example: Analyzing the Order of Emojis Based on the Text Context

# In[4]:


import numpy as np 

def predict_next_emoji(text, context_emojis):
    """Given the input text and the preceding emojis, predict the next emoji."""
    # Encode text
    text_emb = encode_text(text).reshape(1, -1)
    # Encode the Emoji sequence (first convert to a numpy array and then reshape)
    emoji_ids_list = encode_emojis(context_emojis)  # Obtain the list
    emoji_ids = np.array(emoji_ids_list).reshape(1, -1)  # After converting to an array, then reshaping
    # Predict the probablility
    pred_probs = model.predict([emoji_ids, text_emb], verbose=0)[0]
    # Keep the top 3 prediction results
    top3_ids = np.argsort(pred_probs)[-3:][::-1]
    top3_emojis = [(id2emoji[idx], pred_probs[idx]) for idx in top3_ids]
    return top3_emojis

#example 
test_text = "今天生日超开心！"
test_context_emojis = ["[蛋糕]"]
pred = predict_next_emoji(test_text, test_context_emojis)
print(f"Text：{test_text}")
print(f"Pre order Emoji：{test_context_emojis}")
print(f"Predicted next Emoji：{pred}")


# ## Context - Emoji Combination Association Rules
# ## Analyzing Frequent Emoji Sequences in Different Textual Contexts:

# In[5]:


# Group by text keywords (such as "birthday", "happy", "sad")
happy_texts = train_df[train_df["text"].str.contains("开心|快乐|幸福")]
birthday_texts = train_df[train_df["text"].str.contains("生日|生快")]

# Counting the order of emojis in the context of "birthday"
birthday_emoji_pairs = []
for _, row in birthday_texts.iterrows():
    if len(row["context_emojis"]) >=1:
        pair = (row["context_emojis"][-1], row["target_emoji"])
        birthday_emoji_pairs.append(pair)

from collections import Counter
birthday_pair_counter = Counter(birthday_emoji_pairs)
print("High-frequency Emoji combinations in the context of birthdays：")
print(birthday_pair_counter.most_common(5))


# ## Context - Quantitative Association of Emoji Order (Constructing "Context Strength" Index)

# In[6]:


import numpy as np
from collections import defaultdict

# 1. Define the key context words and their weights (measuring the matching strength between the context and the text)
context_keywords = {
    "生日": ["生日", "生快", "蛋糕", "礼物"],
    "开心": ["开心", "快乐", "幸福", "哈哈"],
    "旅行": ["旅行", "打卡", "海边", "飞机"],
    "加油": ["加油", "努力", "坚持", "冲"],
    "难过": ["难过", "哭", "伤心", "泪"]
}

# 2. Calculate the matching strength of the text with various contexts
def calculate_context_strength(text):
    """Calculate the intensity (the frequency of keyword appearance) of the text for each context."""
    strength = defaultdict(int)
    for ctx, keywords in context_keywords.items():
        for kw in keywords:
            strength[ctx] += text.count(kw)
    # Normalization (sum equals 1)
    total = sum(strength.values())
    if total == 0:
        return {ctx: 0 for ctx in context_keywords}
    return {ctx: cnt/total for ctx, cnt in strength.items()}

# 3. Add context intensity features to the training set
train_df["context_strength"] = train_df["text"].apply(calculate_context_strength)

# 4. Analyzing the correlation between contextual intensity and Emoji transfer
def get_transition_by_context_strength(ctx_name, emoji_prev, threshold=0.5):
    """When the intensity of a certain context reaches ≥threshold, the transition distribution of emoji_prev"""
    # Select samples that meet the context intensity criteria and whose preceding Emoji is emoji_prev
    filtered = train_df[
        (train_df["context_strength"].apply(lambda x: x[ctx_name] >= threshold)) &
        (train_df["context_emojis"].apply(lambda x: x[-1] == emoji_prev if x else False))
    ]
    if len(filtered) < 5:
        return {}
    # Statistical distribution of Emoji types
    target_counter = Counter(filtered["target_emoji"])
    total = sum(target_counter.values())
    return {emo: cnt/total for emo, cnt in target_counter.items()}

# Analyze the influence of the intensity of the "birthday" context on the [cake] transfer
birthday_weak = get_transition_by_context_strength("生日", "[蛋糕]", threshold=0.1)
birthday_strong = get_transition_by_context_strength("生日", "[蛋糕]", threshold=0.8)

print("The influence of contextual intensity on the transfer of [cake]：")
print(f"Weak language context（strength<0.8）：{birthday_weak}")
print(f"Strong birthday context（strength≥0.8）：{birthday_strong}")


# ## Extract "Context - Emoji Sentence Order Rule Template"

# In[28]:


# 1. 定义规则模板结构：(语境类型, 前序Emoji序列, 后续Emoji, 置信度)
rule_templates = []

# 2. 遍历各语境，挖掘高置信度模板
for ctx_name in context_keywords:
    # 筛选该语境的样本
    ctx_samples = train_df[train_df["context_strength"].apply(lambda x: x[ctx_name] >= 0.7)]
    if len(ctx_samples) < 20:
        continue

    # 提取长度≥2的Emoji序列
    ctx_sequences = []
    for _, row in ctx_samples.iterrows():
        full_seq = row["context_emojis"] + [row["target_emoji"]]
        if len(full_seq) >= 3:
            ctx_sequences.append(tuple(full_seq))

    # 统计序列频率，筛选高频模板（长度=3）
    seq_counter = Counter(ctx_sequences)
    for seq, count in seq_counter.most_common(10):
        if len(seq) == 3 and count >= 3:
            # 计算置信度：该序列在语境中的出现占比
            ctx_total = sum(seq_counter.values())
            confidence = count / ctx_total
            rule_templates.append({
                "context": ctx_name,
                "prev_seq": seq[:2],
                "next_emo": seq[2],
                "confidence": confidence,
                "count": count
            })

# 3. 按置信度排序输出规则模板
rule_templates = sorted(rule_templates, key=lambda x: x["confidence"], reverse=True)

print("\nTop 10语境-Emoji语序规则模板：")
for i, rule in enumerate(rule_templates[:10]):
    print(f"{i+1}. 语境：{rule['context']} | 前序序列：{rule['prev_seq']} → 后续：{rule['next_emo']} | 置信度：{rule['confidence']:.2%}（出现{rule['count']}次）")


# ## Emoji word order rules for context sharing with similar semantics

# In[26]:


# 1. Calculate the similarity (cosine similarity) of Emoji transitions in different contextsfrom sklearn.metrics.pairwise import cosine_similarity

def get_transition_vector(ctx_name, top_n=20):
    """Obtain the transition probability vector of the top-n emojis in a certain context"""
    ctx_samples = train_df[train_df["context_strength"].apply(lambda x: x[ctx_name] >= 0.7)]
    # Construct the transition matrix（Top-n Emoji）
    top_emojis = [emo for emo, _ in Counter(ctx_samples["target_emoji"]).most_common(top_n)]
    transition_matrix = np.zeros((top_n, top_n))
    # Fill-in transition probability
    for _, row in ctx_samples.iterrows():
        if len(row["context_emojis"]) == 0:
            continue
        prev_emo = row["context_emojis"][-1]
        next_emo = row["target_emoji"]
        if prev_emo in top_emojis and next_emo in top_emojis:
            i = top_emojis.index(prev_emo)
            j = top_emojis.index(next_emo)
            transition_matrix[i,j] += 1
    # Normalization
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums!=0)
    return transition_matrix.flatten()

# 2. Calculate the similarity of each context transition matrix
contexts = list(context_keywords.keys())
ctx_vectors = {ctx: get_transition_vector(ctx) for ctx in contexts}

# Construct a similarity matrix
similarity_matrix = np.zeros((len(contexts), len(contexts)))
for i, ctx1 in enumerate(contexts):
    for j, ctx2 in enumerate(contexts):
        vec1 = ctx_vectors[ctx1].reshape(1, -1)
        vec2 = ctx_vectors[ctx2].reshape(1, -1)
        similarity_matrix[i,j] = cosine_similarity(vec1, vec2)[0][0]

# 3. Visualized cross-cultural similarity
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']

plt.figure(figsize=(8, 6))
im = plt.imshow(similarity_matrix, cmap="Blues")
plt.xticks(range(len(contexts)), contexts, rotation=45)
plt.yticks(range(len(contexts)), contexts)
plt.title("Similarity Matrix of Emoji Order Rules in Different Contexts")

# Add numerical annotations
for i in range(len(contexts)):
    for j in range(len(contexts)):
        plt.text(j, i, f"{similarity_matrix[i,j]:.2f}", ha="center", va="center")

plt.colorbar(im)
plt.tight_layout()
plt.savefig(r"C:\Users\Thomas\Desktop\context_similarity.png", dpi=300)
plt.show()


# ## Evaluate the generalization ability of the rules using the test set

# In[31]:


# 1. Construct validation samples from the test set
test_data = []
for idx, row in df.iterrows():
    text = row["text"]
    emojis = eval(row["emojis"])
    if len(emojis) >=3:
        test_data.append({
            "text": text,
            "full_seq": emojis,
            "context_strength": calculate_context_strength(text)
        })

# 2. Use the rule template to predict the subsequent emojis and calculate the accuracy rate
correct = 0
total = 0
for sample in test_data:
    ctx = max(sample["context_strength"], key=sample["context_strength"].get)
    prev_seq = tuple(sample["full_seq"][:2])
    true_next = sample["full_seq"][2]

    # Matching rule template
    matched_rule = None
    for rule in rule_templates:
        if rule["context"] == ctx and rule["prev_seq"] == prev_seq:
            matched_rule = rule
            break

    if matched_rule:
        total +=1
        if matched_rule["next_emo"] == true_next:
            correct +=1

if total >0:
    rule_accuracy = correct / total
    print(f"\Accuracy of rule template generalization：{rule_accuracy:.2%}")
else:
    print("\nThe test set has no matching samples, and the generalization ability of the rules needs to be verified.")


# In[7]:


# 1. Define rule template structure: (context_type, previous_emoji_sequence, next_emoji, confidence)
rule_templates = []

# 2. Iterate over each context to mine high-confidence templates
for ctx_name in context_keywords:
    # Step 1: Filter samples with strong context alignment (≥0.7) AND non-empty emoji sequences
    # (Exclude posts with no emojis to avoid diluting the denominator)
    ctx_samples = train_df[
        (train_df["context_strength"].apply(lambda x: x.get(ctx_name, 0) >= 0.7)) &  # Strong context
        (train_df["context_emojis"].apply(lambda x: len(x) >= 1))  # At least 1 previous emoji
    ]

    # Skip contexts with insufficient valid samples (minimum 20 required for statistical significance)
    if len(ctx_samples) < 20:
        continue

    # Step 2: Extract full emoji sequences (previous emojis + target emoji) with length ≥3
    valid_full_sequences = []
    for _, row in ctx_samples.iterrows():
        # Combine previous emojis and target emoji to form the complete sequence
        full_emoji_seq = row["context_emojis"] + [row["target_emoji"]]
        # Only keep sequences with length ≥3 (to extract 3-token patterns: [e1, e2] → e3)
        if len(full_emoji_seq) >= 3:
            valid_full_sequences.append(tuple(full_emoji_seq))

    # Skip if no valid 3-length sequences exist
    if len(valid_full_sequences) == 0:
        continue

    # Step 3: Extract 3-length sub-sequences (core pattern: [e1, e2] → e3)
    three_token_sequences = []
    for full_seq in valid_full_sequences:
        # Slice to get all consecutive 3-token sub-sequences in the full sequence
        for i in range(len(full_seq) - 2):
            three_token_seq = full_seq[i:i+3]
            three_token_sequences.append(three_token_seq)

    # Step 4: Count frequency of 3-token sequences
    seq_counter = Counter(three_token_sequences)
    # Total number of valid 3-token sequences (corrected denominator for confidence)
    total_valid_3token_seqs = sum(seq_counter.values())

    # Step 5: Filter high-frequency templates (minimum 3 occurrences)
    for seq, count in seq_counter.most_common(10):
        if count >= 3 and total_valid_3token_seqs > 0:
            # Calculate confidence: (count of the sequence) / (total valid 3-token sequences in context)
            confidence = count / total_valid_3token_seqs
            rule_templates.append({
                "context": ctx_name,
                "prev_seq": seq[:2],  # First 2 emojis (input sequence)
                "next_emo": seq[2],   # 3rd emoji (predicted next emoji)
                "confidence": confidence,
                "count": count
            })

# 3. Sort rule templates by confidence (descending) and output top 10
rule_templates = sorted(rule_templates, key=lambda x: x["confidence"], reverse=True)

print("\nTop 10 Context-Emoji Order Rule Templates:")
for i, rule in enumerate(rule_templates[:10]):
    print(f"{i+1}. Context: {rule['context']} | Previous Sequence: {rule['prev_seq']} → Next Emoji: {rule['next_emo']} | Confidence: {rule['confidence']:.2%} (Occurrences: {rule['count']})")


# In[ ]:




