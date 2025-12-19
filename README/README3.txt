-Dataset Description
The dataset used in this project consists of the following two files:

1. weibo_multi_emoji.cs (572.0K)
Purpose: It is used for the training and testing phases of the model, supporting the model's basic learning, parameter tuning, and effect verification.

2. weibo_data.rar (5.3M)
Purpose: It is used to test the generalization ability of the model after training is completed, to evaluate the model's adaptability and robustness on new data.



-Data Usage Logic
Model iteration is completed through the "training + testing dataset", and then the actual transferability of the model is verified using an independent "generalization test dataset" to ensure the objectivity of the evaluation results.