Project_Root/
│
├── saved_models/                     # [Auto-Created] Directory for models
│   ├── distilbert-base-uncased/      # [CRITICAL] Local BERT model folder
│   ├── baseline_mlp.pth              # Saved checkpoints
│   ├── agent_actor.pth
│   ├──  agent_critic.pth
│   └── robust_mlp_model.pth
│
├── train.tsv                         # [REQUIRED] Training dataset (SST-2)
├── test.tsv                          # [REQUIRED] Test dataset (SST-2)
│
├── visualization_utils.py            # [REQUIRED] Visualization module
├── saved_models                 # Script to download BERT,RL agent,MLP model locally…… we trained before(of course you can change codes and retrain a new one sir)
├── DM_RL_Emoji.py                    # Main script for MLP Experiment
├── DM_CV_Emoji.py                     # Main script for Logistic Regression Experiment
│
└── README1.txt                        # This file

