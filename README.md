# Self-Supervised Image Pretraining and Classification

Self-supervised pre-training using [Bootstrap Your Own Latent (BYOL)](https://github.com/lucidrains/byol-pytorch/tree/master). 

BYOL relies on two neural networks, referred to as **online networks** and **target networks**, that interact and learn from each other. From an augmented view of an image, the online network is trained to predict the target network representation of the same image under a different augmented view. At the same time, the target network is updated with a slow-moving average of the online network. ([Reference](https://sh-tsang.medium.com/review-byol-bootstrap-your-own-latent-a-new-approach-to-self-supervised-learning-6f770a624441))


## Install Packages
With **Python3.8**, run the following command to install all the packages listed in the requirements.txt:

    pip3 install -r requirements.txt

## Pretrain on Mini-ImageNet

    python3 p1_pretrain.py --train_image_dir <your_train_dataset> --ckpt_dir <your_save_ckpt_dir>

+ Backbone: ResNet50
+ Pre-training Hyperparams
    - epoch: 1000
    - batch size: 256
    - initial lr: 0.00001
    - optimizer: Adam
    - scheduler: cosine annealing with warmup
+ Pretraining Result:
    ![image](https://github.com/user-attachments/assets/06473c0a-4dcf-41f3-ae67-98e505df9ad0)



## Fine-tune on Office-Home Dataset 

    python3 p1_finetune.py --backbone <your_backbone_path> --train_image_dir <your_train_dataset> --train_label <your_train_label_csv> --ckpt_dir <your_save_ckpt_dir>

+ Classifier: 2x FC layer (hidden_size=256) with dropout rate of 0.1
+ Comparative Analysis
    | Setting | Pre-training <br> (Mini-ImageNet) | Fine-tuning <br>(Office-Home dataset) | Validation Acc. % <br>(Office-Home dataset) | 
    | -------- | -------- | -------- | ------- |
    | A     | -     | Train full model (backbone + classifier) | 62.1 |
    | B     | w/ label     | Train full model (backbone + classifier) | 61.3 |
    | C     | w/o label     | Train full model (backbone + classifier) | 63.5 |
    | D     | w/ label     | Train classifier only | 26.0 |
    | E     | w/o label     | Train classifier only | 47.9 |

## Inference

Use the model you just trained **OR** download the pretrained model directly:

    gdown 1QX2Cjf4iX13xKrMr8lzdp1_t5s87KegF # classifier_ep200_acc0.63.pth

Run the following command to test it on a dataset:

    python3 p1_inference.py --input_csv=<test_data_csv> --input_image_dir=<test_data_folder> --output_csv=<pred_ouput_csv>

The output csv file will contain the model predictions.

## Visualization

The notebook ```p1_tsne.ipynb``` allows you to visualize the learned visual representation using **t-SNE (t-distributed Stochastic Neighbor Embedding)** with the output of the second last layer.

| Epoch 1 (val_acc: 14.5%) | Epoch 200 (val_acc: 63.5%) |
| -------- | -------- |
| ![image](https://github.com/user-attachments/assets/97d7b9c7-5a1f-4b09-8291-e7c0bfa61572) | ![image](https://github.com/user-attachments/assets/c449804c-672e-455c-b2fb-7a5192c9c6ee) |

