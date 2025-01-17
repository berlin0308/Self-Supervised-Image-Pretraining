# Self-Supervised Image Pretraining and Classification



## Install Packages
With **Python3.8**, run the following command to install all the packages listed in the requirements.txt:

    pip3 install -r requirements.txt

## Pretrain on Mini-ImageNet

    python3 p1_pretrain.py --train_image_dir <your_train_dataset> --ckpt_dir <your_save_ckpt_dir>

## Fine-tune on Office-Home Dataset 

    python3 p1_finetune.py --backbone <your_backbone_path> --train_image_dir <your_train_dataset> --train_label <your_train_label_csv> --ckpt_dir <your_save_ckpt_dir>

## Inference

**OR** download the pretrained model directly:

    gdown 1QX2Cjf4iX13xKrMr8lzdp1_t5s87KegF # classifier_ep200_acc0.63.pth

# Q&A
If you have any problems related to HW1, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question to NTU COOL under HW1 Discussions
