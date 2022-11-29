#!/bin/bash
#SBATCH --job-name bao_icml                    # 任务名叫 example
#SBATCH --gres gpu:1 # 每个子任务都用一张 A100 GPU
#SBATCH --time 72:00:00                      # 子任务 1 天 1 小时就能跑完
#SBATCH --output %A_%a.out                  # 100个程序输出重定向到 [任务id]_[子任务序号].out
#SBATCH --mail-user x.tongda@nyu.edu        # 这些程序开始、结束、异常突出的时候都发邮件告诉我
#SBATCH --mail-type ALL

python -u train.py --model=VAE_TwoLayer_Alt
# python -u infer.py --resume=./VAE_TwoLayer/model.ckpt-3280.pt
