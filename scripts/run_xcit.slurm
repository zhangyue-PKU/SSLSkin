#!/bin/bash
#SBATCH -J xcit            # Job 名
#SBATCH -o ./backup/xcit_small/xcit_small.out  # 输出, 目录./out必须存在，否则无法成功提交job. 也可删除此行由系统自动指定.
#SBATCH --qos=short       # qos(quality of service): normal or debug, 对应不同优先级及最大可用时长.
#SBATCH -p RTX3090         # 指定 partition: geforce,V100,RTX3090,etc.
#SBATCH --cpus-per-task=20  # 申请 cpu processor 数; 可用内存与申请 cpu processor 数成正比.
#SBATCH --mem=64G          # 申请10G内存
#SBATCH --gres=gpu:2       # 申请 gpu 数
#SBATCH -N 1               # 申请节点数,一般为1
#SBATCH -t 2-00:00:00        # 申请 Job 运行时长0小时5分钟0秒,若要申请一天时间以上,如申请1天,书写格式为#SBATCH -t 1-00:00:00


# 上述每一项 SBATCH 参数不指定时系统会指定一个默认值

# 随着 Job 的提交和执行，slurm 会帮助用户在申请的节点上挨个执行下述命令
python -m torch.distributed.launch --nproc_per_node=2 main_dino.py --arch xcit_small_12_p16 --data_path ./data/pretrain --output_dir ./backup/xcit_small --num_workers 20