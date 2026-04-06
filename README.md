# univfd_repro_v3

这版代码只保留你这次复现真正要用的两条线：
- 主模型：`CLIP:ViT-L/14 + linear head`
- baseline：`ResNet50 end-to-end`

数据来源对齐论文 / 官方 repo：
- `datasets/train/progan`
- `datasets/val/progan`
- `datasets/test/*`
- `datasets/diffusion_datasets/*`

---

## 1. 这份 README 是干什么的

你直接照着这里做就行，不用来回翻聊天记录。

这套代码面向 **Linux / WSL / 云服务器 SSH**。
如果你是在服务器上跑，最常用的命令就是：

```bash
bash setup_env.sh
bash downloaddata.sh --root ./datasets --parts train val test diffusion
bash scripts/train_linear.sh
bash scripts/eval_subset.sh
```

---

## 2. `.sh` 是什么，怎么用

`.sh` 是 Linux shell 脚本，就是把一串命令写成一个文件。
用法很简单：

```bash
bash 文件名.sh
```

比如：

```bash
bash setup_env.sh
bash downloaddata.sh --root ./datasets --parts train val test diffusion
bash scripts/train_linear.sh
bash scripts/train_resnet.sh
bash scripts/eval_subset.sh
```

如果你是通过 SSH 连服务器，基本就是这么用。

---

## 3. 环境要求

推荐：
- Linux 服务器
- 或 Windows + WSL
- Python 3.10 / 3.11 / 3.12 都可以
- CUDA 可用时默认走 GPU

### 3.1 不用 conda 也能跑

默认推荐用 `venv`，不强制 conda。

```bash
python -m venv .venv
source .venv/bin/activate
bash setup_env.sh
```

### 3.2 `setup_env.sh` 做了什么

它本质上就是：
- 检查当前 Python 版本
- 升级 pip
- 安装 `requirements.txt`
- 打印 torch 和 CUDA 是否可用

所以它默认假设：
- 你当前机器已经有 Python
- 如果你是 PyTorch 镜像，通常已经自带 torch / CUDA 运行环境

---

## 4. 项目目录结构

```text
univfd_repro_v3/
  README.md
  requirements.txt
  setup_env.sh
  downloaddata.sh
  scripts/
    train_linear.sh
    train_resnet.sh
    eval_subset.sh
  src/
    augment.py
    datasets.py
    eval.py
    models.py
    official_data.py
    train.py
    utils.py
```

---

## 5. 数据目录结构

下载并解压后，预期目录是：

```text
datasets/
  train/
    progan/
      airplane/
        0_real/
        1_fake/
      bedroom/
        0_real/
        1_fake/
      ...
  val/
    progan/
      airplane/
        0_real/
        1_fake/
      ...
  test/
    progan/
      0_real/
      1_fake/
    stylegan/
      0_real/
      1_fake/
    biggan/
      0_real/
      1_fake/
    cyclegan/
      0_real/
      1_fake/
    stargan/
      0_real/
      1_fake/
    gaugan/
      0_real/
      1_fake/
    ...
  diffusion_datasets/
    laion/
      *.png / *.jpg
    ldm_200/
      *.png / *.jpg
    ldm_100/
      *.png / *.jpg
    glide_100_27/
      *.png / *.jpg
    dalle/
      *.png / *.jpg
```

说明：
- `train/val` 用的是官方 ProGAN 分类目录
- `test` 用的是官方 CNN-based 生成器测试域
- `diffusion_datasets` 是官方额外放出的 diffusion / autoregressive 测试集

---

## 6. 下载官方数据

### 6.1 全量下载

```bash
bash downloaddata.sh --root ./datasets --parts train val test diffusion
```

### 6.2 只下部分数据

比如只下 train 和 val：

```bash
bash downloaddata.sh --root ./datasets --parts train val
```

比如只下 test 和 diffusion：

```bash
bash downloaddata.sh --root ./datasets --parts test diffusion
```

### 6.3 下载脚本支持的参数

```bash
--root             数据保存根目录
--parts            选择下载哪些部分：train val test diffusion
--keep-archives    保留压缩包
--no-skip-existing 即使本地已有文件也重新下载
```

例子：

```bash
bash downloaddata.sh --root /data/univfd --parts train val test diffusion --keep-archives
```

### 6.4 下载脚本依赖什么

它会用到：
- `wget` 或 `aria2c`
- `7z`
- `unzip`
- `gdown`（脚本会自动安装）

如果服务器缺少命令，可以先装：

```bash
sudo apt update
sudo apt install -y p7zip-full unzip wget aria2
```

### 6.5 是否需要登录 Hugging Face / Google Drive

通常：
- Hugging Face 公共链接：**不用登录**
- Google Drive 公开分享链接：**一般也不用登录**

但要注意：
- GDrive 可能限流
- 公开链接可能失效

所以如果 diffusion 下载失败，优先怀疑链接限流，不是你代码写错。

---

## 7. 训练主模型

主模型是：
- `CLIP:ViT-L/14 + linear head`
- backbone 冻结
- 只训练最后线性层

直接跑：

```bash
bash scripts/train_linear.sh
```

它实际调用的是：

```bash
python -m src.train \
  --train-root ./datasets/train/progan \
  --val-root ./datasets/val/progan \
  --arch clip_linear \
  --epochs 10 \
  --batch-size 64 \
  --lr 5e-5 \
  --num-workers 4 \
  --amp \
  --save-every 1 \
  --patience 5 \
  --output-dir ./runs/clip_vitl14_progan
```

---

## 8. 训练 baseline

baseline 是：
- `ResNet50 end-to-end`

直接跑：

```bash
bash scripts/train_resnet.sh
```

它实际调用的是：

```bash
python -m src.train \
  --train-root ./datasets/train/progan \
  --val-root ./datasets/val/progan \
  --arch resnet50 \
  --epochs 10 \
  --batch-size 64 \
  --lr 1e-4 \
  --num-workers 4 \
  --amp \
  --save-every 1 \
  --patience 5 \
  --output-dir ./runs/resnet50_progan
```

---

## 9. 评测

默认评测这 4 个官方 source：
- `stylegan`
- `biggan`
- `ldm_200`
- `dalle`

直接跑：

```bash
bash scripts/eval_subset.sh
```

它实际调用的是：

```bash
python -m src.eval \
  --data-root ./datasets \
  --sources stylegan biggan ldm_200 dalle \
  --arch clip_linear \
  --checkpoint ./runs/clip_vitl14_progan/ckpts/best.pt \
  --batch-size 64 \
  --num-workers 4 \
  --amp \
  --output-dir ./eval_results/clip_subset
```

如果你想自己指定 source，可以直接手动跑：

```bash
python -m src.eval \
  --data-root ./datasets \
  --sources stylegan gaugan ldm_200 glide_100_27 dalle \
  --arch clip_linear \
  --checkpoint ./runs/clip_vitl14_progan/ckpts/best.pt \
  --batch-size 64 \
  --num-workers 4 \
  --amp \
  --output-dir ./eval_results/custom_eval
```

---

## 10. 训练日志和输出文件

### 10.1 训练输出目录

主模型默认输出到：

```text
runs/clip_vitl14_progan/
```

baseline 默认输出到：

```text
runs/resnet50_progan/
```

### 10.2 训练目录内容

```text
runs/
  clip_vitl14_progan/
    logs/
      console.log
      train_history.csv
    ckpts/
      best.pt
      last.pt
      epoch_*.pt
    plots/
      train_loss.png
      val_metrics.png
    reports/
      args.json
      env.json
      meta.json
      best_metrics.json
      final_report.json
```

### 10.3 这些文件分别看什么

#### `logs/console.log`
完整终端日志，适合 SSH 下 `tail -f` 看训练。

#### `logs/train_history.csv`
每个 epoch 的数值记录。包括：
- `epoch`
- `lr`
- `train_loss`
- `ap`
- `roc_auc`
- `acc`
- `real_acc`
- `fake_acc`
- `best_threshold`
- `best_acc`
- `precision`
- `recall`
- `gpu_mem_mb`
- `train_seconds`
- `eval_seconds`
- `epoch_seconds`

#### `ckpts/best.pt`
按验证集 AP 保存的最优权重。

#### `ckpts/last.pt`
最后一个 epoch 的权重。

#### `plots/train_loss.png`
训练 loss 曲线。

#### `plots/val_metrics.png`
验证集 AP / Acc / Best-Acc 曲线。

#### `reports/best_metrics.json`
最佳 epoch 的关键指标。

#### `reports/final_report.json`
最终训练总结。

---

## 11. 评测输出目录

默认输出到：

```text
eval_results/clip_subset/
```

目录结构：

```text
eval_results/
  clip_subset/
    per_source_results.csv
    summary.csv
    summary.json
    plots/
      ap_by_source.png
      acc_by_source.png
      best_acc_by_source.png
      pr_<source>.png
      roc_<source>.png
```

### 11.1 评测指标都算了什么

每个 source 会输出：
- `AP`
- `ROC-AUC`
- `Acc@0.5`
- `Best-threshold Acc`
- `real_acc`
- `fake_acc`
- `precision`
- `recall`

### 11.2 汇总结果看哪里

#### `per_source_results.csv`
每个测试域一行，最适合直接贴到表格里。

#### `summary.csv`
所有 source 的平均结果。

#### `summary.json`
便于程序读。

#### `plots/ap_by_source.png`
各 source 的 AP 柱状图。

#### `plots/acc_by_source.png`
各 source 的 Acc 柱状图。

#### `plots/best_acc_by_source.png`
各 source 的 best-threshold Acc 柱状图。

#### `plots/pr_<source>.png`
每个 source 的 PR 曲线。

#### `plots/roc_<source>.png`
每个 source 的 ROC 曲线。

---

## 12. 支持的训练参数

`src.train` 目前支持这些常用参数：

```text
--train-root
--val-root
--train-categories
--val-categories
--limit-real
--limit-fake
--arch {clip_linear,resnet50}
--epochs
--batch-size
--num-workers
--lr
--weight-decay
--seed
--amp
--device
--output-dir
--resume
--save-every
--patience
```

### 12.1 子集训练

如果你想只跑部分 ProGAN 类别：

```bash
python -m src.train \
  --train-root ./datasets/train/progan \
  --val-root ./datasets/val/progan \
  --train-categories airplane,bedroom,church \
  --val-categories airplane,bedroom,church \
  --arch clip_linear \
  --epochs 10 \
  --batch-size 64 \
  --lr 5e-5 \
  --amp \
  --output-dir ./runs/clip_subset_categories
```

### 12.2 限制样本数

如果你想每个 split 限制 real/fake 数量：

```bash
python -m src.train \
  --train-root ./datasets/train/progan \
  --val-root ./datasets/val/progan \
  --limit-real 20000 \
  --limit-fake 20000 \
  --arch clip_linear \
  --epochs 10 \
  --batch-size 64 \
  --lr 5e-5 \
  --amp \
  --output-dir ./runs/clip_limited
```

### 12.3 恢复训练

```bash
python -m src.train \
  --train-root ./datasets/train/progan \
  --val-root ./datasets/val/progan \
  --arch clip_linear \
  --epochs 20 \
  --resume ./runs/clip_vitl14_progan/ckpts/last.pt \
  --output-dir ./runs/clip_vitl14_progan
```

---

## 13. SSH 上怎么跑

### 13.1 最简单

```bash
ssh your_server
cd your_repo
python -m venv .venv
source .venv/bin/activate
bash setup_env.sh
bash downloaddata.sh --root ./datasets --parts train val test diffusion
bash scripts/train_linear.sh
```

### 13.2 后台跑

```bash
nohup bash scripts/train_linear.sh > logs/train_linear.out 2>&1 &
tail -f logs/train_linear.out
```

### 13.3 更推荐 `tmux`

```bash
tmux new -s univfd
bash scripts/train_linear.sh
```

断开后重新连：

```bash
tmux attach -t univfd
```

---

## 14. 完整工作流

如果你现在就是要直接工作，不做 smoke test，那流程就是：

### 第一步：拉代码

```bash
git clone 你的仓库
cd 你的仓库
```

### 第二步：建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate
```

### 第三步：装依赖

```bash
bash setup_env.sh
```

### 第四步：下载官方数据

```bash
bash downloaddata.sh --root ./datasets --parts train val test diffusion
```

### 第五步：训练主模型

```bash
bash scripts/train_linear.sh
```

### 第六步：评测主模型

```bash
bash scripts/eval_subset.sh
```

### 第七步：训练 baseline

```bash
bash scripts/train_resnet.sh
```

### 第八步：评测 baseline

手动改一下 `--arch` 和 `--checkpoint` 再跑：

```bash
python -m src.eval \
  --data-root ./datasets \
  --sources stylegan biggan ldm_200 dalle \
  --arch resnet50 \
  --checkpoint ./runs/resnet50_progan/ckpts/best.pt \
  --batch-size 64 \
  --num-workers 4 \
  --amp \
  --output-dir ./eval_results/resnet_subset
```

---

## 15. 最后你该看哪些结果

### 主模型训练看：
- `runs/clip_vitl14_progan/logs/train_history.csv`
- `runs/clip_vitl14_progan/plots/train_loss.png`
- `runs/clip_vitl14_progan/plots/val_metrics.png`
- `runs/clip_vitl14_progan/reports/best_metrics.json`
- `runs/clip_vitl14_progan/reports/final_report.json`

### 主模型评测看：
- `eval_results/clip_subset/per_source_results.csv`
- `eval_results/clip_subset/summary.csv`
- `eval_results/clip_subset/plots/ap_by_source.png`
- `eval_results/clip_subset/plots/acc_by_source.png`

### baseline 看：
- `runs/resnet50_progan/...`
- `eval_results/resnet_subset/...`

---

## 16. 当前版本的已知限制

这版已经能直接工作，但你要知道几点：
- 训练数据下载依赖外部公开链接，链接失效会影响下载
- diffusion 数据可能被 GDrive 限流
- 目前没有 DDP 多卡训练
- 目前没有 EMA
- 目前主要目标是 **规范复现单机版**，不是大规模分布式工程版

---

## 17. 一句话版

如果你只想记最短流程，就记这个：

```bash
python -m venv .venv
source .venv/bin/activate
bash setup_env.sh
bash downloaddata.sh --root ./datasets --parts train val test diffusion
bash scripts/train_linear.sh
bash scripts/eval_subset.sh
```

