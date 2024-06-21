# Feedback-ICL

Language Options:  [中文](#中文说明) | [English](#english-version) 

### 中文说明

#### 论文信息

本仓库开源了以下论文的代码：

- 标题：[Improving In-Context Learning with Prediction Feedback for Sentiment Analysis](https://arxiv.org/pdf/2406.02911)
- 作者：Hongling Xu, Qianlong Wang, Yice Zhang, Min Yang, Xi Zeng, Bing Qin, Ruifeng Xu*
- 会议：ACL-2024 Findings（短文）

#### 运行说明
- 各任务数据集位于dataset中（可以参考./dataset/sampling_example.ipynb完成候选样例采样）
- 获取候选样本的先验预测结果
    ```bash
    bash scripts/run_pre.sh
    ```
- 在候选样本中进行样例检索
    ```bash
    bash scripts/run_sim.sh
    ```
- 推理
    ```bash
    bash scripts/run_ficl.sh
    ```



---

### English Version

#### Information

This repository open-sources the code for the following paper:

- Title: [Improving In-Context Learning with Prediction Feedback for Sentiment Analysis](https://arxiv.org/pdf/2406.02911)
- Authors: Hongling Xu, Qianlong Wang, Yice Zhang, Min Yang, Xi Zeng, Bing Qin, Ruifeng Xu*
- Conference: ACL-2024 Findings (short)

### Instructions

- The datasets for each task are located in the `dataset` folder (refer to `./dataset/sampling_example.ipynb` for completing candidate example sampling).
- Obtain the prior prediction results for the candidate samples:
    ```bash
    bash scripts/run_pre.sh
    ```
- Perform in-context example retrieval among the candidate samples:
    ```bash
    bash scripts/run_sim.sh
    ```
- Inference:
    ```bash
    bash scripts/run_ficl.sh
    ```

