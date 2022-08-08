# Anomaly Detection

**Time Series Anomaly Detection with Reinforcement Learning**

YAI 9기 이상민

---

## Introduction

I thought the time seires anomaly detection and sparse reward problem of reinforcement learning had analogy. Many cases(time stamps) are not anomaly, so if our agent could get rewards only at anomaly points, time series anomaly detection would be sparse reward problem. I tried using Intrinsic Curiosity Module, which uses intrinsic rewards to solve the sparse rewards problem. In the beginning, to make the time series anoamly detection task a sparse reward problem, I approximated TN reward and FP to zero, which just sent positive (+ $\epsilon$ ) and negative signal (- $\epsilon$ ). But then the agent went to anomaly too much (and so Recall $\approx$ 1).

So, I must re-approach to solve this problem. I have put out the encoder from ICM and make it shared by DQN and ICM. Encoder(I use LSTM) is trained with inverse model in ICM by supervised learning and can extract proper features. With these features, Q-function in DQN approximates Q-value better.

I use two buffers; Anomalous buffer and Normal buffer. Agent's anomalous exprience is memorized in Anomalous buffer and normal experience in Normal buffer.

1.  Easy to control : _Reward_

    - if you need to never miss anomalies, then just give more negative reward to False Negative

2.  Sampling anomaly data as much as you want : _Replay Buffer_
    - One of the reason anomaly detection problem is so hard is excessively unbalanced data to train the model.

---

## Main

![model](https://user-images.githubusercontent.com/92682815/182543308-2bab3cf1-e151-461f-a778-77342035d275.jpg)

- LSTM used as encoder
- Action taken by Q-value and eps-greedy policy
- Total Reward = intrinsic(agent's pred err) + extrinsic(env)

<img src = "https://user-images.githubusercontent.com/92682815/182543366-aeb40527-c909-40c8-a5c8-23c887e1ffce.jpg" height = 20%/>

- encoder trained by inverse model in ICM

![buffer](https://user-images.githubusercontent.com/92682815/183408236-766c2548-52a7-4987-9139-d4660c13a4cd.jpg)

- Store separately anomalous experience and normal experience in two buffers
- Sampling anomaly data and not anomaly data from each buffers
- batch size = $\alpha \times$ batch size + $(1-\alpha) \times$ batch size where $\alpha$ is anomaly samples ratio that you want

---

## Metric

F1-scoring ; Harmonic mean of Precision and Recall

$$\LARGE{\text{F1 score } = \frac {2}{(\frac{1}{Precision} + \frac{1}{Recall})} \\ = 2 \times \frac{Precison \times Recall}{Precision + Recall}} $$

$$\large{\text{where } Precision = \frac{TP}{TP+FP} \text{ and } Recall = \frac{TP}{TP+FN}}$$

|               | Yahoo A1 | Yahoo A2 | AIOps KPI |
| ------------- | -------- | -------- | --------- |
| **Precision** | 0.87     | 0.56     | 0.96      |
| **Recall**    | 0.84     | 0.56     | 0.96      |
| **F1-score**  | 0.85     | 0.56     | 0.96      |

---

## Data

- **Yahoo A1 Benchmark**  
_Real_(traffic to Yahoo services)  
time-series representing the metrics of various Yahoo services.
<div>
<img src = "https://user-images.githubusercontent.com/92682815/182004942-c50aa1ce-18f5-4346-9d2d-a3460473841e.png" width=45%>
<img src = "https://user-images.githubusercontent.com/92682815/182004946-0bb37448-cc51-4e05-b96b-248cf36e45b4.png" width=45%>
</div>

- **Yahoo A2 Benchmark**  
_Synthetic_(simulated)
<div>
<img src = "https://user-images.githubusercontent.com/92682815/182004955-a8afdd1d-3c4a-4f21-96a0-78898ca15f6d.png" width=45%>
<img src = "https://user-images.githubusercontent.com/92682815/182004962-12e76d5b-f352-4682-aea6-1c698b03ccd4.png" width=45%>
</div>

- **AIOps KPI**  
For Time series (labeled) Anomaly detection Datasets from AIOps Challenge
<div>
<img src = "https://user-images.githubusercontent.com/92682815/183410671-9b55b334-cbe5-41b9-aa05-300e9a8b4c3f.png" width=45%>
<img src = "https://user-images.githubusercontent.com/92682815/183410678-1f90b5aa-c3ef-49fd-99b6-b29bb21eeea3.png" width=45%>
</div>

---

## Test

    python test.py

---

## Reference

- _[Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363)_
