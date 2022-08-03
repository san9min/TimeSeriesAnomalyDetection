# Anomaly_Detection

**Anomaly Detection with Reinforcement Learning for time series**

YAI 9기 이상민

---

## Introduction

I looked at time seires anomaly detection from the point of view of sparse reward problem in reinforcement learning. 

Many cases(time stamps) are not anomaly.  So if our agent could get rewards only at anomaly points, time series anomaly detection would be sparse reward problem. 

I tried using Intrinsic Curiosity Module, which uses intrinsic rewards to solve the sparse rewards problem. 

---

## Model

![model](https://user-images.githubusercontent.com/92682815/182004788-fac8fa81-793c-4029-af67-acb9d9c355a4.png)

* LSTM used as Q-net
* Action taken by Q-value and eps-greedy policy
* Total Reward = intrinsic(agent's pred err) + extrinsic(env)

---

## Metric

F1-scoring ; Harmonic mean of Precision and Recall

$$\text{F1 score } = \frac {2}{(\frac{1}{Precision} + \frac{1}{Recall})}$$

$$\text{where } Precision = \frac{TP}{TP+FP} \text{ and } Recall = \frac{TP}{TP+FN}$$

---

## Data


- **Yahoo A1 Benchmark**  
_Real_
<div>
<img src = "https://user-images.githubusercontent.com/92682815/182004942-c50aa1ce-18f5-4346-9d2d-a3460473841e.png" width=45%>
<img src = "https://user-images.githubusercontent.com/92682815/182004946-0bb37448-cc51-4e05-b96b-248cf36e45b4.png" width=45%>
</div>

- **Yahoo A2 Benchmark**  
_Synthetic_
<div>
<img src = "https://user-images.githubusercontent.com/92682815/182004955-a8afdd1d-3c4a-4f21-96a0-78898ca15f6d.png" width=45%>
<img src = "https://user-images.githubusercontent.com/92682815/182004962-12e76d5b-f352-4682-aea6-1c698b03ccd4.png" width=45%>
</div>

---

## Reference

* _[Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363)_
