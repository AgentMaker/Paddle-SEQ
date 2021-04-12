# PaddleSequence(PaddleSEQ) - EAP

超简单的序列处理框架，任务支持不止NLP！



> 当前支持如下

文本分类：LSTM、GRU、BiLSTMAttention、BOW、CNN、SENTA  
文档/功能/模型 正在完善中...

## 尝鲜体验

训练部分 - 可自动从数据中出识别常见的深度学习任务类型，无需手动选择

```python
from paddleseq import AutoDataset, AutoModel, SEQNetwork

train_texts = ["文本1", "文本2", "文本3", "文本4", "文本5"]
train_labels = ["标签1", "标签2", "标签3", "标签4", "标签5"]
dataset = AutoDataset(train_texts=train_texts, train_labels=train_labels)

model = AutoModel(dataset, network=SEQNetwork.LSTM).run(batch_size=1)

```

推理部分 - 正在建设

```python
out = model.infer(["文本1"])
print(out)
```