# PaddleSEQ(PaddleSequence) - EAP

低代码的序列数据处理框架，使用时只需输入数据和几行代码即可自动完成模型搭建、训练、预测等工作，显著降低使用成本。  

## 项目依赖 - 在使用前需自行安装  
> paddlepaddle >= 2.0.1  或 paddlepaddle-gpu >= 2.0.1  
> paddlenlp  
> wget  

### 支持模型以及任务

#### 文本分类
LSTM  
BiLSTM   
BiLSTMAttention  
GRU   
BiGRU  
BOW  
CNN
## 预览效果
当前版本仍为EAP阶段，预计0.10-alpha版本在本月发布，发布时将提供whl包
Warning：暂不支持模型一键保存，推理时可能需要预先读取训练数据，下次迭代将支持该部分

### 训练部分 - 可自动从数据中出识别常见的深度学习任务类型，无需手动选择

```python
from paddleseq import AutoDataset, AutoModel, SEQNetwork

train_texts = ["文本1", "文本2", "文本3", "文本4", "文本5"]
train_labels = ["标签1", "标签2", "标签3", "标签4", "标签5"]
dataset = AutoDataset(train_texts=train_texts, train_labels=train_labels)

model = AutoModel(dataset, network=SEQNetwork.LSTM).run(batch_size=1)

```

### 推理部分 - 加载模型后输入数据即可得到结果

```python
out = model.infer(["文本1"])
print(out) -> "标签1"
```

### 尝鲜体验
#### 代码部分
```python
from paddleseq import AutoDataset, AutoModel, SEQDevice, SEQNetwork
from paddlenlp.datasets.chnsenticorp import ChnSentiCorp

# 加载ChnSentiCorp数据，并转换为["文本1", "文本2", "文本3", "文本4", "文本5"]形式
def get_chnsenticorp_data(mode="train"):
    texts = list()
    labels = list()
    for sample in ChnSentiCorp(mode):
        texts.append(sample[0])
        labels.append("积极" if sample[1] == "1" else "消极")
    return texts, labels


train_texts, train_labels = get_chnsenticorp_data()
dev_texts, dev_labels = get_chnsenticorp_data("dev")

# 调用AutoDataset
dataset = AutoDataset(train_texts=train_texts,
                      train_labels=train_labels,
                      eval_texts=dev_texts,
                      eval_labels=dev_labels)

# 准备模型 此处可指定device=SEQDevice.GPU参数来设置GPU模式，默认为在PaddlePaddle-GPU版本后自动选择为GPU模式，其他情况为CPU执行
model = AutoModel(dataset, network=SEQNetwork.LSTM)

# 开始训练
model.run(batch_size=16, epochs=30)

# 抽取5条数据进行预测
for i in range(5, 10):
    t = dev_texts[i]
    out = model.infer(t)
    # 输出文本前30个字以及对应的预测Top-5结果
    print(t[:min(30, len(t))], out)
```
#### 执行结果
```
房间地毯太脏，临近火车站十分吵闹，还好是双层玻璃。服务一般， [('消极', 0.99804676), ('积极', 0.0019532393)]
本来想没事的时候翻翻，可惜看不下去，还是和张没法比，他的书能 [('消极', 0.9957178), ('积极', 0.0042821327)]
这台机外观十分好,本人喜欢,性能不错,是LED显示屏,无线网 [('积极', 0.8226457), ('消极', 0.17735426)]
全键盘带数字键的 显卡足够强大.N卡相对A卡,个人偏向N卡  [('积极', 0.9961617), ('消极', 0.003838286)]
做工很漂亮，老婆很喜欢。T4200足够了，性价比不错的机器。 [('积极', 0.9995653), ('消极', 0.0004346288)]
```

## 项目许可
本项目采用MIT开源协议为开源许可，使用时请标注项目来源PaddleSEQ/PaddleSequence & 组织AgentMaker & GT-ZhangAcer
