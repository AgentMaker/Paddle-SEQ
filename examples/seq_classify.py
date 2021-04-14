from paddleseq import AutoDataset, AutoModel, SEQDevice, SEQNetwork
from paddlenlp.datasets.chnsenticorp import ChnSentiCorp

import random
import numpy as np
import paddle

# 设置随机种子，确保稳定性
random.seed(88)
np.random.seed(88)
paddle.seed(88)


def get_chnsenticorp_data(mode="train"):
    texts = list()
    labels = list()
    for sample in ChnSentiCorp(mode):
        texts.append(sample[0])
        labels.append("积极" if sample[1] == "1" else "消极")
    return texts, labels


train_texts, train_labels = get_chnsenticorp_data()
dev_texts, dev_labels = get_chnsenticorp_data("dev")

dataset = AutoDataset(train_texts=train_texts,
                      train_labels=train_labels,
                      eval_texts=dev_texts,
                      eval_labels=dev_labels)

# 此处可使用SEQDevice.CPU（CPU模式），切换GPU需要安装PaddlePaddle-GPU版本且需要更换为SEQDevice.GPU
model = AutoModel(dataset, network=SEQNetwork.LSTMSenta, device=SEQDevice.CPU)
model.run(batch_size=16, epochs=30, log_freq=100)
out = model.infer("GT太菜了，水平真不行")
print(out)
