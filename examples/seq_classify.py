from paddleseq import AutoDataset, AutoModel, SEQDevice, SEQNetwork
from paddlenlp.datasets.chnsenticorp import ChnSentiCorp


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
                      eval_labels=dev_texts)

model = AutoModel(dataset, network=SEQNetwork.LSTM, device=SEQDevice.CPU)
model.run(batch_size=16, epochs=5)
out = model.infer("商品的不足暂时还没发现，京东的订单处理速度实在.......周二就打包完成，周五才发货...")
print(out)
