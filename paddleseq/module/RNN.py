import os
from paddlenlp.models import senta

from paddleseq import SEQNetwork
from paddleseq.dataset.convert import ClassesConvertRNN


# 需统一设置seq_len = None
class LSTMModel(senta.LSTMModel):
    def __init__(self, vocab_size, num_classes, *args, **kwargs):
        super(LSTMModel, self).__init__(vocab_size, num_classes, *args, **kwargs)

    def forward(self, text, seq_len=None):
        return super(LSTMModel, self).forward(text, None)


class BiLSTMAttentionModel(senta.BiLSTMAttentionModel):
    def __init__(self, vocab_size, num_classes, *args, **kwargs):
        super(BiLSTMAttentionModel, self).__init__(vocab_size, num_classes, *args, **kwargs)

    def forward(self, text, seq_len=None):
        return super(BiLSTMAttentionModel, self).forward(text, None)


class GRUModel(senta.GRUModel):
    def __init__(self, vocab_size, num_classes, *args, **kwargs):
        super(GRUModel, self).__init__(vocab_size, num_classes, *args, **kwargs)

    def forward(self, text, seq_len=None):
        return super(GRUModel, self).forward(text, None)


class BoWModel(senta.BoWModel):
    def __init__(self, vocab_size, num_classes, *args, **kwargs):
        super(BoWModel, self).__init__(vocab_size, num_classes, *args, **kwargs)

    def forward(self, text, seq_len=None):
        return super(BoWModel, self).forward(text, None)


class TextCNNModel(senta.TextCNNModel):
    def __init__(self, vocab_size, num_classes, *args, **kwargs):
        super(TextCNNModel, self).__init__(vocab_size, num_classes, *args, **kwargs)

    def forward(self, text, seq_len=None):
        return super(TextCNNModel, self).forward(text, None)


class Senta(senta.Senta):
    def __init__(self, vocab_size, num_classes, *args, **kwargs):
        super(Senta, self).__init__(vocab_size, num_classes, *args, **kwargs)

    def forward(self, text, seq_len=None):
        return super(Senta, self).forward(text, None)


# config 传参顺序需要保持一致，原classes可能存在没有**kwargs的情况
RNN_MODELS = {
    SEQNetwork.LSTM: {
        "network": LSTMModel,
        "classes_convert": ClassesConvertRNN,
        "classes_configs": {"vocab_size": 1256610,
                            "num_classes": None}
    },
    SEQNetwork.BiLSTMAttention: {
        "network": BiLSTMAttentionModel,
        "classes_convert": ClassesConvertRNN,
        "classes_configs": {"vocab_size": 1256610,
                            "num_classes": None}
    },
    SEQNetwork.GRU: {
        "network": GRUModel,
        "classes_convert": ClassesConvertRNN,
        "classes_configs": {"vocab_size": 1256610,
                            "num_classes": None}
    },
    SEQNetwork.BOW: {
        "network": BoWModel,
        "classes_convert": ClassesConvertRNN,
        "classes_configs": {"vocab_size": 1256610,
                            "num_classes": None}
    },
    SEQNetwork.CNN: {
        "network": TextCNNModel,
        "classes_convert": ClassesConvertRNN,
        "classes_configs": {"vocab_size": 1256610,
                            "num_classes": None}
    },
    SEQNetwork.SENTA: {
        "network": Senta,
        "classes_convert": ClassesConvertRNN,
        "classes_configs": {"vocab_size": 1256610,
                            "num_classes": None}
    }}
