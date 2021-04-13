import os.path

import numpy as np
import paddle

from paddlenlp.data import Vocab, JiebaTokenizer, Pad

from paddleseq.seq_tools.down_file import download


class BaseConvert:
    def __init__(self):
        self.convert_op = None
        self.max_len = None


class ClassesConvertRNN(BaseConvert):
    def __init__(self, vocab_file_path=None, max_len=128):
        vocab_file_name = "rnn_word_dict.txt"
        if vocab_file_path is None:
            vocab_file_path = "./vocab"
            download("https://paddlenlp.bj.bcebos.com/data/senta_word_dict.txt", vocab_file_path, vocab_file_name)
        super(ClassesConvertRNN, self).__init__()

        self.max_len = max_len
        self.vocab = Vocab.load_vocabulary(
            os.path.join(vocab_file_path, vocab_file_name),
            unk_token='[UNK]',
            pad_token='[PAD]')
        # 初始化分词器
        self.tokenizer = JiebaTokenizer(self.vocab)

        # 添加OP
        self.convert_op = self.classes_encoder

    def classes_encoder(self, text, label=None):
        ipt = self.tokenizer.encode(text)
        pad = np.zeros(self.max_len, dtype="int64")
        pad[:min(len(ipt), self.max_len)] = np.array(ipt[:min(len(ipt), self.max_len)]).astype("int64")
        if label is not None:
            lab = np.array(label).astype("int64")
            return pad, lab
        else:
            return pad
