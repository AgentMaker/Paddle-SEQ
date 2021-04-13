from collections import Counter

import paddle
from paddle.io import Dataset

from paddleseq import SEQTask
from paddleseq.dataset.label_manager import ClassesLabel
from paddleseq.dataset.convert import ClassesConvertRNN


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()
        self.eval_mode = False

        # 进入model时将被设置
        self.convert_op = None

        self.train_texts = None
        self.train_texts_b = None
        self.train_labels = None

        self.eval_texts = None
        self.eval_texts_b = None
        self.eval_labels = None

    def ret_sample(self, index, is_eval: bool = False):
        data = [self.train_texts, self.train_texts_b, self.train_labels] if not is_eval \
            else [self.eval_texts, self.eval_texts_b, self.eval_labels]
        pack = []
        for d in data:
            if d is not None:
                pack.append(d[index])
        return pack

    def sample(self, index):
        if self.eval_mode:
            return self.ret_sample(index, is_eval=True)
        else:
            return self.ret_sample(index)

    def __getitem__(self, index):
        if self.convert_op:
            return self.convert_op(*self.sample(index))
        else:
            return self.sample(index)

    def __len__(self):
        return len(self.train_texts) if len(self.train_texts) != 0 is False else len(self.eval_texts)

    def is_train(self):
        self.eval_mode = False

    def is_eval(self):
        if self.eval_labels is not None:
            self.eval_mode = True


class AutoDataset(BaseDataset):
    def __init__(self,
                 train_texts: list,
                 train_texts_b: list = None,
                 train_labels: list = None,
                 eval_texts: list = None,
                 eval_texts_b: list = None,
                 eval_labels: list = None,
                 task=SEQTask.AUTO):
        super(AutoDataset, self).__init__()
        self.task = task

        self.train_texts = train_texts
        self.train_texts_b = train_texts_b
        self.train_labels = train_labels
        self.eval_texts = eval_texts
        self.eval_texts_b = eval_texts_b
        self.eval_labels = eval_labels

        if self.eval_labels is None:
            self.labels = self.train_labels
        elif self.train_labels is None:
            self.labels = list()
        else:
            self.labels = self.train_labels + self.eval_labels

    def _apply_classes_dataset(self):
        self.label_opt = ClassesLabel(self.labels)
        self.label_encoder_opt = self.label_opt.encoder
        self.label_decoder_opt = self.label_opt.decoder
        self.task = SEQTask.CLASSES_TASK
        self.labels = self.label_opt.encoded()
        self.train_labels = self.labels[:len(self.train_labels)]
        self.eval_labels = self.labels[len(self.train_labels):]
        self.weight = self.label_opt.weight
        self.classes_num = self.label_opt.classes_num()
        convert = ClassesConvertRNN()
        self.convert_op = convert.convert_op
        self.max_len = convert.max_len

    def analysis_task(self):
        print("AutoDataset：正在从 分类/匹配/相似度/序列标注/文本生成 中搜索数据集适合的深度学习任务类型...")
        select_task_mag = "\n若未选择正确任务类型，可手动指定AutoDataset(task=paddleseq.SEQTask.任务名)"

        # 判断是否为分类任务 - 若Label中重复较多且无单个Label样本出现视为分类任务
        if len(set(self.labels)) / len(self.labels) <= 0.5 and min(Counter(self.labels)) != 1:
            print("AutoDataset：当前数据集适合分类任务" + select_task_mag)
            self.task = SEQTask.CLASSES_TASK
            self._apply_classes_dataset()

        elif len(self.labels) == len(self.train_texts) == len(self.train_texts_b):
            print("AutoDataset：当前数据集适合文本匹配/相似度任务" + select_task_mag)
            # 也可能适合文本搜索
            self.task = SEQTask.AUTO
        # elif len(self.texts) == len(self.texts_b) and len(self.labels) == 0:
        #     # 也可能适合机器翻译、阅读理解
        #     print("AutoDataset：当前数据集适合实体标注任务")
        #     self.task = SEQTask.
        # elif len(self.texts_b) == 0 and len(self.labels) == 0:
        #     print("AutoDataset：当前数据集适合文本生成任务")
        #     self.task = SEQTask.MATCH_TASK
        else:
            raise Exception("AutoDataset：暂时无法判断适合哪种深度学习任务" + select_task_mag)
