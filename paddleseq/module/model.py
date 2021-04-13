from typing import List
from copy import deepcopy, copy

import paddle
import paddlenlp.metrics

from paddleseq import SEQNetwork
from paddleseq import SEQDevice
from paddleseq import SEQTask

from .unit.re_loss import SmoothCE
from paddleseq.module.task_manager import *


class AutoModel(paddle.Model):
    def __init__(self,
                 dataset,
                 network=SEQNetwork.LSTM,
                 task_config=SEQTask.AUTO,
                 device=SEQDevice.AUTO):
        # Flags
        self.prepare_flag = False

        self.train_dataset = dataset
        # 此处有更好方案
        self.eval_dataset = copy(dataset).is_eval()

        self.network = network

        if task_config == SEQTask.AUTO:
            self.train_dataset.analysis_task()
            self.select_task_config()
        else:
            self.task_config = task_config
            # 此处添加xxx_task_config()

        self.input_list = self.model_config.input_list
        self.label_list = self.model_config.label_list

        # optimizer, loss_opt, metrics
        self.prepare_ops = self.model_config.prepare_ops

        self.network = self.model_config.network

        if device == SEQDevice.AUTO:
            pass
        elif device == SEQDevice.GPU:
            paddle.device.set_device("gpu")
        elif device == SEQDevice.XPU:
            paddle.device.set_device("xpu")
        elif device == SEQDevice.CPU:
            paddle.device.set_device("cpu")
        else:
            raise Exception("无法识别该device，建议设置为device=SEQDevice.AUTO")

        super().__init__(network=self.network, inputs=self.input_list, labels=self.label_list)

    def _prepare(self, is_infer=False):
        if is_infer:
            self.prepare()
        else:
            self.prepare(*self.prepare_ops)

    def set_inputs(self, inputs: List[paddle.static.InputSpec]):
        self.input_list = inputs

    def set_labels(self, labels: List[paddle.static.InputSpec]):
        self.label_list = labels

    def select_task_config(self):
        # 正在工程
        self.model_config = ClassesConfig(self.train_dataset, self.network)

    def run(self,
            batch_size=8,
            epochs=5,
            eval_freq=1,
            log_freq=10,
            save_dir=None,
            save_freq=1,
            verbose=2,
            drop_last=False,
            shuffle=True,
            num_workers=0,
            callbacks=None):
        if not self.prepare_flag:
            self._prepare()
            self.prepare_flag = True
        self.fit(
            train_data=self.train_dataset,
            eval_data=self.eval_dataset,
            batch_size=batch_size,
            epochs=epochs,
            eval_freq=eval_freq,
            log_freq=log_freq,
            save_dir=save_dir,
            save_freq=save_freq,
            verbose=verbose,
            drop_last=drop_last,
            shuffle=shuffle,
            num_workers=num_workers,
            callbacks=callbacks)

    def infer(self, text):
        if not self.prepare_flag:
            self._prepare()
            self.prepare_flag = True
        text = self.train_dataset.convert_op(text)
        out = self.predict_batch([[text]])
        out = self.model_config.get_result(out, self.train_dataset.label_decoder_opt)
        return out


class ModelConfig:
    def __init__(self,
                 dataset,
                 network=SEQNetwork.LSTM):
        self.dataset = dataset
        self.network = None
        self.input_list = None
        self.label_list = None
        self.prepare_ops = None

        self.build(network)

    def build(self, network_name):
        self.network = None


class ClassesConfig(ModelConfig):
    def __init__(self,
                 dataset,
                 network=SEQNetwork.LSTM):
        self.dataset = dataset
        super(ClassesConfig, self).__init__(dataset, network)

    def build(self, network_name, **kwargs):
        network_config = MODELS[network_name]
        network_config["classes_configs"]["num_classes"] = self.dataset.classes_num
        network_name = network_config["network"](*network_config["classes_configs"].values())
        self.network = network_name
        self.input_list = [paddle.static.InputSpec([-1, self.dataset.max_len], dtype="int64", name="inputs")]
        self.label_list = [paddle.static.InputSpec([-1, 1], dtype="int64", name="label")]
        weight = paddle.to_tensor(self.dataset.weight, dtype="float32")
        loss_opt = paddle.nn.CrossEntropyLoss(weight=weight)
        optimizer = paddle.optimizer.AdamW(0.0001, parameters=self.network.parameters())
        metrics = paddle.metric.Accuracy()
        self.prepare_ops = [optimizer, loss_opt, metrics]

    @staticmethod
    def get_result(out, decoder):
        out = paddle.to_tensor(out)
        prob = paddle.nn.functional.softmax(out).numpy()
        top_k = paddle.tensor.argsort(out).numpy()
        return [(decoder(t), p) for p, t in zip(prob[0][0][:min(5, len(prob[0][0]))],
                                                top_k[0][0][:min(5, len(prob[0][0]))])]
