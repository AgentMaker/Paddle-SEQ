import paddle


class SmoothCE(paddle.nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        """
        平滑的交叉熵损失函数，可进一步提升准确率
        """
        super(SmoothCE, self).__init__(*args, **kwargs)

    def forward(self, ipt, label):
        label = paddle.nn.functional.one_hot(label, ipt.shape[-1])
        label = paddle.nn.functional.label_smooth(label)
        self.soft_label = True

        ret = paddle.nn.functional.cross_entropy(
            input,
            label,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            soft_label=self.soft_label,
            axis=self.axis,
            name=self.name)
        return ret
