import pickle


class LabelManager:
    def __init__(self, cache_path="./cache/label.cache"):
        pass


class ClassesLabel:
    def __init__(self, label_list):
        self.label_list = label_list
        self.encoder_dict = dict()
        self.decoder_dict = dict()
        self.weight = None
        self._make_map()

    def encoded(self):
        return [self.encoder_dict[text] for text in self.label_list]

    def decoded(self):
        return [self.decoder_dict[text] for text in self.label_list]

    def classes_num(self):
        return len(self.weight)

    def _make_map(self):
        label_set = dict()
        for label in self.label_list:
            if label in label_set:
                label_set[label] += 1
            else:
                label_set[label] = 1

        label_max_num = max(label_set.values())
        self.weight = [round(min(label_max_num / value * 0.5, 10.), 7) for value in label_set.values()]
        label_reset = sorted(label_set.items(), key=lambda item: item[1], reverse=True)
        for key, value in label_reset:
            self.encoder_dict[key] = len(self.encoder_dict)
            self.decoder_dict[len(self.decoder_dict)] = key


if __name__ == '__main__':
    label_list = ["张", "张", "张", "GT", "T", "T"]
    label_info = ClassesLabel(label_list)
    pass
