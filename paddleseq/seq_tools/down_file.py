import os

import wget


def download(url, path, file_name):
    if not os.path.exists(path):
        os.mkdir(path)
    file_path = os.path.join(path, file_name)
    if not os.path.exists(file_path):
        wget.download(url, out=file_path)
