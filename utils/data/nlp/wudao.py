import os
import glob
import json


class WuDao:
    def __init__(self, root):
        self.files = glob.glob(os.path.join(root, '*.json'))

    def open_func_file(self, file):
        with open(file) as f:
            contents = json.load(f)
        return contents

if __name__ == '__main__':
    from dlx.utils.data.nlp.file_segments_dataloader import FileSegmentsDataloader

    root = '/dataset/fd5061f6/chinese_data/WuDao/'
    wudao_dataset = WuDao(root)

    dataloader = FileSegmentsDataloader(wudao_dataset)