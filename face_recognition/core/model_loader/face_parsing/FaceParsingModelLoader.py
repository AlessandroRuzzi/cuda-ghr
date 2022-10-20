"""
@author: fengyu, wangjun
@date: 20220620
@contact: fengyu_cnyc@163.com
"""


import torch

from face_recognition.core.model_loader.BaseModelLoader import BaseModelLoader

class FaceParsingModelLoader(BaseModelLoader):
    def __init__(self, model_path, model_category, model_name, meta_file='model_meta.json'):
        super().__init__(model_path, model_category, model_name, meta_file)

        self.cfg['input_height'] = self.meta_conf['input_height']
        self.cfg['input_width'] = self.meta_conf['input_width']

        
    def load_model(self):
        try:
            model = torch.jit.load(self.cfg['model_file_path'],map_location=torch.device("cpu"))
        except Exception as e:
            raise e
        else:
            return model, self.cfg