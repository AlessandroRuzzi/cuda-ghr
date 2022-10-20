"""
@author: JiXuan Xu, Jun Wang
@date: 20201023
@contact: jun21wangustc@gmail.com 
"""

import torch

from face_recognition.core.model_loader.BaseModelLoader import BaseModelLoader

class FaceAlignModelLoader(BaseModelLoader):
    def __init__(self, model_path, model_category, model_name, meta_file='model_meta.json'):
        super().__init__(model_path, model_category, model_name, meta_file)
        self.cfg['img_size'] = self.meta_conf['input_width']
        
    def load_model(self):
        try:
            model = torch.load(self.cfg['model_file_path'],map_location=torch.device("cpu"))
        except Exception as e:
            raise e
        else:
            return model, self.cfg
