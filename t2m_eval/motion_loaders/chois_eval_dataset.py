import sys
sys.path.append("/viscam/u/jiamanli/github/scene_aware_manip")
sys.path.append("../../")

import os
import numpy as np
import joblib 
import trimesh  
import json 

import random 

import codecs as cs

import torch
from torch.utils.data import Dataset

import pytorch3d.transforms as transforms 

from human_body_prior.body_model.body_model import BodyModel

from torch.utils.data._utils.collate import default_collate

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)


class CHOISEvaluationDataset(Dataset):
    def __init__(
        self,
        res_npz_folder,
        word_vectorizer=None, 
    ):
        self.res_npz_folder = res_npz_folder 
        self.window_data_dict = self.load_res_npz_files() 

        self.w_vectorizer = word_vectorizer 

        data_root_folder = "/move/u/jiamanli/datasets/semantic_manip/processed_data"
        self.language_anno_folder = os.path.join(data_root_folder, "omomo_text_anno_txt_data") 
     
        # For jpos mean, std
        mean_std_jpos_path = os.path.join(data_root_folder, "t2m_mean_std_jpos.p")
        if os.path.exists(mean_std_jpos_path):
            mean_std_dict = joblib.load(mean_std_jpos_path)
            jpos_mean, jpos_std = mean_std_dict['jpos_mean'], mean_std_dict['jpos_std'] 
        
        self.mean_jpos = torch.from_numpy(jpos_mean).float() # 72  
        self.std_jpos = torch.from_numpy(jpos_std).float() # 72 

    def load_res_npz_files(self):
        window_data_dict = {} 
        cnt = 0

        npz_files = os.listdir(self.res_npz_folder)
        for npz_name in npz_files:
            npz_path = os.path.join(self.res_npz_folder, npz_name)

            npz_data = np.load(npz_path)

            window_data_dict[cnt] = {} 
            window_data_dict[cnt]['global_jpos'] = npz_data['global_jpos'] # T X 24 X 3 
            window_data_dict[cnt]['seq_name'] = str(npz_data['seq_name'])

            cnt += 1 

        return window_data_dict 

    def load_language_annotation(self, seq_name):
        # seq_name: sub16_clothesstand_000, etc. 
        # json_path = os.path.join(self.language_anno_folder, seq_name+".json")
        # json_data = json.load(open(json_path, 'r'))
        
        # text_anno = json_data[seq_name]

        # Load .txt file 
        txt_path = os.path.join(self.language_anno_folder, seq_name+".txt")
        with cs.open(txt_path) as f:
            for line in f.readlines():
                text_dict = {}
                line_split = line.strip().split('#')
                caption = line_split[0]
                tokens = line_split[1].split(' ')
               
                text_dict['caption'] = caption
                text_dict['tokens'] = tokens
                
        f.close() 

        return text_dict 

    def __len__(self):
        return len(self.window_data_dict)
    
    def normalize_jpos_mean_std(self, ori_jpos):
        # ori_jpos: T X 72/BS X T X 72
        if ori_jpos.dim() == 3: # BS X T X 72 
            norm_jpos = (ori_jpos - self.mean_jpos[None, None, :])/self.std_jpos[None, None, :] 
        else: # T X 72 
            norm_jpos = (ori_jpos - self.mean_jpos[None, :])/self.std_jpos[None, :] 

        return norm_jpos 
    
    def de_normalize_jpos_mean_std(self, norm_jpos):
        # norma_jpos: T X 72/BS X T X 72 
        if norm_jpos.dim() == 3:
            ori_jpos = norm_jpos * self.std_jpos[None, None, :] + self.mean_jpos[None, None, :]
        else:
            ori_jpos = norm_jpos * self.std_jpos[None, :] + self.mean_jpos[None, :]

        return ori_jpos 

    def __getitem__(self, index):  
        ori_jpos = self.window_data_dict[index]['global_jpos'].reshape(-1, 24*3) # T X 72 
        ori_jpos = torch.from_numpy(ori_jpos).float() # T X 72 
        data_input = self.normalize_jpos_mean_std(ori_jpos) # T X 72 

        seq_name = self.window_data_dict[index]['seq_name'] 
      
        new_data_input = data_input 

        # Add padding. 
        actual_steps = new_data_input.shape[0]
        # if actual_steps < self.window:
        #     paded_new_data_input = torch.cat((new_data_input, torch.zeros(self.window-actual_steps, new_data_input.shape[-1])), dim=0)
        #     # paded_ori_data_input = torch.cat((ori_data_input, torch.zeros(self.window-actual_steps, ori_data_input.shape[-1])), dim=0)  
        # else:
        paded_new_data_input = new_data_input 
            # paded_ori_data_input = ori_data_input 
        
        # Load language annotation 
        text_data = self.load_language_annotation(seq_name) 
        caption, tokens = text_data['caption'], text_data['tokens']

        max_text_len = 30 
        if len(tokens) < max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, \
            paded_new_data_input, actual_steps, '_'.join(tokens)

        # data_input_dict['motion']: T X (22*3+22*6) range [-1, 1]
        # data_input_dict['obj_bps]: T X N X 3 
