import spacy

from tqdm import tqdm
import codecs as cs
import os 
from os.path import join as pjoin

import json 

type_dict = {
    "whitechair": "chair",
    "woodchair": "chair", 
    # "white": "chair", # For the sequences with inconsistet names. 
    "smalltable": "table",
    "largetable": "table",
    # "bugtable": "table", # For the sequences with typo. 
    "smallbox": "box",
    "largebox": "box",
    "plasticbox": "box", 
    "suitcase": "suitcase", 
    "trashcan": "trashcan", 
    "monitor": "monitor",
    "floorlamp": "floorlamp",
    "tripod": "tripod",
    "clothesstand": "tripod",
}

nlp = spacy.load('en_core_web_sm')
def process_text(sentence):
    sentence = sentence.replace('-', '')
    doc = nlp(sentence)
    word_list = []
    pos_list = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
            word_list.append(token.lemma_)
        else:
            word_list.append(word)
        pos_list.append(token.pos_)
    return word_list, pos_list

def convert_object_name_to_type_name(caption):
    for object_name in type_dict:
        if object_name in caption:
            type_name = type_dict[object_name]
            new_caption = caption.replace(object_name, type_name)

            break 

    return new_caption 

def process_omomo(ori_json_folder, dest_txt_folder): 
    if not os.path.exists(dest_txt_folder):
        os.makedirs(dest_txt_folder)

    json_files = os.listdir(ori_json_folder)
    for json_name in json_files:
        json_path = os.path.join(ori_json_folder, json_name)

        json_data = json.load(open(json_path, 'r'))

        seq_name = json_name.replace(".json", "")
        caption = json_data[seq_name] 

        # Convert object name to type name since there's no differentce in human motion that "lift chair wood" and "lift chair white"
        new_caption = convert_object_name_to_type_name(caption)

        word_list, pose_list = process_text(new_caption)
        start = 0.0
        end = 0.0
        tokens = ' '.join(['%s/%s' % (word_list[i], pose_list[i]) for i in range(len(word_list))])
        text_save_path = os.path.join(dest_txt_folder, json_name.replace(".json", ".txt"))
        with cs.open(text_save_path, 'a+') as f:
            f.write('%s#%s#%s#%s\n' % (new_caption, tokens, start, end))

if __name__ == "__main__":
    data_root_folder = "/move/u/jiamanli/datasets/semantic_manip/processed_data"
    ori_json_folder = os.path.join(data_root_folder, "omomo_text_anno_json_data")
    dest_txt_folder = os.path.join(data_root_folder, "omomo_text_anno_txt_data") 
    process_omomo(ori_json_folder, dest_txt_folder)
