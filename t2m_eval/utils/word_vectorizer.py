import numpy as np
import pickle
from os.path import join as pjoin
import os 
import io 
import pickle as pkl 
import array 

POS_enumerator = {
    'VERB': 0,
    'NOUN': 1,
    'DET': 2,
    'ADP': 3,
    'NUM': 4,
    'AUX': 5,
    'PRON': 6,
    'ADJ': 7,
    'ADV': 8,
    'Loc_VIP': 9,
    'Body_VIP': 10,
    'Obj_VIP': 11,
    'Act_VIP': 12,
    'Desc_VIP': 13,
    'OTHER': 14,
}

Loc_list = ('left', 'right', 'clockwise', 'counterclockwise', 'anticlockwise', \
    'forward', 'back', 'backward',
            'up', 'down', 'straight', 'curve')

Body_list = ('arm', 'chin', 'foot', 'feet', 'face', 'hand', 'mouth', 'leg', 'waist', \
    'eye', 'knee', 'shoulder', 'thigh')

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

Obj_List = ('stair', 'dumbbell', 'chair', 'window', 'floor', 'car', \
    'ball', 'handrail', 'baseball', 'basketball', \
    'table', 'box', 'suitcase', 'trashcan', 'monitor', 'floorlamp', 'tripod', 'clothesstand') # Added by jiaman

Act_list = ('walk', 'run', 'swing', 'pick', 'bring', 'kick', 'put', 'squat', 'throw', 'hop', \
    'dance', 'jump', 'turn', 'stumble', 'dance', 'stop', 'sit', 'lift', 'lower', 'raise', \
    'wash', 'stand', 'kneel', 'stroll', 'rub', 'bend', 'balance', 'flap', 'jog', 'shuffle', \
    'lean', 'rotate', 'spin', 'spread', 'climb', \
    'move', 'place', 'slide', 'push', 'pull', 'release', 'drag', 'scoot', 'grab', 'hold', 'tilt', 'flip', 'grasp') # Added by jiaman

Desc_list = ('slowly', 'carefully', 'fast', 'careful', 'slow', 'quickly', 'happy', 'angry', 'sad', 'happily',
             'angrily', 'sadly')

VIP_dict = {
    'Loc_VIP': Loc_list,
    'Body_VIP': Body_list,
    'Obj_VIP': Obj_List,
    'Act_VIP': Act_list,
    'Desc_VIP': Desc_list,
}


class WordVectorizer(object):
    def __init__(self, meta_root, prefix):
        vectors = np.load(pjoin(meta_root, '%s_data.npy'%prefix))
        words = pickle.load(open(pjoin(meta_root, '%s_words.pkl'%prefix), 'rb'))
        word2idx = pickle.load(open(pjoin(meta_root, '%s_idx.pkl'%prefix), 'rb'))
        self.word2vec = {w: vectors[word2idx[w]] for w in words}

    def _get_pos_ohot(self, pos):
        pos_vec = np.zeros(len(POS_enumerator))
        if pos in POS_enumerator:
            pos_vec[POS_enumerator[pos]] = 1
        else:
            pos_vec[POS_enumerator['OTHER']] = 1
        return pos_vec

    def __len__(self):
        return len(self.word2vec)

    def __getitem__(self, item):
        word, pos = item.split('/')
        if word in self.word2vec:
            word_vec = self.word2vec[word]
            vip_pos = None
            for key, values in VIP_dict.items():
                if word in values:
                    vip_pos = key
                    break
            if vip_pos is not None:
                pos_vec = self._get_pos_ohot(vip_pos)
            else:
                pos_vec = self._get_pos_ohot(pos)
        else: # Should avoid this! 
            word_vec = self.word2vec['unk']
            pos_vec = self._get_pos_ohot('OTHER')
        return word_vec, pos_vec

def convert_glove_vec_to_humanml_rep():
    data_folder = "/move/u/jiamanli/github/text-to-motion/glove_840B"

    glove_vec_path = os.path.join(data_folder, "glove.840B.300d.txt")

    dest_vec_npy_path = os.path.join(data_folder, "our_vab_data.npy")
    dest_word2idx_pkl_path = os.path.join(data_folder, "our_vab_idx.pkl")
    dest_word_pkl_path = os.path.join(data_folder, "our_vab_words.pkl")

    dct = {}
    word_list = []
    vectors = []

    # Read in the data.
    with io.open(glove_vec_path, 'r', encoding='utf-8') as savefile:
        for i, line in enumerate(savefile):
            tokens = line.split(' ')

            word = tokens[0]
            entries = tokens[1:]

            word_list.append(word)

            dct[word] = i
            curr_vector = []
            curr_vector.extend(float(x) for x in entries)
            vectors.append(curr_vector)

    pkl.dump(dct, open(dest_word2idx_pkl_path, 'wb'))
    pkl.dump(word_list, open(dest_word_pkl_path, 'wb'))

    vectors = np.asarray(vectors)
    np.save(dest_vec_npy_path, vectors) 

if __name__ == "__main__":
   convert_glove_vec_to_humanml_rep()

