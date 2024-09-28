import os

from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.train_options import TrainDecompOptions
from utils.plot_script import *

from networks.modules import *
from networks.trainers import DecompTrainerV3
# from data.dataset import MotionDatasetV2
from data.omomo_dataset import CanoObjectTrajDataset 
# from scripts.motion_process import *
from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizer, POS_enumerator
from vis_skeleton_motion import show3Dpose_animation 

def plot_t2m(data, save_dir, ds):
    # data: BS X 2 X T X D 
    num_steps = data.shape[2]
   
    # data = train_dataset.inv_transform(data)
    for i in range(len(data)):
        # joint_data = data[i][:, :, :24*3].reshape(-1, num_steps, 24, 3) # 2 X T X 24 X 3
        # joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        # joint = ds.de_normalize_jpos_min_max(joint_data) # 2 X T X 24 X 3 
        joint_data = data[i][:, :, :24*3] # 2 X T X 72 
        joint = ds.de_normalize_jpos_mean_std(joint_data) # 2 X T X 72 
        joint = joint.reshape(-1, num_steps, 24, 3) # 2 X T X 24 X 3 
        save_path = pjoin(save_dir, '%02d.mp4'%(i))
        # plot_3d_motion(save_path, kinematic_chain, joint, title="None", fps=fps, radius=radius)
        show3Dpose_animation(joint.detach().cpu().numpy(), ds.parents, save_path) 

if __name__ == '__main__':
    parser = TrainDecompOptions()
    opt = parser.parse()

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        # self.opt.gpu_id = int(self.opt.gpu_id)
        torch.cuda.set_device(opt.gpu_id)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    # For OMOMO dataset 
    dim_pose = 24*3
    meta_root = pjoin(opt.checkpoints_dir, opt.dataset_name, 'For_CHOIS_Eval_Motion_AE', 'meta')

    dim_word = 300
    dim_pos_ohot = len(POS_enumerator)

    w_vectorizer = WordVectorizer('/move/u/jiamanli/github/text-to-motion/glove_840B', 'our_vab')

    # Define models 
    movement_enc = MovementConvEncoder(dim_pose, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, dim_pose)

    all_params = 0
    pc_mov_enc = sum(param.numel() for param in movement_enc.parameters())
    print(movement_enc)
    print("Total parameters of prior net: {}".format(pc_mov_enc))
    all_params += pc_mov_enc

    pc_mov_dec = sum(param.numel() for param in movement_dec.parameters())
    print(movement_dec)
    print("Total parameters of posterior net: {}".format(pc_mov_dec))
    all_params += pc_mov_dec

    trainer = DecompTrainerV3(opt, movement_enc, movement_dec)

    # Deffine dataset 
    data_root_folder = "/move/u/jiamanli/datasets/semantic_manip/processed_data"
    train_dataset = CanoObjectTrajDataset(train=True, data_root_folder=data_root_folder, \
                word_vectorizer=w_vectorizer)
    val_dataset = CanoObjectTrajDataset(train=False, data_root_folder=data_root_folder, \
                word_vectorizer=w_vectorizer) 

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                            shuffle=True, pin_memory=True)

    trainer.train(train_loader, val_loader, plot_t2m, train_dataset) 
