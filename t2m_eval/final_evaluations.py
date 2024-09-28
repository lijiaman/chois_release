from datetime import datetime
import numpy as np
import torch
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from motion_loaders.model_motion_loaders import get_motion_loader, get_motion_loader_for_chois_eval 
from utils.get_opt import get_opt
from utils.metrics import *
from networks.evaluator_wrapper import EvaluatorModelWrapper
from collections import OrderedDict
from utils.plot_script import *
# from scripts.motion_process import *
from utils import paramUtil
from utils.utils import *

from options.train_options import TrainTexMotMatchOptions

from os.path import join as pjoin

from vis_skeleton_motion import show3Dpose_animation 
from data.omomo_dataset import CanoObjectTrajDataset 
from utils.word_vectorizer import WordVectorizer, POS_enumerator

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

# def plot_t2m(data, save_dir, captions):
#     data = gt_dataset.inv_transform(data)
#     # print(ep_curves.shape)
#     for i, (caption, joint_data) in enumerate(zip(captions, data)):
#         joint = recover_from_ric(torch.from_numpy(joint_data).float(), wrapper_opt.joints_num).numpy()
#         save_path = pjoin(save_dir, '%02d.mp4'%(i))
#         plot_3d_motion(save_path, paramUtil.t2m_kinematic_chain, joint, title=caption, fps=20)
#         # print(ep_curve.shape)

torch.multiprocessing.set_sharing_strategy('file_system')

def evaluate_matching_score(motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    # print(motion_loaders.keys())
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            _, _, _, sent_lens, motions, m_lens, _ = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict

def get_metric_statistics(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    replication_times = 1 
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(log_file):
    with open(log_file, 'w') as f:
        # all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
        #                            'R_precision': OrderedDict({}),
        #                            'FID': OrderedDict({}),
        #                            'Diversity': OrderedDict({}),
        #                            'MultiModality': OrderedDict({})})
        
        all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   })
        replication_times = 1 
        for replication in range(replication_times):
            motion_loaders = {}
            # mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                # mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(gt_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f)

            # print(f'Time: {datetime.now()}')
            # print(f'Time: {datetime.now()}', file=f, flush=True)
            # mm_score_dict = evaluate_multimodality(mm_motion_loaders, f)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics['Matching Score']:
                    all_metrics['Matching Score'][key] = [item]
                else:
                    all_metrics['Matching Score'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]

            # for key, item in mm_score_dict.items():
            #     if key not in all_metrics['MultiModality']:
            #         all_metrics['MultiModality'][key] = [item]
            #     else:
            #         all_metrics['MultiModality'][key] += [item]


        # print(all_metrics['Diversity'])
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)

            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values))
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)

def check_vis(save_dir):
    w_vectorizer = WordVectorizer('/move/u/jiamanli/github/text-to-motion/glove_840B', 'our_vab')

    data_root_folder = "/move/u/jiamanli/datasets/semantic_manip/processed_data"
    val_dataset = CanoObjectTrajDataset(train=False, data_root_folder=data_root_folder, \
                word_vectorizer=w_vectorizer) 
    
    motion_loaders = {}
    for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
        motion_loader = motion_loader_getter()
        motion_loaders[motion_loader_name] = motion_loader
       
    motion_loaders['ground_truth'] = gt_loader
    for motion_loader_name, motion_loader in motion_loaders.items():
        for idx, batch in enumerate(motion_loader):
            if not (idx % 4 == 0):
                continue 

            word_embeddings, pos_one_hots, captions, sent_lens, motions, m_lens, tokens = batch
            motions = motions[:, :m_lens[0]] # BS X T X 72 
            # plot_t2m(motions.cpu().numpy(), save_path, captions)
            print('-----%d-----'%idx)
            print(captions)
            print(tokens)
            print(sent_lens)
            print(m_lens)

            ani_save_path = pjoin(save_dir, 'animation', '%02d'%(idx))
            os.makedirs(ani_save_path, exist_ok=True)
           
            # data = gt_dataset.inv_transform(motions[0])
            # print(ep_curves.shape)
            
            # save_path = pjoin(save_dir, '%02d.mp4' % (idx))
            plot_t2m(motions[:8, None], pjoin(ani_save_path, '%s' % (motion_loader_name)),
                          val_dataset)


if __name__ == '__main__':
    # dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'
    # dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD01/opt.txt'
    eval_motion_loaders = {
        'CHOIS w classifier guidance': lambda: get_motion_loader_for_chois_eval(
            '/move/u/jiamanli/eccv24_chois/res_npz_files_rebuttal/chois_perturb_mean',
            batch_size,
        ), 

        'CHOIS w recon x0': lambda: get_motion_loader_for_chois_eval(
            '/move/u/jiamanli/eccv24_chois/res_npz_files_rebuttal/chois_recon_guide_w_x0',
            batch_size,
        ), 

        'CHOIS': lambda: get_motion_loader_for_chois_eval(
            '/move/u/jiamanli/eccv24_chois/res_npz_files/chois',
            batch_size,
        ), 

        # 'CHOIS w/o L_geo': lambda: get_motion_loader_for_chois_eval(
        #     '/move/u/jiamanli/eccv24_chois/res_npz_files/chois_wo_l_geo',
        #     batch_size,
        # ), 

        # 'CHOIS w/o guidance': lambda: get_motion_loader_for_chois_eval(
        #     '/move/u/jiamanli/eccv24_chois/res_npz_files/chois_wo_guidance',
        #     batch_size,
        # ), 

        # 'CHOIS w contact guidance only': lambda: get_motion_loader_for_chois_eval(
        #     '/move/u/jiamanli/eccv24_chois/res_npz_files/chois_contact_guidance_only',
        #     batch_size,
        # ), 

        # 'CHOIS w feet-floor guidance only': lambda: get_motion_loader_for_chois_eval(
        #     '/move/u/jiamanli/eccv24_chois/res_npz_files/chois_feetfloor_guidance_only',
        #     batch_size,
        # ), 

        # 'InterDiff': lambda: get_motion_loader_for_chois_eval(
        #     '/move/u/jiamanli/eccv24_chois/res_npz_files/interdiff',
        #     batch_size,
        # ), 

        # 'MDM': lambda: get_motion_loader_for_chois_eval(
        #     '/move/u/jiamanli/eccv24_chois/res_npz_files/mdm',
        #     batch_size,
        # ), 

        # 'Lin-OMOMO': lambda: get_motion_loader_for_chois_eval(
        #     '/move/u/jiamanli/eccv24_chois/res_npz_files/omomo',
        #     batch_size,
        # ), 

        # 'Pred-OMOMO': lambda: get_motion_loader_for_chois_eval(
        #     '/move/u/jiamanli/eccv24_chois/res_npz_files/omomo_pred_obj_input',
        #     batch_size,
        # ), 

        # 'GT-OMOMO': lambda: get_motion_loader_for_chois_eval(
        #     '/move/u/jiamanli/eccv24_chois/res_npz_files/omomo_gt_obj_input',
        #     batch_size,
        # ), 
      
    }

    # eval_motion_loaders = {
    #     'CHOIS': lambda: get_motion_loader_for_chois_eval(
    #         '/move/u/jiamanli/eccv24_chois/res_npz_files/chois_unseen_obj',
    #         batch_size,
    #     ), 

    #     'CHOIS w/o L_geo': lambda: get_motion_loader_for_chois_eval(
    #         '/move/u/jiamanli/eccv24_chois/res_npz_files/chois_wo_l_geo_unseen_obj',
    #         batch_size,
    #     ), 

    #     'CHOIS w/o guidance': lambda: get_motion_loader_for_chois_eval(
    #         '/move/u/jiamanli/eccv24_chois/res_npz_files/chois_wo_guidance_unseen_obj',
    #         batch_size,
    #     ), 

    #     'InterDiff': lambda: get_motion_loader_for_chois_eval(
    #         '/move/u/jiamanli/eccv24_chois/res_npz_files/interdiff_unseen_obj',
    #         batch_size,
    #     ), 

    #     'MDM': lambda: get_motion_loader_for_chois_eval(
    #         '/move/u/jiamanli/eccv24_chois/res_npz_files/mdm_unseen_obj',
    #         batch_size,
    #     ), 

    #     'Lin-OMOMO': lambda: get_motion_loader_for_chois_eval(
    #         '/move/u/jiamanli/eccv24_chois/res_npz_files/omomo_unseen_obj',
    #         batch_size,
    #     ), 

    #     'Pred-OMOMO': lambda: get_motion_loader_for_chois_eval(
    #         '/move/u/jiamanli/eccv24_chois/res_npz_files/omomo_pred_obj_input_unseen_obj',
    #         batch_size,
    #     ), 
       
    # }

    batch_size = 32
    diversity_times = 300 

    gt_loader = get_motion_loader_for_chois_eval(
            '/move/u/jiamanli/eccv24_chois/res_npz_files/gt',
            batch_size)
    # gt_loader, gt_dataset = get_dataset_motion_loader(dataset_opt_path, batch_size, device)
    
    parser = TrainTexMotMatchOptions()
    wrapper_opt = parser.parse()

    device_id = 0
    wrapper_opt.device = torch.device('cuda:%d'%device_id if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device_id)

    # wrapper_opt = get_opt(dataset_opt_path, device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    log_file = './t2m_evaluation_chois_w_classifier_guidance.log'
    evaluation(log_file)

    # save_vis_folder = "/move/u/jiamanli/eccv2024_chois/check_fid_eval_res"
    # check_vis(save_vis_folder)
   