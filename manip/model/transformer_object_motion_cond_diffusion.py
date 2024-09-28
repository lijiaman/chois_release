import math 
import numpy as np

import os 
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from einops import rearrange, reduce

from inspect import isfunction

import torch
from torch import nn
import torch.nn.functional as F

import pytorch3d.transforms as transforms 

from manip.data.cano_traj_dataset import quat_fk_torch, quat_ik_torch 

from manip.model.transformer_module import Decoder 
from manip.lafan1.utils import rotate_at_frame_w_obj_global, rotate_at_frame_w_obj, quat_slerp 

import time as PyTime 

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def wxyz_to_xyzw(input_quat):
    # input_quat: 1 X w X 22 X 4 
    w = input_quat[:, :, :, 0:1]
    x = input_quat[:, :, :, 1:2]
    y = input_quat[:, :, :, 2:3]
    z = input_quat[:, :, :, 3:4]

    out_quat = torch.cat((x, y, z, w), dim=-1) # 1 X w X 22 X 4 

    return out_quat 

def xyzw_to_wxyz(input_quat):
    # input_quat: 1 X w X 22 X 4 
    x = input_quat[:, :, :, 0:1]
    y = input_quat[:, :, :, 1:2]
    z = input_quat[:, :, :, 2:3]
    w = input_quat[:, :, :, 3:4]

    out_quat = torch.cat((w, x, y, z), dim=-1) # 1 X w X 22 X 4 

    return out_quat 

def interpolate_transition(prev_obj_com_pos, prev_obj_rot_mat, prev_jpos, prev_rot_6d, window_obj_com_pos, \
                        window_obj_rot_mat, window_jpos, window_rot_6d):
    # prev_obj_com_pos: 1 X overlap_num X 3
    # prev_obj_q=rot_mat: 1 X overlap_num X 3 X 3
    # prev_jpos: 1 X overlap_num X 24 X 3 
    # prev_rot_6d: 1 X overlap_num X 22 X 6  
    # window_obj_com_pos: 1 X w X 3
    # window_obj_rot_mat: 1 X w X 3 X 3 
    # window_jpos: 1 X w X 24 X 3 
    # window_rot_6d: 1 X w X 22 X 6
    num_overlap_frames = prev_jpos.shape[1]

    fade_out = torch.ones((1, num_overlap_frames, 1)).to(prev_jpos.device) # 1 X overlap_num X 1 
    fade_in = torch.ones((1, num_overlap_frames, 1)).to(prev_jpos.device)
    fade_out = torch.linspace(1, 0, num_overlap_frames)[None, :, None].to(prev_jpos.device) # 1 X overlap_num X 1 
    fade_in = torch.linspace(0, 1, num_overlap_frames)[None, :, None].to(prev_jpos.device)

    window_obj_com_pos[:, :num_overlap_frames, :] = fade_out * prev_obj_com_pos + \
        fade_in * window_obj_com_pos[:, :num_overlap_frames, :]  
    window_jpos[:, :num_overlap_frames, :, :] = fade_out[:, :, None, :] * prev_jpos + \
        fade_in[:, :, None, :] * window_jpos[:, :num_overlap_frames, :, :]

    # stitch joint angles with slerp
    slerp_weight = torch.linspace(0, 1, num_overlap_frames)[None, :, None].to(prev_rot_6d.device) # 1 X overlap_num X 1 
   
    prev_obj_q = transforms.matrix_to_quaternion(prev_obj_rot_mat)
    window_obj_q = transforms.matrix_to_quaternion(window_obj_rot_mat) # 1 X w X 4 

    prev_rot_mat = transforms.rotation_6d_to_matrix(prev_rot_6d)
    prev_q = transforms.matrix_to_quaternion(prev_rot_mat)
    window_rot_mat = transforms.rotation_6d_to_matrix(window_rot_6d) 
    window_q = transforms.matrix_to_quaternion(window_rot_mat) # 1 X w X 22 X 4 

    obj_q_left = prev_obj_q[:, :, None, :] # 1 X overlap_num X 1 X 4, w, x, y, z, real part first in pytorch3d. 
    obj_q_right = window_obj_q[:, :num_overlap_frames, None, :] # 1 X overlap_num X 1 X 4 

    human_q_left = prev_q.clone() # 1 X overlap_num X 22 X 4 
    human_q_right = window_q[:, :num_overlap_frames, :, :] # 1 X overlap_num X 22 X 4 

    obj_q_left = wxyz_to_xyzw(obj_q_left)
    obj_q_right = wxyz_to_xyzw(obj_q_right)
    human_q_left = wxyz_to_xyzw(human_q_left)
    human_q_right = wxyz_to_xyzw(human_q_right)

    # quat_slerp needs xyzw 
    slerped_obj_q = quat_slerp(obj_q_left, obj_q_right, slerp_weight)  # 1 X overlap_num X 1 X 4 
    slerped_human_q = quat_slerp(human_q_left, human_q_right, slerp_weight) # 1 X overlap_num X 22 X 4 

    slerped_obj_q = xyzw_to_wxyz(slerped_obj_q)
    slerped_human_q = xyzw_to_wxyz(slerped_human_q)

    new_obj_q = torch.cat((slerped_obj_q.squeeze(2), window_obj_q[:, num_overlap_frames:, :]), dim=1) # 1 X w X 4 
    new_human_q = torch.cat((slerped_human_q, window_q[:, num_overlap_frames:, :, :]), dim=1) # 1 X w X 22 X 4 

    new_obj_rot_mat = transforms.quaternion_to_matrix(new_obj_q)
    new_human_rot_mat = transforms.quaternion_to_matrix(new_human_q)
    new_human_rot_6d = transforms.matrix_to_rotation_6d(new_human_rot_mat)

    return window_obj_com_pos, new_obj_rot_mat, window_jpos, new_human_rot_6d 


# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered
        
class TransformerDiffusionModel(nn.Module):
    def __init__(
        self,
        d_input_feats,
        d_feats,
        d_model,
        n_dec_layers,
        n_head,
        d_k,
        d_v,
        max_timesteps,
    ):
        super().__init__()
        
        self.d_feats = d_feats 
        self.d_model = d_model
        self.n_head = n_head
        self.n_dec_layers = n_dec_layers
        self.d_k = d_k 
        self.d_v = d_v 
        self.max_timesteps = max_timesteps 

        # Input: BS X D X T 
        # Output: BS X T X D'
        self.motion_transformer = Decoder(d_feats=d_input_feats, d_model=self.d_model, \
            n_layers=self.n_dec_layers, n_head=self.n_head, d_k=self.d_k, d_v=self.d_v, \
            max_timesteps=self.max_timesteps, use_full_attention=True)  

        self.linear_out = nn.Linear(self.d_model, self.d_feats)

        # For noise level t embedding
        dim = 64
        learned_sinusoidal_dim = 16
        time_dim = dim * 4

        learned_sinusoidal_cond = False
        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, d_model)
        )

    def forward(self, src, noise_t, condition, language_embedding=None, padding_mask=None):
        # src: BS X T X D
        # noise_t: int 

        src = torch.cat((src, condition), dim=-1)
       
        noise_t_embed = self.time_mlp(noise_t) # BS X d_model 
        if language_embedding is not None:
            noise_t_embed += language_embedding # BS X d_model 
        noise_t_embed = noise_t_embed[:, None, :] # BS X 1 X d_model 

        bs = src.shape[0]
        num_steps = src.shape[1] + 1

        if padding_mask is None:
            padding_mask = torch.ones(bs, 1, num_steps).to(src.device).bool() # BS X 1 X timesteps

        # Get position vec for position-wise embedding
        pos_vec = torch.arange(num_steps)+1 # timesteps
        pos_vec = pos_vec[None, None, :].to(src.device).repeat(bs, 1, 1) # BS X 1 X timesteps

        data_input = src.transpose(1, 2) # BS X D X T 
        feat_pred, _ = self.motion_transformer(data_input, padding_mask, pos_vec, obj_embedding=noise_t_embed)
       
        output = self.linear_out(feat_pred[:, 1:]) # BS X T X D

        return output # predicted noise, the same size as the input 

    
class ObjectCondGaussianDiffusion(nn.Module):
    def __init__(
        self,
        opt,
        d_feats,
        d_model,
        n_head,
        n_dec_layers,
        d_k,
        d_v,
        max_timesteps,
        out_dim,
        timesteps = 1000,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        input_first_human_pose=False, 
        use_object_keypoints=False, 
    ):
        super().__init__()

        self.bps_encoder = nn.Sequential(
            nn.Linear(in_features=1024*3, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            )

        self.clip_encoder = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            )

        self.input_first_human_pose = input_first_human_pose 
        
        # Input: (BS*T) X 3 X N 
        # Output: (BS*T) X d X N, (BS*T) X d 
        # self.object_encoder = Pointnet() 

        self.use_object_keypoints = use_object_keypoints 

        obj_feats_dim = 256 
        d_input_feats = 2*d_feats+obj_feats_dim

        self.denoise_fn = TransformerDiffusionModel(d_input_feats=d_input_feats, d_feats=d_feats, \
                    d_model=d_model, n_head=n_head, d_k=d_k, d_v=d_v, \
                    n_dec_layers=n_dec_layers, max_timesteps=max_timesteps) 
        # Input condition and noisy motion, noise level t, predict gt motion
        
        self.objective = objective

        self.seq_len = max_timesteps - 1 
        self.out_dim = out_dim 

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        print(alphas_cumprod[0], torch.sqrt(alphas_cumprod)[0], torch.sqrt(1. - alphas_cumprod)[0])
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, x_cond, language_embedding=None, padding_mask=None, clip_denoised=True):
        # x_all = torch.cat((x, x_cond), dim=-1)
        # model_output = self.denoise_fn(x_all, t)

        model_output = self.denoise_fn(x, t, x_cond, language_embedding, padding_mask)

        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t = t, noise = model_output)
        elif self.objective == 'pred_x0':
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    
    def p_mean_variance_reconstruction_guidance(self, x, t, x_cond, guidance_fn, opt_fn=None, \
                                    language_embedding=None, padding_mask=None, \
                                    rest_human_offsets=None, data_dict=None, \
                                    cond_mask=None, \
                                    prev_window_cano_rot_mat=None, \
                                    prev_window_init_root_trans=None, \
                                    contact_labels=None, \
                                    curr_window_ref_obj_rot_mat=None, \
                                    clip_denoised=True):
        # x_all = torch.cat((x, x_cond), dim=-1)
        # model_output = self.denoise_fn(x_all, t)
        with torch.enable_grad():

            # For reconstruction guidance 
            x = x.detach().requires_grad_(True)

            model_output = self.denoise_fn(x, t, x_cond, language_embedding, padding_mask)

            if self.objective == 'pred_noise':
                x_start = self.predict_start_from_noise(x, t = t, noise = model_output)
            elif self.objective == 'pred_x0':
                x_start = model_output
            else:
                raise ValueError(f'unknown objective {self.objective}')

            x_pose_cond = x_cond[:, :, -x.shape[-1]:].detach() 

            classifier_scale = 1e3

            # For classifier guidance 
            # x_start = x_start.detach().requires_grad_(True) 
        
            loss = guidance_fn(t, x_start, x_pose_cond, cond_mask, \
                rest_human_offsets, data_dict, \
                contact_labels=contact_labels, \
                curr_window_ref_obj_rot_mat=curr_window_ref_obj_rot_mat, \
                prev_window_cano_rot_mat=prev_window_cano_rot_mat, \
                prev_window_init_root_trans=prev_window_init_root_trans) # For hand-object interaction loss  
        
            # gradient = torch.autograd.grad(-loss, x)[0] * classifier_scale # BS(1) X 120 X 216 

            gradient = torch.autograd.grad(-loss, x_start)[0] * classifier_scale # BS(1) X 120 X 216 

            # print("gradient mean:{0}".format(gradient.mean()))
            # print("gradient min:{0}".format(gradient.min()))
            # print("gradient max:{0}".format(gradient.max()))
        
            # Peturb predicted clean x 
            tmp_posterior_variance = extract(self.posterior_variance, t, x_start.shape)
            # print("variance max:{0}".format(tmp_posterior_variance.max()))
           
            x_start = x_start + tmp_posterior_variance * gradient.float()

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample_guided_reconstruction_guidance(self, x, t, x_cond, guidance_fn, language_embedding=None, \
                    opt_fn=None, clip_denoised=True, \
                    rest_human_offsets=None, data_dict=None, cond_mask=None, padding_mask=None, \
                    prev_window_cano_rot_mat=None, prev_window_init_root_trans=None, \
                    contact_labels=None, curr_window_ref_obj_rot_mat=None):
        b, *_, device = *x.shape, x.device

        use_reconstruction_guidance = True 

        if use_reconstruction_guidance:
            # x_start = self.q_sample(x_start, t) # Add noise to the target, for debugging. 
            model_mean, _, model_log_variance = self.p_mean_variance_reconstruction_guidance(x=x, t=t, x_cond=x_cond, \
                                        guidance_fn=guidance_fn, language_embedding=language_embedding, opt_fn=opt_fn, \
                                        clip_denoised=clip_denoised, cond_mask=cond_mask, padding_mask=padding_mask, \
                                        rest_human_offsets=rest_human_offsets, data_dict=data_dict, \
                                        prev_window_cano_rot_mat=prev_window_cano_rot_mat, \
                                        prev_window_init_root_trans=prev_window_init_root_trans, \
                                        contact_labels=contact_labels, \
                                        curr_window_ref_obj_rot_mat=curr_window_ref_obj_rot_mat)  
            
            new_mean = model_mean 
        else: # Classifier-guidance 
            model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, x_cond=x_cond, \
                language_embedding=language_embedding, \
                padding_mask=padding_mask, clip_denoised=clip_denoised)

            # classifier_scale = 1e3

            classifier_scale = 10

            # For classifier guidance 
            model_mean = model_mean.detach().requires_grad_(True) 

            loss = guidance_fn(t, model_mean, x_cond, cond_mask, \
                rest_human_offsets, data_dict, \
                contact_labels=contact_labels, \
                curr_window_ref_obj_rot_mat=curr_window_ref_obj_rot_mat, \
                prev_window_cano_rot_mat=prev_window_cano_rot_mat, \
                prev_window_init_root_trans=prev_window_init_root_trans)

            gradient = torch.autograd.grad(-loss, model_mean)[0] * classifier_scale # BS(1) X 120 X 216 

            # Peturb predicted clean x 
            tmp_posterior_variance = extract(self.posterior_variance, t, model_mean.shape)
            print("variance max:{0}".format(tmp_posterior_variance.max()))
           
            new_mean = model_mean + tmp_posterior_variance * gradient.float()

            # new_mean = model_mean + model_variance * gradient.float()
            # new_mean = model_mean + gradient.float()
            # print("model variance:{0}".format(model_variance))
            print("gradient max:{0}".format(gradient.max()))

        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        sampled_x = new_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        return sampled_x 

    def p_sample_loop_guided(self, shape, x_cond, guidance_fn=None, language_embedding=None, opt_fn=None, \
                    rest_human_offsets=None, data_dict=None, contact_labels=None, \
                    cond_mask=None, padding_mask=None):
        device = self.betas.device

        b = shape[0]
        x = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            if guidance_fn is not None and i > 0 and i < 10: 
                x = self.p_sample_guided_reconstruction_guidance(x, torch.full((b,), i, device=device, dtype=torch.long), \
                            x_cond, guidance_fn, language_embedding=language_embedding, \
                            opt_fn=opt_fn, rest_human_offsets=rest_human_offsets, \
                            data_dict=data_dict, contact_labels=contact_labels, \
                            cond_mask=cond_mask, padding_mask=padding_mask)  

                # known_pose_conditions = x_cond[:, :, -x.shape[-1]:]
                # x = (1 - cond_mask) * known_pose_conditions + cond_mask * x 
            else:
                x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), x_cond, \
                    language_embedding, padding_mask=padding_mask)    

        return x # BS X T X D

    @torch.no_grad()
    def p_sample(self, x, t, x_cond, language_embedding=None, padding_mask=None, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, x_cond=x_cond, language_embedding=language_embedding, \
            padding_mask=padding_mask, clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, x_cond, language_embedding=None, padding_mask=None, \
                cond_mask=None, ori_pose_cond=None, return_diff_level_res=False):
        device = self.betas.device

        b = shape[0]
        x = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), x_cond, language_embedding, \
            padding_mask=padding_mask)    

        return x # BS X T X D
    
    # @torch.no_grad()
    def p_sample_loop_sliding_window_w_canonical(self, ds, object_names, trans2joint, \
                                x_start, ori_x_cond, cond_mask, padding_mask, \
                                overlap_frame_num=1, input_waypoints=False, contact_labels=None, language_input=None, \
                                rest_human_offsets=None, data_dict=None, \
                                guidance_fn=None, opt_fn=None):
        # object_names: BS 
        # obj_scales: BS X T
        # trans2joint: BS X 3 
        # first_frame_obj_com2trans: BS X 1 X 3 
        # x_start: BS X T X D(3+9+24*3+22*6) (T can be larger than the window_size), without normalization.  
        # ori_x_cond: BS X 1 X (3+1024*3), the first frame's object BPS + com position. 
        # cond_mask: BS X window_size X D
        # padding_mask: BS X T 
        # contact_labels: BS X T 

        shape = x_start.shape 

        device = self.betas.device

        b = shape[0]
        # assert b == 1
        
        x_all = torch.randn(shape, device=device)

        whole_sample_res = None # BS X T X D (3+9+24*3+22*6)  

        num_steps = shape[1]
        # stride = self.seq_len // 2
        # overlap_frame_num = 1
        stride = self.seq_len - overlap_frame_num 
        window_idx = 0 
        for t_idx in range(0, num_steps, stride):
            if t_idx == 0:
                curr_x = x_all[:, t_idx:t_idx+self.seq_len] # Random noise. 
                curr_x_start = x_start[:, t_idx:t_idx+self.seq_len] # BS X window_szie X D (3+9+24*3+22*6) 
               
                # curr_x_cond = torch.cat((ori_x_cond[:, :, :3], self.bps_encoder(ori_x_cond[:, :, 3:])), dim=-1) # BS X 1 X (3+256) 
                curr_x_cond = self.bps_encoder(ori_x_cond) # BS X 1 X 256 
                curr_x_cond = curr_x_cond.repeat(1, self.seq_len, 1) # BS X T X (3+256) 

                if contact_labels is None:
                    curr_window_contact_labels = None 
                else:
                    curr_window_contact_labels = contact_labels[:, t_idx:t_idx+self.seq_len] 

                if language_input is not None:
                    language_embedding = self.clip_encoder(language_input[window_idx]) # BS X d_model 
                else:
                    language_embedding = None 

                x_pose_cond = curr_x_start * (1. - cond_mask) # Remove noise, overall better than adding random noise. 
                curr_x_cond = torch.cat((curr_x_cond, x_pose_cond), dim=-1) # BS X T X (3+256+3+9) 

                for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                    if guidance_fn is not None and i > 0 and i < 10: 
                        curr_x = self.p_sample_guided_reconstruction_guidance(curr_x, torch.full((b,), i, \
                                    device=device, dtype=torch.long), \
                                    curr_x_cond, language_embedding=language_embedding, \
                                    guidance_fn=guidance_fn, opt_fn=opt_fn, \
                                    rest_human_offsets=rest_human_offsets, \
                                    data_dict=data_dict, cond_mask=cond_mask, \
                                    contact_labels=curr_window_contact_labels)     

                        # known_pose_conditions = curr_x_cond[:, :, -curr_x.shape[-1]:]
                        # curr_x = (1 - cond_mask) * known_pose_conditions + cond_mask * curr_x 
                    
                    else: # padding mask is not used now! 
                        curr_x = self.p_sample(curr_x, torch.full((b,), i, device=device, dtype=torch.long), \
                                curr_x_cond, language_embedding=language_embedding)     
                   
                whole_sample_res = curr_x.clone() # BS X window_size X D (3+9+24*3+22*6)  

                # For debug using GT
                # whole_sample_res = curr_x_start.clone() 
                window_idx += 1 
            else:
                # import pdb 
                # pdb.set_trace() 
                curr_x = x_all[:, t_idx:t_idx+self.seq_len] # Random noise. 
                prev_sample_res = whole_sample_res[:, -overlap_frame_num:, :] # BS X 10 X D 
                curr_x_start_init = x_start[:, t_idx:t_idx+self.seq_len] # BS X window_szie X D (3+9+24*3+22*6) 
                concat_time_frame_idx = overlap_frame_num 
                if contact_labels is not None:
                    curr_window_contact_labels = contact_labels[:, t_idx:t_idx+self.seq_len] 
                
                if curr_x.shape[1] < self.seq_len: # The last window with a smaller size. Better to not use this code. 
                    break 

                # Canonicalize the first human pose in the current window and make the root trans to (0,0).
                global_human_normalized_jpos = prev_sample_res[:, :, 12:12+24*3].reshape(b, -1, 24, 3) # BS X 10 X J(24) X 3
                global_human_jpos = ds.de_normalize_jpos_min_max(global_human_normalized_jpos) # BS X 10 X J X 3 

                global_human_6d = prev_sample_res[:, :, 12+24*3:12+24*3+22*6].reshape(b, -1, 22, 6) # BS X 10 X 22 X 6
                global_human_rot_mat = transforms.rotation_6d_to_matrix(global_human_6d)
                global_human_q = transforms.matrix_to_quaternion(global_human_rot_mat) # BS X 10 X 22 X 4 

                obj_normalized_x = prev_sample_res[:, :, :3] # BS X 10 X 3 
                obj_com_pos = ds.de_normalize_obj_pos_min_max(obj_normalized_x) # BS X 10 X 3 
                
                # rotation wrd first frame's object BPS. 
                obj_rel_rot_mat = prev_sample_res[:, :, 3:3+9].reshape(b, -1, 3, 3) # BS X 10 X 3 X 3  
                
                ref_frame_rot_mat = data_dict['reference_obj_rot_mat'].to(obj_rel_rot_mat.device) # 1 X 1 X 3 X 3 
                obj_rot_mat = ds.rel_rot_to_seq(obj_rel_rot_mat, ref_frame_rot_mat) # wrd rest pose object geometry. 
                # obj_rot_mat = ds.rel_rot_to_seq(obj_rel_rot_mat, x_start[:, 0:1, 3:12].reshape(b, -1, 3, 3)) # Seems wrong!!!! x_start contains the first frame's relative rotation wrt itself, not wrt rest pose. 
                
                obj_q = transforms.matrix_to_quaternion(obj_rot_mat) # BS X 10 X 4 
                # The object rotation here is not wrt rest pose geometry, but the first frame's object rotation. 

                # This code is used for inputting first human pose. 
                if self.input_first_human_pose:
                    new_glob_jpos, new_glob_q, new_obj_com_pos, new_obj_q = \
                    rotate_at_frame_w_obj(global_human_jpos.data.cpu().numpy(), global_human_q.data.cpu().numpy(), \
                    obj_com_pos.data.cpu().numpy(), obj_q.data.cpu().numpy(), \
                    trans2joint.data.cpu().numpy(), ds.parents, n_past=1, floor_z=True, use_global_human=True)
                    # 1 X T X J X 3, 1 X T X J X 4, 1 X T X 3, 1 X T X 4 
                else:
                    # This code is used for not inputting first human pose. 
                    new_glob_jpos, new_glob_q, new_obj_com_pos, new_obj_q = rotate_at_frame_w_obj_global( \
                    obj_com_pos.data.cpu().numpy(), obj_q.data.cpu().numpy(), ds.parents, n_past=1, floor_z=True, \
                    global_q=global_human_q.data.cpu().numpy(), global_x=global_human_jpos.data.cpu().numpy(), use_global=True) 
                    # BS X T X J X 3, BS X T X J X 4, BS X T X 3, BS X T X 4 

                new_glob_jpos = torch.from_numpy(new_glob_jpos).float().to(prev_sample_res.device)
                new_glob_q = torch.from_numpy(new_glob_q).float().to(prev_sample_res.device) 
                new_obj_com_pos = torch.from_numpy(new_obj_com_pos).float().to(prev_sample_res.device)
                new_obj_q = torch.from_numpy(new_obj_q).float().to(prev_sample_res.device) # wrd rest pose's rotation. 

                global_human_root_jpos = new_glob_jpos[:, :, 0, :].clone() # BS X T X 3
                global_human_root_trans = global_human_root_jpos + trans2joint[:, None, :].to(global_human_root_jpos.device) # BS X T X 3 

                move_to_zero_trans = global_human_root_trans[:, 0:1, :].clone() # Move the first frame's root joint x, y to 0,  BS X 1 X 3
                move_to_zero_trans[:, :, 2] = 0 # BS X 1 X 3 

                # move_to_zero_trans = new_obj_com_pos[:, 0:1, :].clone()
                # move_to_zero_trans[:, :, 2] = 0 # BS X 1 X 3 

                global_human_root_trans -= move_to_zero_trans 
                global_human_root_jpos -= move_to_zero_trans 
                new_glob_jpos -= move_to_zero_trans[:, :, None, :] 
                new_obj_com_pos = new_obj_com_pos - move_to_zero_trans # BS X T X 3 

                new_glob_rot_mat = transforms.quaternion_to_matrix(new_glob_q) # BS X T X J X 3 X 3 
                new_glob_rot_6d = transforms.matrix_to_rotation_6d(new_glob_rot_mat) # BS X T X J X 6 

                # Get the rotation matrix to convert current sequence to canonicalization.
                new_obj_rot_mat = transforms.quaternion_to_matrix(new_obj_q) # BS X T X 3 X 3 
                # The relative rotation matrix wrt 1st frame's object after canonicalization. 
                
                # The matrix that convert the orientation to canonicalized direction. 
                cano_rot_mat = torch.matmul(new_glob_rot_mat[:, 0, 0, :, :], \
                            global_human_rot_mat[:, 0, 0, :, :].transpose(1, 2)) # BS X 3 X 3 
               
                # Add original object position information to current window. (in a new canonicalized frame)  
                # This is only for given the target single frame as condition. 
                curr_end_frame_init = curr_x_start_init.clone() # BS X W X D (3+9+24*3+22*6)
                curr_end_obj_com_pos = ds.de_normalize_obj_pos_min_max(curr_end_frame_init[:, :, :3]) # BS X W X 3 
                curr_end_obj_com_pos = torch.matmul(cano_rot_mat[:, None, :, :].repeat(1, \
                                    curr_end_obj_com_pos.shape[1], 1, 1), \
                                    curr_end_obj_com_pos[:, :, :, None]) # BS X W X 3 X 1 
                curr_end_obj_com_pos = curr_end_obj_com_pos.squeeze(-1) # BS X W X 3
                curr_end_obj_com_pos -= move_to_zero_trans # BS X W X 3 

                # Assign target frame's object position infoirmation. 
                curr_end_frame = torch.zeros_like(curr_end_frame_init) # BS X W X D

                # Assign previous window's object information as condition to generate for current window.  
                curr_end_frame[:, :, :3] = ds.normalize_obj_pos_min_max(curr_end_obj_com_pos)

                # gt_obj_normalized_com_pos = x_start[:, 0:1, :3] # BS X 1 X 3  
                # gt_obj_com_pos = ds.de_normalize_obj_pos_min_max(gt_obj_normalized_com_pos)
                # gt_obj_x = ds.com_to_obj_trans(gt_obj_com_pos, first_frame_obj_com2trans) # BS X 1 X 3 

                curr_obj_bps_list = []
                new_obj_com_pos_list = []
                for bs_idx in range(b):
                    # Compute the first frame's objec BPS representation in the current window. 
                    # obj_verts, tmp_obj_faces = ds.load_object_geometry(object_names[bs_idx], obj_scales[bs_idx, t_idx:t_idx+1], \
                    #             new_obj_x[bs_idx], new_obj_rot_mat[bs_idx]) # 10 X Nv X 3, tensor
                    # obj_verts, tmp_obj_faces = ds.load_object_geometry(object_names[bs_idx], obj_scales[bs_idx, 0:1], \
                    #             new_obj_x[bs_idx], new_obj_rot_mat[bs_idx]) # 10 X Nv X 3, tensor
                    # obj_rest_verts, obj_mesh_faces = ds.convert_rest_pose_obj_geometry(object_names[bs_idx], \
                    #     obj_scales[bs_idx, 0:1].cuda(), \
                    #     gt_obj_x[bs_idx].cuda(), obj_rot_mat[bs_idx, 0:1].cuda())

                    obj_rest_verts, obj_mesh_faces = ds.load_rest_pose_object_geometry(object_names[bs_idx])
                    obj_rest_verts = torch.from_numpy(obj_rest_verts).to(new_obj_rot_mat.device)
                    obj_verts = ds.load_object_geometry_w_rest_geo(new_obj_rot_mat[bs_idx], \
                        new_obj_com_pos[bs_idx], obj_rest_verts.float())
                
                    center_verts = obj_verts.mean(dim=1) # 10 X 3 

                    object_bps = ds.compute_object_geo_bps(obj_verts[0:1].cpu(), center_verts[0:1].cpu()) # 1 X 1024 X 3 

                    curr_obj_bps_list.append(object_bps) 
                    new_obj_com_pos_list.append(center_verts) 

                curr_obj_bps = torch.stack(curr_obj_bps_list)[:, None, :, :].cuda() # BS X 1 X 1024 X 3 
                curr_obj_com_pos = torch.stack(new_obj_com_pos_list).cuda() # BS X 10 X 3 

                # curr_x_cond = torch.cat((curr_obj_com_pos[:, 0:1, :], \
                #             self.bps_encoder(curr_obj_bps.reshape(b, 1, -1))), dim=-1) # BS X 1 X (3+256) 
                curr_x_cond = self.bps_encoder(curr_obj_bps.reshape(b, 1, -1)) # BS X 1 X 256 
                curr_x_cond = curr_x_cond.repeat(1, self.seq_len, 1) # BS X T X (3+256) 

                # Prepare canonicalized results of previous overlapped frames. 
                curr_normalized_obj_com_pos = ds.normalize_obj_pos_min_max(curr_obj_com_pos) # BS X 10 X 3 
                curr_normalized_global_jpos = ds.normalize_jpos_min_max(new_glob_jpos) # BS X T X J X 3 
                curr_rel_rot_mat = ds.prep_rel_obj_rot_mat_w_reference_mat(new_obj_rot_mat, new_obj_rot_mat[:, 0:1]) # BS X T X 3 X 3 
               
                cano_prev_sample_res = torch.cat((curr_normalized_obj_com_pos, curr_rel_rot_mat.reshape(b, -1, 9), \
                                curr_normalized_global_jpos.reshape(b, -1, 24*3), new_glob_rot_6d.reshape(b, -1, 22*6)), dim=-1)
                
                if self.use_object_keypoints:
                    cano_prev_sample_res = torch.cat((cano_prev_sample_res, prev_sample_res[:, :, -4:]), dim=-1)

                curr_start_frame = cano_prev_sample_res[:, 0:1].clone() # BS X 1 X D 

                if input_waypoints:
                    curr_x_start = torch.cat((curr_start_frame, curr_end_frame[:, 1:, :]), dim=1) 
                    # import pdb 
                    # pdb.set_trace() 
                else:
                    # Only use the single target frame. 
                    curr_x_start = torch.cat((curr_start_frame, torch.zeros(b, self.seq_len-2, \
                                curr_end_frame.shape[-1]).to(curr_end_frame.device), curr_end_frame[:, -1:, :]), dim=1) 
                
                x_pose_cond = curr_x_start * (1. - cond_mask) # Remove noise, overall better than adding random noise. 
                curr_x_cond = torch.cat((curr_x_cond, x_pose_cond), dim=-1) # BS X T X (3+256+3+9) 

                if language_input is not None:
                    language_embedding = self.clip_encoder(language_input[window_idx]) # BS X d_model 
                else:
                    language_embedding = None 
        
                for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                   
                    # Apply previous window prediction as additional condition, direcly replacement. 

                    if guidance_fn is not None and i > 0 and i < 10: 
                        curr_x = self.p_sample_guided_reconstruction_guidance(curr_x, torch.full((b,), i, device=device, dtype=torch.long), \
                                    curr_x_cond, language_embedding=language_embedding, \
                                    guidance_fn=guidance_fn, opt_fn=opt_fn, \
                                    rest_human_offsets=rest_human_offsets, \
                                    data_dict=data_dict, cond_mask=cond_mask, \
                                    prev_window_cano_rot_mat=cano_rot_mat, \
                                    prev_window_init_root_trans=global_human_jpos[:, 0:1, 0, :], \
                                    contact_labels=curr_window_contact_labels, \
                                    curr_window_ref_obj_rot_mat=new_obj_rot_mat[:, 0:1, :, :])     
                    else:
                        curr_x = self.p_sample(curr_x, torch.full((b,), i, device=device, dtype=torch.long), curr_x_cond, \
                                           language_embedding=language_embedding)    
                  
                    # Overwrite the first few frames with previous results, need to canonicalize previous frames' res. 
                    if i > 0:
                        # Use a paper's method to apply conditions.
                        prev_conditions = torch.cat((cano_prev_sample_res, torch.zeros(b, \
                                        self.seq_len-cano_prev_sample_res.shape[1], cano_prev_sample_res.shape[-1]).to(cano_prev_sample_res.device)), dim=1)
                        # x_w_conditions = self.q_sample(prev_conditions, torch.full((b,), i-1, device=device, dtype=torch.long))
                        x_w_conditions = prev_conditions 
                        prev_condition_mask = torch.ones(b, cano_prev_sample_res.shape[1], cano_prev_sample_res.shape[-1]).to(cano_prev_sample_res.device)
                        prev_condition_mask = torch.cat((prev_condition_mask, \
                                torch.zeros(b, self.seq_len-cano_prev_sample_res.shape[1], cano_prev_sample_res.shape[-1]).to(cano_prev_sample_res.device)), dim=1)

                        curr_x = prev_condition_mask * x_w_conditions + (1 - prev_condition_mask) * curr_x 

                apply_interpolation = True  
                if apply_interpolation:
                    prev_com_pos = cano_prev_sample_res[:, :, :3]
                    prev_obj_rot_mat = cano_prev_sample_res[:, :, 3:3+9].reshape(b, -1, 3, 3)
                    prev_human_jpos = cano_prev_sample_res[:, :, 12:12+24*3].reshape(b, -1, 24, 3)
                    prev_human_rot_6d = cano_prev_sample_res[:, :, 12+24*3:12+24*3+22*6].reshape(b, -1, 22, 6)

                    curr_x_obj_com_pos = curr_x[:, :, :3] # 1 X w X 3 
                    # curr_x_obj_rel_rot_mat = curr_x[:, :, 3:3+9].reshape(b, -1, 3, 3)
                    curr_x_obj_rot_mat = curr_x[:, :, 3:3+9].reshape(b, -1, 3, 3) 
                    curr_x_human_jpos = curr_x[:, :, 12:12+24*3].reshape(b, -1, 24, 3)
                    curr_x_human_rot_6d = curr_x[:, :, 12+24*3:12+24*3+22*6].reshape(b, -1, 22, 6)

                    curr_x_obj_com_pos, curr_x_obj_rot_mat, curr_x_human_jpos, curr_x_human_rot_6d = \
                        interpolate_transition(prev_com_pos, prev_obj_rot_mat, prev_human_jpos, prev_human_rot_6d, \
                                        curr_x_obj_com_pos, curr_x_obj_rot_mat, curr_x_human_jpos, curr_x_human_rot_6d)

                    if self.use_object_keypoints:
                        curr_x = torch.cat((curr_x_obj_com_pos, curr_x_obj_rot_mat.reshape(b, -1, 9), \
                                            curr_x_human_jpos.reshape(b, -1, 24*3), \
                                            curr_x_human_rot_6d.reshape(b, -1, 22*6), \
                                            curr_x[:, :, -4:]), dim=-1)
                    else:
                        curr_x = torch.cat((curr_x_obj_com_pos, curr_x_obj_rot_mat.reshape(b, -1, 9), \
                                            curr_x_human_jpos.reshape(b, -1, 24*3), curr_x_human_rot_6d.reshape(b, -1, 22*6)), dim=-1)

                if self.use_object_keypoints:
                    tmp_curr_x = curr_x[:, :, :-4] 
                else:
                    tmp_curr_x = curr_x.clone() 

                # Convert the results of this window to be at the canonical frame of the first frame in this whole sequence. 
                converted_obj_com_pos, converted_obj_rot_mat, converted_human_jpos, converted_rot_6d = \
                    self.apply_rotation_to_data(ds, trans2joint, cano_rot_mat, new_obj_rot_mat, tmp_curr_x)
                # 1 X window X 3, 1 X window X 3 X 3, 1 X window X 24 X 3, 1 X window X 22 X 6
                # converted_obj_rot_mat is rotation matrix wrt rest pose, need to convert it to be rel wrt first frame. 
                
                # converted_obj_rel_rot_mat = ds.prep_rel_obj_rot_mat_w_reference_mat(converted_obj_rot_mat, \
                #                     x_start[:, 0:1, 3:12].reshape(b, -1, 3, 3)) # seems wrong 
                converted_obj_rel_rot_mat = ds.prep_rel_obj_rot_mat_w_reference_mat(converted_obj_rot_mat, \
                                    ref_frame_rot_mat) 

                aligned_human_trans = global_human_jpos[:, 0:1, 0, :] - converted_human_jpos[:, 0:1, 0, :]
                # aligned_human_trans = global_human_jpos[:, -1:, 0, :] - converted_human_jpos[:, concat_time_frame_idx-1:concat_time_frame_idx, 0, :]
                converted_human_jpos += aligned_human_trans[:, :, None, :]

                converted_obj_com_pos += aligned_human_trans 
                converted_normalized_obj_com_pos = ds.normalize_obj_pos_min_max(converted_obj_com_pos)

                converted_normalized_human_jpos = ds.normalize_jpos_min_max(converted_human_jpos) 

                converted_curr_x = torch.cat((converted_normalized_obj_com_pos.reshape(b, self.seq_len, -1), \
                            converted_obj_rel_rot_mat.reshape(b, self.seq_len, -1), \
                            converted_normalized_human_jpos.reshape(b, self.seq_len, -1), \
                            converted_rot_6d.reshape(b, self.seq_len, -1)), dim=-1) 
                
                if self.use_object_keypoints:
                    converted_curr_x = torch.cat((converted_curr_x, curr_x[:, :, -4:]), dim=-1) 

                if apply_interpolation:
                    whole_sample_res = torch.cat((whole_sample_res[:, :-concat_time_frame_idx], converted_curr_x), dim=1) 
                else:
                    whole_sample_res = torch.cat((whole_sample_res, converted_curr_x[:, concat_time_frame_idx:]), dim=1) 

                window_idx += 1 

                # For debug 
                # whole_sample_res = torch.cat((whole_sample_res, converted_curr_x[:, :-concat_time_frame_idx]), dim=1) 

        return whole_sample_res # BS X T X D (3+9+24*3+22*6)

    def apply_rotation_to_data(self, ds, trans2joint, cano_rot_mat, new_obj_rot_mat, curr_x):
        # cano_rot_mat:BS X 3 X 3, convert from the coodinate frame which canonicalize the first frame of a sequence to 
        # the frame that canonicalize the first frame of a window.
        # trans2joint: BS X 3 
        # new_obj_rot_mat: BS X 10(overlapped -length) X 3 X 3
        # curr_x: BS X window_size X D  
        # This function is to convert window data to sequence data. 
        bs, timesteps, _ = curr_x.shape 

        pred_obj_normalized_com_pos = curr_x[:, :, :3] # BS X window_size X 3 
        pred_obj_com_pos = ds.de_normalize_obj_pos_min_max(pred_obj_normalized_com_pos) 
        pred_obj_rel_rot_mat = curr_x[:, :, 3:12].reshape(bs, timesteps, 3, 3) # BS X window_size X 3 X 3, relative rotation wrt current window's first frames's cano.   
        pred_obj_rot_mat = ds.rel_rot_to_seq(pred_obj_rel_rot_mat, new_obj_rot_mat) # BS X window_size X 3 X 3 
        pred_human_normalized_jpos = curr_x[:, :, 12:12+24*3] # BS X window_size X (24*3)
        pred_human_jpos = ds.de_normalize_jpos_min_max(pred_human_normalized_jpos.reshape(bs, timesteps, 24, 3)) # BS X window_size X 24 X 3 
        pred_human_rot_6d = curr_x[:, :, 12+24*3:] # BS X window_size X (22*6) 

        pred_human_rot_mat = transforms.rotation_6d_to_matrix(pred_human_rot_6d.reshape(bs, timesteps, 22, 6)) # BS X T X 22 X 3 X 3 

        converted_obj_com_pos = torch.matmul(cano_rot_mat[:, None, :, :].repeat(1, timesteps, \
                            1, 1).transpose(2, 3), \
                            pred_obj_com_pos[:, :, :, None]).squeeze(-1) # BS X window_size X 3 
        # converted_obj_com_pos = torch.matmul(cano_rot_mat[:, None, :, :].repeat(1, timesteps, \
        #                     1, 1).transpose(2, 3), \
        #                     pred_obj_com_pos[:, :, :, None]-trans2joint[:, None, :, None].to(pred_obj_com_pos.device)).squeeze(-1) # BS X window_size X 3 
        # converted_obj_com_pos += trans2joint[:, None, :].to(converted_obj_com_pos.device)

        converted_obj_rot_mat = torch.matmul(cano_rot_mat[:, None, :, :].repeat(1, timesteps, \
                            1, 1).transpose(2, 3), pred_obj_rot_mat) # BS X window_size X 3 X 3 
     
        converted_human_jpos = torch.matmul(cano_rot_mat[:, None, None, :, :].repeat(1, timesteps, 24, 1, 1).transpose(3, 4), \
                    pred_human_jpos[:, :, :, :, None]).squeeze(-1) # BS X T X 24 X 3 
        converted_rot_mat = torch.matmul(cano_rot_mat[:, None, None, :, :].repeat(1, timesteps, 22, 1, 1).transpose(3, 4), \
                    pred_human_rot_mat) # BS X T X 22 X 3 X 3 

        converted_rot_6d = transforms.matrix_to_rotation_6d(converted_rot_mat) 

        # converted_obj_com_pos = ds.normalize_obj_pos_min_max(converted_obj_com_pos)
        # converted_human_jpos = ds.normalize_jpos_min_max(converted_human_jpos)
        # converted_curr_x = torch.cat((converted_obj_com_pos, converted_obj_rot_mat, \
        #         converted_human_jpos, converted_rot_6d), dim=-1) 

        return converted_obj_com_pos, converted_obj_rot_mat, converted_human_jpos, converted_rot_6d 

    # @torch.no_grad()
    def sample(self, x_start, ori_x_cond, cond_mask=None, padding_mask=None, \
            language_input=None, contact_labels=None, rest_human_offsets=None, \
            data_dict=None, guidance_fn=None, opt_fn=None, return_diff_level_res=False):
        # naive conditional sampling by replacing the noisy prediction with input target data. 
        self.denoise_fn.eval() 
        self.bps_encoder.eval()
        self.clip_encoder.eval()

        if ori_x_cond is not None:
            # (BPS representation) Encode object geometry to low dimensional vectors.            
            x_cond = self.bps_encoder(ori_x_cond) # BS X 1 X 256 
            x_cond = x_cond.repeat(1, self.seq_len, 1) # BS X T X (3+256) 
        else:
            x_cond = None 

        if cond_mask is not None:
            x_pose_cond = x_start * (1. - cond_mask) # Remove noise, overall better than adding random noise. 
            
            if x_cond is not None:
                x_cond = torch.cat((x_cond, x_pose_cond), dim=-1) # BS X T X (3+256+3+9) 
            else:
                x_cond = x_pose_cond 
       
        if language_input is not None:
            language_embedding = self.clip_encoder(language_input)
        else:
            language_embedding = None 
        
        if guidance_fn is not None:
            sample_res = self.p_sample_loop_guided(x_start.shape, \
                    x_cond, guidance_fn, opt_fn=opt_fn, \
                    language_embedding=language_embedding, rest_human_offsets=rest_human_offsets, \
                    data_dict=data_dict, contact_labels=contact_labels, \
                    cond_mask=cond_mask, padding_mask=padding_mask)
            # BS X T X D
        else:
            sample_res = self.p_sample_loop(x_start.shape, x_cond, \
                    language_embedding=language_embedding, padding_mask=padding_mask, \
                    cond_mask=cond_mask, ori_pose_cond=x_start, return_diff_level_res=return_diff_level_res)
            # BS X T X D

        self.denoise_fn.train()
        self.bps_encoder.train()
        self.clip_encoder.train() 

        return sample_res  

    def predict_noise_from_start(self, x_t, t, x_start):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_start) /
            (extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))
        )

    def ddim_sample(self, x_start, ori_x_cond, cond_mask=None, padding_mask=None, \
            language_input=None, contact_labels=None, rest_human_offsets=None, \
            data_dict=None, guidance_fn=None, opt_fn=None, return_diff_level_res=False):
        # naive conditional sampling by replacing the noisy prediction with input target data. 
        self.denoise_fn.eval() 
        self.bps_encoder.eval()
        self.clip_encoder.eval()

        shape = x_start.shape 
        batch, device, total_timesteps, sampling_timesteps, eta = \
        shape[0], self.betas.device, self.num_timesteps, 50, 1

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]


        start_time = PyTime.time() 

        x = torch.randn(shape, device = device)

        if ori_x_cond is not None:
            # (BPS representation) Encode object geometry to low dimensional vectors.            
            # x_cond = torch.cat((ori_x_cond[:, :, :3], self.bps_encoder(ori_x_cond[:, :, 3:])), dim=-1) # BS X 1 X (3+256) 
            x_cond = self.bps_encoder(ori_x_cond) # BS X 1 X 256 
            x_cond = x_cond.repeat(1, self.seq_len, 1) # BS X T X (3+256) 
        else:
            x_cond = None 

        if cond_mask is not None:
            x_pose_cond = x_start * (1. - cond_mask) # Remove noise, overall better than adding random noise. 
            # x_pose_cond = x_start * (1. - cond_mask) + cond_mask * torch.randn_like(x_start).to(x_start.device)

            if x_cond is not None:
                x_cond = torch.cat((x_cond, x_pose_cond), dim=-1) # BS X T X (3+256+3+9) 
            else:
                x_cond = x_pose_cond 
       
        if language_input is not None:
            language_embedding = self.clip_encoder(language_input)
        else:
            language_embedding = None 
        
        if guidance_fn is not None:
            for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
                time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

                if time < 100:
                    # For reconstruction guidance 
                    x = x.detach().requires_grad_(True)

                    # model_output = self.denoise_fn(x, t, x_cond, language_embedding, padding_mask)

                    x_start = self.denoise_fn(x, time_cond, x_cond, padding_mask=padding_mask, \
                        language_embedding=language_embedding)

                    # if self.objective == 'pred_noise':
                    #     x_start = self.predict_start_from_noise(x, t = t, noise = model_output)
                    # elif self.objective == 'pred_x0':
                    #     x_start = model_output
                    # else:
                    #     raise ValueError(f'unknown objective {self.objective}')

                    x_pose_cond = x_cond[:, :, -x.shape[-1]:].detach() 

                    classifier_scale = 1e3

                    # x_start = x_start.detach().requires_grad_(True) 
                
                    loss = guidance_fn(time_cond, x_start, x_pose_cond, cond_mask, \
                        rest_human_offsets, data_dict, \
                        contact_labels=None, \
                        curr_window_ref_obj_rot_mat=None, \
                        prev_window_cano_rot_mat=None, \
                        prev_window_init_root_trans=None) # For hand-object interaction loss  
                
                    gradient = torch.autograd.grad(-loss, x)[0] * classifier_scale # BS(1) X 120 X 216 

                    # gradient = torch.autograd.grad(-loss, x_start)[0] * classifier_scale # BS(1) X 120 X 216 

                    # print("gradient mean:{0}".format(gradient.mean()))
                    # print("gradient min:{0}".format(gradient.min()))
                    # print("gradient max:{0}".format(gradient.max()))

                    # import pdb 
                    # pdb.set_trace() 
                
                    # Peturb predicted clean x 
                    tmp_posterior_variance = extract(self.posterior_variance, time_cond, x_start.shape)
                    # print("variance max:{0}".format(tmp_posterior_variance.max()))
                
                    x_start = x_start + tmp_posterior_variance * gradient.float()
                
                else:
                    x_start = self.denoise_fn(x, time_cond, x_cond, padding_mask=padding_mask, \
                            language_embedding=language_embedding)

                pred_noise = self.predict_noise_from_start(x, time_cond, \
                    x_start=x_start)

                if time_next < 0:
                    x = x_start
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(x)

                x = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise

            # sample_res = self.p_sample_loop_guided(x_start.shape, \
            #         x_cond, guidance_fn, opt_fn=opt_fn, \
            #         language_embedding=language_embedding, rest_human_offsets=rest_human_offsets, \
            #         data_dict=data_dict, contact_labels=contact_labels, \
            #         cond_mask=cond_mask, padding_mask=padding_mask)
            # BS X T X D
            sample_res = x 
        else:
            for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
                time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
                x_start = self.denoise_fn(x, time_cond, x_cond, padding_mask=padding_mask, \
                        language_embedding=language_embedding)

                pred_noise = self.predict_noise_from_start(x, time_cond, \
                    x_start=x_start)

                if time_next < 0:
                    x = x_start
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(x)

                x = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise

            # sample_res = self.p_sample_loop(x_start.shape, x_cond, \
            #         language_embedding=language_embedding, padding_mask=padding_mask, \
            #         cond_mask=cond_mask, ori_pose_cond=x_start, return_diff_level_res=return_diff_level_res)
            # BS X T X D

            sample_res = x 
            
        end_time = PyTime.time() 
        print("DDIM single pass takes: {0} seconds".format(end_time-start_time))
        
        self.denoise_fn.train()
        self.bps_encoder.train()
        self.clip_encoder.train() 

        return sample_res  

    # @torch.no_grad()
    def sample_sliding_window_w_canonical(self, ds, object_names, trans2joint, \
                                x_start, ori_x_cond, cond_mask=None, padding_mask=None, \
                                overlap_frame_num=1, input_waypoints=False, \
                                contact_labels=None, language_input=None, \
                                rest_human_offsets=None, data_dict=None, \
                                guidance_fn=None, opt_fn=None):
        # object_names: BS 
        # object_scales: BS X T
        # trans2joint: BS X 3 
        # first_frame_obj_com2tran: BS X 1 X 3 
        # x_start: BS X T X D(3+9+24*3+22*6) (T can be larger than the window_size) 
        # ori_x_cond: BS X 1 X (3+1024*3)
        # cond_mask: BS X window_size X D
        # padding_mask: BS X T 
        self.denoise_fn.eval()
        self.bps_encoder.eval()
        self.clip_encoder.eval()

        sample_res = self.p_sample_loop_sliding_window_w_canonical(ds, object_names, \
                trans2joint, x_start, \
                ori_x_cond, cond_mask=cond_mask, padding_mask=padding_mask, \
                overlap_frame_num=overlap_frame_num, input_waypoints=input_waypoints, \
                contact_labels=contact_labels, language_input=language_input, \
                rest_human_offsets=rest_human_offsets, data_dict=data_dict, \
                guidance_fn=guidance_fn, opt_fn=opt_fn)

        # BS X T X D
        self.denoise_fn.train()
        self.bps_encoder.train()
        self.clip_encoder.train() 
      
        return sample_res  

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, x_cond, t, language_embedding=None, noise=None, \
        padding_mask=None, rest_human_offsets=None, data_dict=None, ds=None):
        # x_start: BS X T X D
        # x_cond: BS X T X D_cond
        # padding_mask: BS X 1 X T 
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise) # noisy motion in noise level t. 

        model_out = self.denoise_fn(x, t, x_cond, language_embedding=language_embedding, padding_mask=padding_mask)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if padding_mask is not None:
            loss = self.loss_fn(model_out, target, reduction = 'none') * padding_mask[:, 0, 1:][:, :, None]
        else:
            loss = self.loss_fn(model_out, target, reduction = 'none') # BS X T X D 

        loss = reduce(loss, 'b ... -> b (...)', 'mean') # BS X (T*D)

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        loss_reshaped = loss.reshape(x_start.shape[0], self.seq_len, -1) 

        # import pdb 
        # pdb.set_trace() 

        loss_object = loss_reshaped[:, :, :12]

        if loss_reshaped.shape[-1] == 12: # objetc motion only
            loss_human = torch.zeros(1) 
        else:
            loss_human = loss_reshaped[:, :, 12:]

        if self.use_object_keypoints:
            hand_idx = [20, 21, 22, 23]
            foot_idx = [7, 8, 10, 11]

            bs, num_steps, _ = model_out.shape 

            # model_contact = model_out[:, :, -4:]

            gt_global_jpos = target[:, :, 12:12+24*3].reshape(bs, num_steps, 24, 3)
            gt_global_jpos = ds.de_normalize_jpos_min_max(gt_global_jpos) # BS X T X 24 X 3 
            gt_global_hand_jpos = gt_global_jpos[:, :, hand_idx, :] # BS X T X 4 X 3 
            gt_global_foot_jpos = gt_global_jpos[:, :, foot_idx, :] # BS X T X 4 X 3 

            global_jpos = model_out[:, :, 12:12+24*3].reshape(bs, num_steps, 24, 3)
            global_jpos = ds.de_normalize_jpos_min_max(global_jpos) # BS X T X 24 X 3  

            # static_idx = model_contact > 0.95  # BS x T x 4

            # FK to get joint positions. rest_human_offsets: BS X 24 X 3 
            curr_seq_local_jpos = rest_human_offsets[:, None].repeat(1, num_steps, 1, 1).cuda() # BS X T X 24 X 3  
            curr_seq_local_jpos = curr_seq_local_jpos.reshape(bs*num_steps, 24, 3) # (BS*T) X 24 X 3 
            curr_seq_local_jpos[:, 0, :] = global_jpos.reshape(bs*num_steps, 24, 3)[:, 0, :] # (BS*T) X 3 
            
            global_joint_rot_6d = model_out[:, :, 12+24*3:12+24*3+22*6].reshape(bs, num_steps, 22, 6) # BS X T X 22 X 6
            global_joint_rot_mat = transforms.rotation_6d_to_matrix(global_joint_rot_6d) # BS X T X 22 X 3 X 3 
            local_joint_rot_mat = quat_ik_torch(global_joint_rot_mat.reshape(-1, 22, 3, 3)) # (BS*T) X 22 X 3 X 3 
            _, human_jnts = quat_fk_torch(local_joint_rot_mat, curr_seq_local_jpos)
            human_jnts = human_jnts.reshape(bs, num_steps, 24, 3) # BS X T X 24 X 3 

            pred_global_hand_jpos = human_jnts[:, :, hand_idx, :] # BS X T X 4 X 3 
            pred_global_foot_jpos = human_jnts[:, :, foot_idx, :] # BS X T X 4 X 3 

            # Add fk loss
            # fk_loss = self.loss_fn(
            #     human_jnts, gt_global_jpos, reduction="none"
            # ) * padding_mask[:, 0, 1:][:, :, None, None] # BS X T X 24 X 3 
            
            # Add fk loss for hand and feet only
            fk_hand_loss = self.loss_fn(
                pred_global_hand_jpos, gt_global_hand_jpos, reduction="none"
            ) * padding_mask[:, 0, 1:][:, :, None, None] # BS X T X 24 X 3 
            fk_hand_loss = reduce(fk_hand_loss, "b ... -> b (...)", "mean")

            fk_hand_loss = fk_hand_loss * extract(self.p2_loss_weight, t, fk_hand_loss.shape)

            fk_foot_loss = self.loss_fn(
                pred_global_foot_jpos, gt_global_foot_jpos, reduction="none"
            ) * padding_mask[:, 0, 1:][:, :, None, None] # BS X T X 24 X 3 
            fk_foot_loss = reduce(fk_foot_loss, "b ... -> b (...)", "mean")

            fk_foot_loss = fk_foot_loss * extract(self.p2_loss_weight, t, fk_foot_loss.shape)

            fk_loss = fk_hand_loss + fk_foot_loss 

            # Add foot contact loss 
            # model_feet = gt_global_jpos[:, :, foot_idx]  # foot positions (BS, T, 4, 3), GT debug 
            # model_feet = human_jnts[:, :, foot_idx]  # foot positions (BS, T, 4, 3)
            # model_foot_v = torch.zeros_like(model_feet)
            # model_foot_v[:, :-1] = (
            #     model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
            # )  # (N, S-1, 4, 3)
            # model_foot_v[~static_idx] = 0
           
            # foot_loss = self.loss_fn(
            #     model_foot_v, torch.zeros_like(model_foot_v), reduction="none"
            # ) * padding_mask[:, 0, 1:][:, :, None, None] 
            # foot_loss = reduce(foot_loss, "b ... -> b (...)", "mean")

            # foot_loss = foot_loss * extract(self.p2_loss_weight, t, foot_loss.shape)
            # print("foot loss gt:{0}".format(foot_loss.mean()))

            # Tmp use wring loss name here. 
            model_semantic_contact = model_out[:, :, -4:] # BS X T X 4 
            foot_loss = self.loss_fn(
                model_semantic_contact, target[:, :, -4:], reduction="none"
            ) * padding_mask[:, 0, 1:][:, :, None]
            foot_loss = reduce(foot_loss, "b ... -> b (...)", "mean")
            foot_loss = foot_loss * extract(self.p2_loss_weight, t, foot_loss.shape)

            # Add "FK loss" for object 
            rest_pose_obj_kpts = data_dict['rest_pose_obj_pts'].to(model_out.device) # BS X K X 3 
            gt_seq_obj_kpts = data_dict['ori_obj_keypoints'].to(model_out.device) # BS X T X K X 3 

            pred_obj_rel_rot_mat = model_out[:, :, 3:3+9].reshape(bs, num_steps, 3, 3) # BS X T X 3 X 3  
            ref_obj_rot_mat = data_dict['reference_obj_rot_mat'].to(model_out.device) # BS X 1 X 3 X 3 
            ref_obj_rot_mat = ref_obj_rot_mat.repeat(1, pred_obj_rel_rot_mat.shape[1], 1, 1) # BS X T X 3 X 3
            pred_obj_rot_mat = torch.matmul(pred_obj_rel_rot_mat, ref_obj_rot_mat.to(pred_obj_rel_rot_mat.device)) # BS X T X 3 X 3

            pred_normalized_obj_com_pos = model_out[:, :, :3] # BS X T X 3 
            pred_obj_com_pos = ds.de_normalize_obj_pos_min_max(pred_normalized_obj_com_pos) # BS X T X 3 

            pred_seq_obj_kpts = torch.matmul(pred_obj_rot_mat[:, :, None, :, :].repeat(1, 1, rest_pose_obj_kpts.shape[1], 1, 1), \
                    rest_pose_obj_kpts[:, None, :, :, None].repeat(1, num_steps, 1, 1, 1)) + pred_obj_com_pos[:, :, None, :, None] # BS X T X K X 3 
            # BS X T X K X 3 X 3, BS X T X K X 3 X 1 + BS X T X 1 X 3 X 1 --> BS X T X K X 3 X 1 

            pred_seq_obj_kpts = pred_seq_obj_kpts.squeeze(-1) # BS X T X K X 3  

            loss_obj_pts = self.loss_fn(
                pred_seq_obj_kpts, gt_seq_obj_kpts, reduction="none"
            ) * padding_mask[:, 0, 1:][:, :, None, None] # BS X T X 24 X 3 
            loss_obj_pts = reduce(loss_obj_pts, "b ... -> b (...)", "mean")

            loss_obj_pts = loss_obj_pts * extract(self.p2_loss_weight, t, loss_obj_pts.shape)
           
            return loss.mean(), loss_object.mean(), loss_human.mean(), \
                foot_loss.mean(), fk_loss.mean(), loss_obj_pts.mean()   
        
        return loss.mean(), loss_object.mean(), loss_human.mean() 

    def forward(self, x_start, ori_x_cond, cond_mask=None, padding_mask=None, \
        language_input=None, contact_labels=None, rest_human_offsets=None, data_dict=None, ds=None): 
        # x_start: BS X T X D, we predict object motion 
        # (relative rotation matrix 9-dim with respect to the first frame, absolute translation 3-dim)
        # ori_x_cond: BS X 1 X D' (com pos + BPS representation), we only use the first frame.  
        # language_embedding: BS X D(512) 
        # contact_labels: BS X T 
        # rest_human_offsets: BS X 24 X 3
        bs = x_start.shape[0] 
        t = torch.randint(0, self.num_timesteps, (bs,), device=x_start.device).long()

        # (BPS representation) Encode object geometry to low dimensional vectors. 
        if ori_x_cond is not None:
            # x_cond = torch.cat((ori_x_cond[:, :, :3], self.bps_encoder(ori_x_cond[:, :, 3:])), dim=-1) # BS X 1 X (3+256) 
            x_cond = self.bps_encoder(ori_x_cond) # BS X 1 X 256 
            x_cond = x_cond.repeat(1, self.seq_len, 1) # BS X T X (3+256) 
        else:
            x_cond = None 

        if cond_mask is not None:
            x_pose_cond = x_start * (1. - cond_mask) 
            if x_cond is not None:
                x_cond = torch.cat((x_cond, x_pose_cond), dim=-1) # BS X T X (3+256+3+9) 
            else:
                x_cond = x_pose_cond 

        if language_input is not None:
            language_embedding = self.clip_encoder(language_input) # BS X d_model 
        else:
            language_embedding = None 

        if self.use_object_keypoints:
            curr_loss, curr_loss_obj, curr_loss_human, curr_loss_feet, curr_loss_fk, curr_loss_obj_pts = \
                        self.p_losses(x_start, x_cond, t, \
                        language_embedding=language_embedding, padding_mask=padding_mask, \
                        rest_human_offsets=rest_human_offsets, data_dict=data_dict, ds=ds)  

            return curr_loss, curr_loss_obj, curr_loss_human, curr_loss_feet, curr_loss_fk, curr_loss_obj_pts 
        else:
            curr_loss, curr_loss_obj, curr_loss_human = self.p_losses(x_start, x_cond, t, \
                        language_embedding=language_embedding, padding_mask=padding_mask, \
                        rest_human_offsets=rest_human_offsets, data_dict=data_dict, ds=ds)  

            return curr_loss, curr_loss_obj, curr_loss_human 
        