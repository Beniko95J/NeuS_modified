import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.dataset_dtu import DatasetDTU
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
from models.eval import evaluation_shinyblender
import open3d as o3d
import json
import torchvision
import torch.nn as nn
import math
import imageio


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', appendix='', is_continue=False):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        conf_text = conf_text.replace('APPENDIX', appendix)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        if 'dtu' in case:
            d_factory = DatasetDTU
        else:
            d_factory = Dataset

        self.dataset = d_factory(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])

        # Intermediate Mesh 
        self.scene = None
        
        # Write evaluation metric
        self.result = open(os.path.join(self.base_exp_dir, 'result.txt'), 'a')

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            data, intrinsic, intrinsic_inv, pose = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)

            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              intrinsics=intrinsic,
                                              intrinsics_inv=intrinsic_inv,
                                              poses=pose)

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            gradient_ref_error = render_out['gradient_ref_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error
            if gradient_ref_error is not None:
                eikonal_loss += gradient_ref_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_fine_loss +\
                   eikonal_loss * self.igr_weight +\
                   mask_loss * self.mask_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def load_ckpt_validation(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d, _, intrinsic, intrinsic_inv, pose = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb,
                                              intrinsics=intrinsic,
                                              intrinsics_inv=intrinsic_inv,
                                              poses=pose)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

    def validate_image_refneus(self, idx=-1, resolution_level=-1, only_normals=False, pose=None):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: camera: {}'.format(idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        if pose is None:  # FIXME
            rays_o, rays_d, uv, intrinsic, intrinsic_inv, pose = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        else:
            rays_o, rays_d, uv = self.dataset.gen_rays_visu(idx, pose, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        uv = uv.reshape(-1, 2).split(self.batch_size)
        
        out_rgb_fine = []
        out_normal_fine = []
        out_depth_sdf = []
        out_normal_sdf = []

        for rays_o_batch, rays_d_batch, uv_batch in zip(rays_o, rays_d, uv):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb,
                                              intrinsics=intrinsic,
                                              intrinsics_inv=intrinsic_inv,
                                              poses=pose)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                normals = render_out['normal_map']
                out_normal_fine.append(normals)
            if feasible('depth_sdf') and feasible('normal_sdf'):
                out_depth_sdf.append(render_out['depth_sdf'])
                out_normal_sdf.append(render_out['normal_sdf'])
            del render_out
            
        normal_img = None
        depth_sdf_img = None
        normal_sdf_img = None
        if len(out_normal_fine) > 0:
            normal_img =  torch.from_numpy(np.concatenate(out_normal_fine, axis=0)) / 2. + 0.5
            normal_img = normal_img.permute(1, 0).reshape([3, H, W]) 
        if len(out_depth_sdf) > 0:
            depth_sdf_img = torch.from_numpy(np.concatenate(out_depth_sdf, axis=0))
            depth_sdf_img = depth_sdf_img.reshape([H, W])
            depth_sdf_img = depth_sdf_img / depth_sdf_img.max()
            normal_sdf_img = torch.from_numpy(np.concatenate(out_normal_sdf, axis=0)) / 2. + 0.5
            normal_sdf_img = normal_sdf_img.permute(1, 0).reshape([3, H, W])
        
        vis_validation_dir = os.path.join(self.base_exp_dir, 'vis_validation')
        os.makedirs(os.path.join(vis_validation_dir, 'normals_all'), exist_ok=True)
        os.makedirs(os.path.join(vis_validation_dir, 'test_images_all'), exist_ok=True)
        os.makedirs(os.path.join(vis_validation_dir, 'normals'), exist_ok=True)

        if depth_sdf_img is not None:
            os.makedirs(os.path.join(vis_validation_dir, 'depth_sdf'), exist_ok=True)
            os.makedirs(os.path.join(vis_validation_dir, 'normal_sdf'), exist_ok=True)
            torchvision.utils.save_image(depth_sdf_img.clone(), os.path.join(vis_validation_dir, 'depth_sdf', '{:0>8d}_0_{}.png'.format(self.iter_step, idx)), nrow=8)
            torchvision.utils.save_image(normal_sdf_img.clone(), os.path.join(vis_validation_dir, 'normal_sdf', '{:0>8d}_0_{}.png'.format(self.iter_step, idx)), nrow=8)

        torchvision.utils.save_image(normal_img.clone(), os.path.join(vis_validation_dir, 'normals', '{:0>8d}_0_{}.png'.format(self.iter_step, idx)), nrow=8)
        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)
            
        if only_normals:
            torchvision.utils.save_image(normal_img.clone(), os.path.join(vis_validation_dir, 'normals_all', '{:0>8d}_0_{}.png'.format(self.iter_step, idx)), nrow=8)
            cv.imwrite(os.path.join(vis_validation_dir, 'test_images_all', '{:0>8d}_0_{}.png'.format(self.iter_step, idx)), img_fine[..., 0])
            normal_img[0,:,:], normal_img[2,:,:] = normal_img[2,:,:].clone(), normal_img[0,:,:].clone()
            return normal_img.permute(1,2,0) * 256., torch.from_numpy(img_fine[..., 0] / 256.)
        

        os.makedirs(os.path.join(vis_validation_dir, 'validations_fine'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(vis_validation_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
        if pose is not None:
            img_fine = torch.from_numpy(img_fine[..., 0])
            img_fine[:,:,0], img_fine[:,:,2] = img_fine[:,:,2].clone(), img_fine[:,:,0].clone()
            return img_fine, normal_img.permute(1,2,0)

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    def validate_mesh_refneus(self, result, resolution=64, threshold=0.0, ckpt_path=None, validate_normal=False):
        if ckpt_path is not None:
            self.load_ckpt_validation(ckpt_path)
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)
        
        vertices, triangles =\
        self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
        
        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}_inter_mesh.ply'.format(self.iter_step)))
        
        # For visibility identification
        mesh_ = o3d.io.read_triangle_mesh(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}_inter_mesh.ply'.format(self.iter_step)))
        mesh_ = o3d.t.geometry.TriangleMesh.from_legacy(mesh_)
        scene = o3d.t.geometry.RaycastingScene()
        cube_id = scene.add_triangles(mesh_)
        
        if self.iter_step % 10000 == 0 and self.iter_step != 0: 
            resolution = 512

            vertices, triangles =\
                self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
            os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
            
            mesh = trimesh.Trimesh(vertices, triangles)
            mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

            mesh.apply_transform(self.dataset.scale_mat)  #transform to orignial space for evaluation
            mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}_eval.ply'.format(self.iter_step)))
            mesh_eval = o3d.io.read_triangle_mesh(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}_eval.ply'.format(self.iter_step)))
            with open(os.path.join(self.conf['dataset'].data_dir, 'test_info.json'), 'r') as f:
                text_info = json.load(f)
            points_for_plane = text_info['points']
            max_dist_d = text_info['max_dist_d']
            max_dist_t = text_info['max_dist_t']
            try:
                nonvalid_bbox = text_info['nonvalid_bbox']
            except:
                nonvalid_bbox = None
            # mean_d2s, mean_s2d, over_all = evaluation_shinyblender(mesh_eval, os.path.join(self.conf['dataset'].data_dir, 'dense_pcd.ply'),self.base_exp_dir, 
            #                                                        max_dist_d=max_dist_d, max_dist_t=max_dist_t, points_for_plane=points_for_plane, nonvalid_bbox=nonvalid_bbox )

            # result.write(f'{self.iter_step}: ')
            # result.write(f'{mean_d2s} {mean_s2d} {over_all}')
            # result.write('\n') 
            # result.flush()
            if self.iter_step == self.end_iter - 1 or ckpt_path is not None and validate_normal:
                self.validate_all_normals()

        logging.info('End')
        return scene

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()


    def validate_all_normals(self):
        total_MAE = 0
        total_PNSR = 0
        idxs = [i for i in range(self.dataset.n_images)]
        f = open(os.path.join(self.base_exp_dir, 'result_normal.txt'), 'a')
        for idx in idxs:
            normal_maps, color_fine = self.validate_image_refneus(idx, resolution_level=1, only_normals=True)
            try:
                GT_normal = torch.from_numpy(self.dataset.normal_np[idx])
                GT_color = torch.from_numpy(self.dataset.images_np[idx])
                PSNR = 20.0 * torch.log10(1.0 / ((color_fine - GT_color)**2).mean().sqrt())
                total_PNSR += PSNR
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                cos_loss = cos(normal_maps.view(-1, 3), GT_normal.view(-1, 3))
                cos_loss = torch.clamp(cos_loss, (-1.0 + 1e-10), (1.0 - 1e-10))
                loss_rad = torch.acos(cos_loss)
                loss_deg = loss_rad * (180.0 / math.pi)
                total_MAE += loss_deg.mean()
                f.write(str(idx) + '_MAE:')
                f.write(str(loss_deg.mean().data.item()) + '    ')
                f.write(str(idx) + '_psnr:')
                f.write(str(PSNR.data.item()))
                f.write('\n')
                f.flush()
            except:
                continue
        MAE = total_MAE / self.dataset.n_images
        PSNR = total_PNSR / self.dataset.n_images
        f.write('\n')
        f.write('MAE_final:')
        f.write(str(MAE.data.item()) + '    ')
        f.write('PSNR_final:')
        f.write(str(PSNR.data.item()))
        f.close()


if __name__ == '__main__':
    print('Hello Wooden')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--appendix', type=str, default='')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--validate_normal', default=False, action="store_true")

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case,  args.appendix, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        # runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
        runner.validate_mesh_refneus(runner.result, resolution=512, threshold=args.mcube_threshold, ckpt_path=args.ckpt_path, validate_normal=args.validate_normal)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
