#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import torchvision
import os

count_epoch = 0

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, rendered_depth, rendered_depth_loss, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        gaussian_type = pc.get_type)

    # Show rgb&depth image
    global count_epoch
    count_epoch = count_epoch + 1
    if count_epoch > 0 and count_epoch%100 == 0:
        trans = torchvision.transforms.ToPILImage()

        rgb_image = trans(torch.clamp(rendered_image, 0.0, 1.0))
        if not os.path.exists('output/rgb/'):
            os.mkdir('output/rgb/')
        if not os.path.exists('output/depth/'):
            os.mkdir('output/depth/')
        rgb_image.save(f'output/rgb/rgb_iter_{count_epoch:0>6d}.png')
        # depth_image = trans((rendered_depth*5000).short())

        depth_array = rendered_depth.squeeze().detach().cpu().numpy()
        depth_array_min = np.min(depth_array)
        depth_array_max = np.max(depth_array)
        depth_array = (depth_array - depth_array_min)/(depth_array_max - depth_array_min + 1e-8)
        # depth_image = pil.Image.fromarray(depth_array, mode='F')
        # 获取Magma colormap
        magma_cmap = get_cmap('magma')
        # 应用Magma colormap
        magma_array = magma_cmap(depth_array)[:, :, :3] * 255
        # 将数组转换回8位整数
        magma_array = np.uint8(magma_array)
        # 创建新的PIL图像
        magma_img = pil.fromarray(magma_array)
        magma_img.save(f'output/depth/depth_iter_{count_epoch:0>6d}.png')

        #
        # image_np = rendered_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        # depth_np = rendered_depth.squeeze().detach().cpu().numpy()
        # vmax_depth = np.percentile(depth_np, 95)
        #
        # plt.figure(num='image', figsize=(8, 8))
        # plt.subplot(211)
        # plt.imshow(image_np)
        # plt.title("Rgb prediction", fontsize=22)
        #
        # plt.subplot(212)
        # plt.imshow(depth_np, cmap='magma', vmax=vmax_depth)
        # plt.title("Depth prediction", fontsize=22)
        #
        # plt.show(block=False)
        # plt.pause(0.01)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "render_depth": rendered_depth,
            "rendered_depth_loss": rendered_depth_loss,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
