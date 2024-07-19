# GeoGaussian: Geometry-aware Gaussian Splatting for Scene Rendering (ECCV 2024)

<p align="center" width="100%">
    <img width="33%" src="./img/logo.gif">
</p>

--------
<p align="center">
	<a href="https://yanyan-li.github.io/project/gs/geogaussian.html"><img src="https://img.shields.io/badge/GeoGaussian-ProjectPage-green.svg"></a>
     <a href="http://arxiv.org/abs/2403.11324"><img src="https://img.shields.io/badge/GeoGaussian-Paper-yellow.svg"></a>
    <a href="https://"><img src="https://img.shields.io/badge/GeoGaussian-video-blue.svg"></a>
</p>


## 📖 Abstract
During the Gaussian Splatting optimization process, the scene
geometry can gradually deteriorate if its structure is not deliberately 
preserved, especially in non-textured regions such as walls, ceilings, 
and furniture surfaces. This degradation significantly affects 
the rendering quality of novel views that deviate significantly 
from the viewpoints in the training data. 
To mitigate this issue, we propose a novel approach called **GeoGaussian**. 
Based on the smoothly connected areas observed from point clouds, 
this method introduces a novel pipeline to initialize thin Gaussians 
aligned with the surfaces, where the characteristic can be transferred 
to new generations through a carefully designed densification strategy. 
Finally, the pipeline ensures that the scene
geometry and texture are maintained through constrained optimization 
processes with explicit geometry constraints. Benefiting from 
the proposed architecture, the generative ability of 3D Gaussians 
is enhanced, especially in structured regions.
Our proposed pipeline achieves state-of-the-art performance 
in novel view synthesis and geometric reconstruction, 
as evaluated qualitatively and quantitatively on public datasets.

<div align="center">
<img width="100%" alt="image" src="img/gif/compressed-R0-out.gif">
<img width="100%" alt="image" src="img/gif/compressed-o2-out.gif">
<img width="100%" alt="image" src="img/gif/compressed-icl-r2-out.gif">
</div>

<div style="display: flex; justify-content: space-between; align-items: center;">  
    <span> </span>  
    <span>3DGS</span>  
    <span> </span>
    <span>LightGS</span>  
    <span> </span>
    <span>GeoGaussian</span>
    <span> </span>
</div>

*Comparisons of novel view rendering on public datasets. At some challenging
viewpoints having bigger differences in translation and orientation motions compared
with training views, 3DGS and LightGS have issues with photorealistic rendering.*

## 📋 TODO Lists
- [✔] *Repo* - Create repo for [GeoGaussian](https://github.com/yanyan-li/GeoGaussian).
- [✔] *Code* - Release for our methods
- [✔] *Code* - Randomly sample N points for each Gaussian point [script](sample_gaussian_model.py)
- [✔] *Dataset* - Upload [dataset](#-dataset) download link.
- [✔] *ReadMe* - Teaser( [I](img/gif/compressed-o2-out.gif) [II](img/gif/compressed-o2-out.gif) [III](img/gif/compressed-icl-r2-out.gif) ) images & [Abstract](#-abstract).
- [✔] *ReadMe* - Geometry-aware [Strategies](#-geometry-aware-strategies).
- [ ] *ReadMe* - Repository [Setup](#-setup-of-geogaussian).
- [✔] *ReadMe* - [Results](#-results) for Table I. & Table II.
- [✔] *ReadMe* - [License](#-license) & [Acknowledgment](#-acknowledgment) & [Citation](#-citation).
- [✔] *License* - Released under the [Gaussian-Splatting License](LICENSE.md).

## 🚀 Geometry-aware Strategies
<div align="center">
<img width="90%" alt="image" src="img/strategy.png">
</div>

- A parameterization with explicit geometry meaning for thin 3D Gaussians is employed in our carefully designed initialization and densification strategies} to establish reasonable 3D Gaussian models.
- A geometrically consistent constraint is proposed to encourage thin Gaussians to align with the smooth surfaces.



## 💾 Dataset
Based on the SLAM method, **PlanarSLAM**, we create new point clouds rather then using results of COLMAP for experiments. 

<p align="center" width="100%">
    <img width="90%" src="./img/dataset_img.png">
</p>

**New Features of this type of input**
<ol>
<li> Points lying on the non-textured regions </li>
<li> Global plane instances that are represented in different colors </li>
<li> Surface normal vector of each planar point </li>
</ol>

The subdataset can be obtained via 
[Replica (PlanarSLAM)](https://drive.google.com/drive/folders/1LO0a-M__cZJu3TnaMX-fEP4YxFs5LDGZ?usp=drive_link), 
[TUM RGB-D (PlanarSLAM)](https://drive.google.com/drive/folders/1hDPRH3FGg_HpQYwZWg_wgZbonClVbcbC?usp=drive_link), 
[ICL NUIM (PlanarSLAM)](https://drive.google.com/drive/folders/1UV7DqybCUcYl3Yn4kV030lQOKhwGUHU6?usp=drive_link). 
Then you need to place the raw dataset images in the ``results`` folder. 
The raw images can be obtained via 
[Replica](https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip), 
[TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download), 
[ICL NUIM](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html).

For each sequence, the structure is organized as follows
```
Replica_r2
	|______PointClouds.ply          # sparse point clouds from the SLAM system
 	|______KeyFrameTrajectory2.txt  # camera poses from the SLAM system
  	|______results                  # folder for all raw images
```
*our code provides the interface to deal with the type of data format.*


## 🎓 Baseline
**1. Gaussian-Splatting with Planar Point Clouds**
[Repo](https://github.com/yanyan-li/gaussian-splatting-using-PlanarSLAM?tab=readme-ov-file)




## 🔧 Setup of GeoGaussian
### The tutorial will be released soon 🔜 !


## 📊 Results
<style>
    @media (prefers-color-scheme: light) {
        best_data {
            color: #0d1117;
            background: #57cc99;
        }
        second_data {
            color: #0d1117;
            background: #c7f9cc;
        }
    }
    @media (prefers-color-scheme: dark) {
        best_data {
            color: #57cc99;
            background: #0d1117;
        }
        second_data {
            color: #c7f9cc;
            background: #0d1117;
        }
    }
 </style>



<table>
    <tr>
        <td colspan="2"><div align="center">Methods<br>Data</div></td> 
        <td colspan="4"><div align="center">3DGS</div></td> 
        <td colspan="4"><div align="center">LightGS</div></td> 
        <td colspan="4"><div align="center">GeoGaussian(Ours)</div></td>
    </tr>
    <tr>
        <td rowspan="3"><div align="left">R1</div></td>    
        <td rowspan="3"><div align="center">PSNR↑<br>SSIM↑<br>LPIPS↓</div></td>
        <td>30.49</td> 
        <td>33.98</td>
        <td>37.45</td>
        <td>37.60</td>
        <td><second_data>30.54</second_data></td>
        <td><second_data>34.06</second_data></td>
        <td><second_data>37.72</second_data></td>
        <td><best_data>38.44</best_data></td>
        <td><best_data>31.65</best_data></td>
        <td><best_data>35.17</best_data></td>
        <td><best_data>38.00</best_data></td>
        <td><second_data>38.24</second_data></td>
    </tr>
    <tr>
        <td><second_data>0.932</second_data></td>
        <td><second_data>0.951</second_data></td>
        <td>0.964</td>
        <td>0.965</td>
        <td><second_data>0.932</second_data></td>
        <td><second_data>0.951</second_data></td>
        <td><second_data>0.965</second_data></td>
        <td><second_data>0.967</second_data></td>
        <td><best_data>0.937</best_data></td>
        <td><best_data>0.957</best_data></td>
        <td><best_data>0.968</best_data></td>
        <td><best_data>0.979</best_data></td>
    </tr>
    <tr>
        <td><second_data>0.051</second_data></td>
        <td>0.036</td>
        <td>0.029</td>
        <td>0.028</td>
        <td><second_data>0.051</second_data></td>
        <td><second_data>0.035</second_data></td>
        <td><second_data>0.028</second_data></td>
        <td><second_data>0.025</second_data></td>
        <td><best_data>0.041</best_data></td>
        <td><best_data>0.027</best_data></td>
        <td><best_data>0.022</best_data></td>
        <td><best_data>0.021</best_data></td>
    </tr>
    <tr>
        <td rowspan="3"><div align="left">R2</div></td>    
        <td rowspan="3"><div align="center">PSNR↑<br>SSIM↑<br>LPIPS↓</div></td>
        <td>31.53</td>
        <td>35.82</td>
        <td>>38.53</td>
        <td>38.70</td>
        <td><second_data>31.54</second_data></td>
        <td><second_data>35.93</second_data></td>
        <td><second_data>38.78</second_data></td>
        <td><second_data>39.07</second_data></td>
        <td><best_data>32.13</best_data></td>
        <td><best_data>36.81</best_data></td>
        <td><best_data>38.84</best_data></td>
        <td><best_data>39.14</best_data></td>
    </tr>
    <tr>
        <td><second_data>0.935</second_data></td> 
        <td><second_data>0.959</second_data></td> 
        <td><second_data>0.968</second_data></td> 
        <td><second_data>0.968</second_data></td> 
        <td><second_data>0.935</second_data></td> 
        <td><second_data>0.959</second_data></td> 
        <td><second_data>0.968</second_data></td> 
        <td><second_data>0.968</second_data></td> 
        <td><best_data>0.943</best_data></td> 
        <td><best_data>0.963</best_data></td> 
        <td><best_data>0.969</best_data></td> 
        <td><best_data>0.970</best_data></td> 
    </tr>
    <tr>
        <td>0.050</td> 
        <td><second_data>0.031</second_data></td> 
        <td>0.028</td> 
        <td>0.029</td> 
        <td><second_data>0.049</second_data></td> 
        <td><second_data>0.031</second_data></td> 
        <td><second_data>0.027</second_data></td> 
        <td><second_data>0.028</second_data></td> 
        <td><best_data>0.041</best_data></td> 
        <td><best_data>0.025</best_data></td> 
        <td><best_data>0.024</best_data></td> 
        <td><best_data>0.024</best_data></td>
    </tr>
    <tr>
        <td rowspan="3"><div align="left">OFF3</div></td>    
        <td rowspan="3"><div align="center">PSNR↑<br>SSIM↑<br>LPIPS↓</div></td>
        <td>30.90</td> 
        <td>33.86</td> 
        <td>36.26</td> 
        <td>36.56</td> 
        <td><second_data>30.93</second_data></td> 
        <td><second_data>33.90</second_data></td> 
        <td><second_data>36.38</second_data></td> 
        <td><second_data>36.63</second_data></td> 
        <td><best_data>31.62</best_data></td> 
        <td><best_data>33.91</best_data></td> 
        <td><best_data>36.42</best_data></td> 
        <td><best_data>36.66</best_data></td>
    </tr>
    <tr>
        <td><second_data>0.928</second_data></td> 
        <td>0.946</td> 
        <td><second_data>0.958</second_data></td> 
        <td><second_data>0.959</second_data></td> 
        <td><second_data>0.928</second_data></td> 
        <td><second_data>0.947</second_data></td> 
        <td><second_data>0.958</second_data></td> 
        <td>0.958</td> 
        <td><best_data>0.938</best_data></td> 
        <td><best_data>0.953</best_data></td> 
        <td><best_data>0.963</best_data></td> 
        <td><best_data>0.964</best_data></td>
    </tr>
    <tr>
        <td><second_data>0.052</second_data></td> 
        <td><second_data>0.040</second_data></td> 
        <td>0.037</td> 
        <td>0.036</td> 
        <td><second_data>0.052</second_data></td> 
        <td><second_data>0.040</second_data></td> 
        <td><second_data>0.036</second_data></td> 
        <td><second_data>0.037</second_data></td> 
        <td><best_data>0.040</best_data></td> 
        <td><best_data>0.032</best_data></td> 
        <td><best_data>0.029</best_data></td> 
        <td><best_data>0.029</best_data></td>
    </tr>
    <tr>
        <td rowspan="3"><div align="left">OFF4</div></td>    
        <td rowspan="3"><div align="center">PSNR↑<br>SSIM↑<br>LPIPS↓</div></td>
        <td><second_data>29.5</second_data></td> 
        <td>32.98</td> 
        <td>37.70</td> 
        <td>38.48</td> 
        <td>29.51</td> 
        <td><second_data>32.97</second_data></td> 
        <td><second_data>37.95</second_data></td> 
        <td><second_data>38.59</second_data></td> 
        <td><best_data>31.90</best_data></td> 
        <td><best_data>34.61</best_data></td> 
        <td><best_data>38.30</best_data></td> 
        <td><best_data>38.74</best_data></td>
    </tr>
    <tr>
        <td>0.920</td> 
        <td><second_data>0.941</second_data></td> 
        <td><second_data>0.962</second_data></td> 
        <td><second_data>0.964</second_data></td> 
        <td>0.920</td> 
        <td><second_data>0.941</second_data></td> 
        <td><second_data>0.962</second_data></td> 
        <td><second_data>0.964</second_data></td> 
        <td><best_data>0.936</best_data></td> 
        <td><best_data>0.953</best_data></td> 
        <td><best_data>0.966</best_data></td> 
        <td><best_data>0.967</best_data></td>
    </tr>
    <tr>
        <td><second_data>0.070</second_data></td> 
        <td><second_data>0.049</second_data></td> 
        <td>0.037</td> 
        <td><second_data>0.035</second_data></td> 
        <td><second_data>0.070</second_data></td> 
        <td><second_data>0.049</second_data></td> 
        <td><second_data>0.036</second_data></td> 
        <td>0.036</td> 
        <td><best_data>0.050</best_data></td> 
        <td><best_data>0.036</best_data></td> 
        <td><best_data>0.030</best_data></td> 
        <td><best_data>0.031</best_data></td>
    </tr>
    <tr>
        <td rowspan="3"><div align="left">Avg.</div></td>    
        <td rowspan="3"><div align="center">PSNR↑<br>SSIM↑<br>LPIPS↓</div></td>
        <td>30.62</td> 
        <td>34.16</td> 
        <td>37.49</td> 
        <td>37.84</td> 
        <td><second_data>30.63</second_data></td> 
        <td><second_data>34.22</second_data></td> 
        <td><second_data>37.71</second_data></td> 
        <td><second_data>38.18</second_data></td> 
        <td><best_data>31.83</best_data></td> 
        <td><best_data>35.13</best_data></td> 
        <td><best_data>38.18</best_data></td> 
        <td><best_data>38.20</best_data></td>
    </tr>
    <tr>
        <td><second_data>0.929</second_data></td> 
        <td>0.949</td> 
        <td><second_data>0.964</second_data></td> 
        <td><second_data>0.964</second_data></td> 
        <td><second_data>0.929</second_data></td> 
        <td><second_data>0.950</second_data></td> 
        <td><second_data>0.964</second_data></td> 
        <td><second_data>0.964</second_data></td> 
        <td><best_data>0.939</best_data></td> 
        <td><best_data>0.957</best_data></td> 
        <td><best_data>0.967</best_data></td> 
        <td><best_data>0.970</best_data></td>
    </tr>
    <tr>
        <td><second_data>0.056</second_data></td> 
        <td><second_data>0.039</second_data></td> 
        <td><second_data>0.032</second_data></td> 
        <td><second_data>0.032</second_data></td> 
        <td><second_data>0.056</second_data></td> 
        <td><second_data>0.039</second_data></td> 
        <td><second_data>0.032</second_data></td> 
        <td><second_data>0.032</second_data></td> 
        <td><best_data>0.043</best_data></td> 
        <td><best_data>0.030</best_data></td> 
        <td><best_data>0.026</best_data></td> 
        <td><best_data>0.026</best_data></td>
    </tr>
</table>

<div>
<i>
Table I. Comparison of rendering on the Replica dataset. 
↓ indicates the lower the better, 
↑ indicates the higher the better.
The best score is in <best_data>dark green</best_data>. 
The second best score is in <second_data>light green</second_data>.
</i>
</div>


--------

<table>
    <tr>
        <td colspan="2"><div align="center">methods</div></td> 
        <td>R0</td> 
        <td>R1</td> 
        <td>R2</td> 
        <td>OFF0</td> 
        <td>OFF1</td> 
        <td>OFF2</td> 
        <td>OFF3</td> 
        <td>OFF4</td> 
        <td>Avg.</td> 
    </tr>
    <tr>
        <td rowspan="4"><div align="left">3DGS<br><br>GeoGaussian<br>(Ours)</div></td>    
        <td rowspan="2"><div align="center">mean (m)<br>std (m)</div></td>
        <td>0.026</td> 
        <td>0.025</td>
        <td>0.042</td>
        <td>0.017</td>
        <td><best_data>0.019</best_data></td>
        <td>0.039</td>
        <td>0.032</td>
        <td>0.032</td>
        <td>0.029</td>
    </tr>
    <tr>
        <td>0.066</td>
        <td>0.081</td>
        <td>0.146</td>
        <td>0.050</td>
        <td><best_data>0.055</best_data></td>
        <td><best_data>0.201</best_data></td>
        <td>0.066</td>
        <td>0.112</td>
        <td>0.097</td>
    </tr>
    <tr>
<td rowspan="2"><div align="center">mean (m)<br>std (m)</div></td>
        <td><best_data>0.018</best_data></td>
        <td><best_data>0.014</best_data></td>
        <td><best_data>0.015</best_data></td>
        <td><best_data>0.020</best_data></td>
        <td>0.029</td>
        <td><best_data>0.013</best_data></td>
        <td><best_data>0.018</best_data></td>
        <td><best_data>0.014</best_data></td>
        <td><best_data>0.018</best_data></td>
    </tr>
    <tr>
        <td><best_data>0.032</best_data></td> 
        <td><best_data>0.016</best_data></td> 
        <td><best_data>0.028</best_data></td> 
        <td><best_data>0.042</best_data></td> 
        <td>0.067</td> 
        <td>0.024</td> 
        <td><best_data>0.020</best_data></td> 
        <td><best_data>0.023</best_data></td> 
        <td><best_data>0.031</best_data></td> 
    </tr>
</table>

*Table II. Comparison of reconstruction performance on the Replica dataset.
Based on the ground truth mesh models provided by the
Replica dataset, we align these mesh models with point clouds from Gaussian
models, where we randomly sample three points in each Gaussian ellipsoid.*

## ⭕️ Acknowledgment
This project is based on [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) 
and relies on data provided by [PlanarSLAM](https://github.com/yanyan-li/PlanarSLAM) work. 
Thanks for the contribution of the open source community.


## 📋 License
This project is released under the [Gaussian-Splatting License](LICENSE.md).


## ✉️ Citation
If you find this project useful in your research, please consider cite:

*BibTex*
```
@article{li2024geogaussian,
  title={GeoGaussian: Geometry-aware Gaussian Splatting for Scene Rendering},
  author={Li, Yanyan and Lyu, Chenyu and Di, Yan and Zhai, Guangyao and Lee, Gim Hee and Tombari, Federico},
  journal={arXiv preprint arXiv:2403.11324},
  year={2024}
}
```