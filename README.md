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

<p align="center" width="100%">
    <video src="https://github.com/yanyan-li/GeoGaussian/blob/main/img/teaser_challenging.mp4"></video>
</p>

| 3DGS    | LightGS | GeoGaussian |
| :------: | :------: | :------:
| <img width="100%" src="./img/gif/o2-3DGS.gif">  |  <img width="100%" src="./img/gif/o2-light.gif">   |<img width="100%" src="./img/gif/o2-ours.gif">|

### BibTex
```
@article{li2024geogaussian,
  title={GeoGaussian: Geometry-aware Gaussian Splatting for Scene Rendering},
  author={Li, Yanyan and Lyu, Chenyu and Di, Yan and Zhai, Guangyao and Lee, Gim Hee and Tombari, Federico},
  journal={arXiv preprint arXiv:2403.11324},
  year={2024}
}
```


### 1.Dataset
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

**The subdataset can be obtained via [Replica (PlanarSLAM)](https://drive.google.com/drive/folders/1LO0a-M__cZJu3TnaMX-fEP4YxFs5LDGZ?usp=drive_link), [TUM RGB-D (PlanarSLAM)](https://drive.google.com/drive/folders/1hDPRH3FGg_HpQYwZWg_wgZbonClVbcbC?usp=drive_link), [ICL NUIM (PlanarSLAM)](https://drive.google.com/drive/folders/1UV7DqybCUcYl3Yn4kV030lQOKhwGUHU6?usp=drive_link). Then you need to place the raw dataset images in the ``results`` folder. The raw images can be obtained via [Replica](https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip), [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download), [ICL NUIM](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html).**

### 2.Baseline
**1. Gaussian-Splatting with Planar Point Clouds**
[Repo](https://github.com/yanyan-li/gaussian-splatting-using-PlanarSLAM?tab=readme-ov-file)




### Setup of GeoGaussian
The code will be released soon!
