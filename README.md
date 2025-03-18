<div align="center">
    <h1>KISS-SLAM</h1>
    <a href="https://github.com/PRBonn/kiss-slam/releases"><img src="https://img.shields.io/github/v/release/PRBonn/kiss-slam?label=version" /></a>
    <a href="https://github.com/PRBonn/kiss-slam/blob/main/LICENSE"><img src="https://img.shields.io/github/license/PRBonn/kiss-slam" /></a>
    <a href="https://github.com/PRBonn/kiss-slam/blob/main/"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <a href="https://github.com/PRBonn/kiss-slam/blob/main/"><img src="https://img.shields.io/badge/mac%20os-000000?&logo=apple&logoColor=white" /></a>
    <br />
    <br />
    <a href="https://github.com/PRBonn/kiss-slam/blob/main/README.md#Install">Install</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/kiss2025iros.pdf">Paper</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href=https://github.com/PRBonn/kiss-slam/issues>Contact Us</a>
  <br />
  <br />

[KISS-SLAM](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/kiss2025iros.pdf) is a simple, robust, and accurate 3D LiDAR SLAM system that **just works**.


![motivation](https://github.com/user-attachments/assets/c784ac23-693c-4872-b80c-79d0ba069f6d)

</div>

<hr />

## Install

```
pip install kiss-slam
```

## Running the system
Next, follow the instructions on how to run the system by typing:
```
kiss_slam_pipeline --help
```

This should print the following help message:

![help](https://github.com/user-attachments/assets/5a6fe624-2aaf-466f-8a18-51039b794000)

### Config
You can generate a default `config.yaml` by typing:

```
kiss_slam_dump_config
```

which will generate a `kiss_slam.yaml` file. Now, you can modify the parameters and pass the file to the `--config` option when running the `kiss_slam_pipeline`.

### Install Python API (developer mode)
For development purposes:

```
sudo apt install git python3-pip libeigen3-dev libsuitesparse-dev
git clone https://github.com/PRBonn/kiss-slam.git
cd kiss-slam
make editable
```

## Citation
If you use this library for any academic work, please cite our original paper:
```bib
@article{kiss2025arxiv,
  author   = {T. Guadagnino and B. Mersch and S. Gupta and I. Vizzo and G. Grisetti and C. Stachniss},
  title    = {{KISS-SLAM: A Simple, Robust, and Accurate 3D LiDAR SLAM System With Enhanced Generalization Capabilities}},
  journal  = {arXiv preprint},
  year     = 2025,
  volume   = {arXiv:2503.12660},
  url      = {https://arxiv.org/pdf/2503.12660},
}
```

## Acknowledgements
This project builds on top of [KISS-ICP](https://github.com/PRBonn/kiss-icp), [MapClosures](https://github.com/PRBonn/kiss-icp), and [g2o](https://github.com/RainerKuemmerle/g2o).

## Contributing

We envision KISS-SLAM as a community-driven project. We love to see how the project is growing, thanks to the contributions from the community. We would love to see your face in the list below; open a Pull Request!

<a href="https://github.com/PRBonn/kiss-slam/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PRBonn/kiss-slam" />
</a>

## Contact Us
For questions or feedback:
- GitHub Issues: https://github.com/PRBonn/kiss-slam/issues
