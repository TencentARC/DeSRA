# DeSRA (ICML 2023)

🚩 Updates

- ✅ The collected datasets, the codes of detecting artifacts and calculating metrics are released.

---

This paper aims at **dealing with GAN-inference artifacts**.
<br>

We design a method to effectively detect regions with GAN-inference artifacts, and further propose a fine-tuning strategy that only requires a small number of artifact
images to eliminate the same kinds of artifacts, which bridges the gap of applying SR algorithms to practical scenarios.

---

### :book: DeSRA: Detect and Delete the Artifacts of GAN-based Real-World Super-Resolution Models

> [[Paper](https://openreview.net/pdf?id=M0bwbIl4Bl)] &emsp; [Project Page] &emsp; [[Video](https://recorder-v3.slideslive.com/#/share?share=82996&s=e6ebdd07-a83b-4f4b-8eab-a5f103c6c46b)] &emsp; [B站] &emsp; [[Poster](https://docs.google.com/presentation/d/18-kVUBRgGKF4JUrN253yURJGDKcFNpaB/edit?usp=drive_web&ouid=113023682396793851067&rtpof=true)] &emsp; [[PPT slides](https://docs.google.com/presentation/d/15zGKWNd6vPuGI-dMf0ZsGrfMnPhXO-8s/edit?rtpof=true)]<br>
> [Liangbin Xie*](https://liangbinxie.github.io/), [Xintao Wang*](https://xinntao.github.io/), [Xiangyu Chen*](https://chxy95.github.io/), [Gen Li](https://scholar.google.com/citations?user=jBxlX7oAAAAJ&hl=en), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en), [Jiantao Zhou](https://www.fst.um.edu.mo/personal/jtzhou/), [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ) <br>
> Tencent ARC Lab; University of Macau; Shenzhen Institutes of Advanced Technology; Shanghai AI Lab

<p align="center">
  <img src="./assets/DeSRA_teasor.jpg", height=400>
</p>

---

## 🔧 Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)
- Option: NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Option: Linux

### Installation

1. Install [mmsegmentation](<https://github.com/open-mmlab/mmsegmentation>) package and install dependent packages. **Note**: The version of mmsegmentation and mmcv-full that used in the experiment are <span style="color:red">0.29.0</span> and <span style="color:red">1.6.1</span>, respectively. Setting up the environment might take some time.

    ```bash
    git clone https://github.com/open-mmlab/mmsegmentation.git
    cd mmsegmentation
    pip install -r requirements.txt
    ```

2. Clone repo and move the provided scripts into the demo folder (a subfolder) in mmsegmentation folder.

    ```bash
    git clone https://github.com/TencentARC/DeSRA
    cd DeSRA
    mv scripts/* mmsegmentation/demo (you need to modify the path)
    ```

If you encounter problem, I also provide the environment that I used in the experiments. You can refer to the [requirements.txt](./requirements.txt)

---

## 📦 Testing datasets

For three representative methods: [RealESRGAN](<https://github.com/xinntao/Real-ESRGAN#-demos-videos>), [LDL](<https://github.com/csjliang/LDL>) and [SwinIR](<https://github.com/JingyunLiang/SwinIR>), we
choose nearly 200 representative images with GAN-inference artifacts to construct this GAN-SR artifact
dataset. You can download from [GoogleDrive](<https://drive.google.com/drive/folders/1jPTvXq_uJvpOaP5uCZ6unmb13Gt2naVC?usp=sharing>) and [BaiduDisk](<https://pan.baidu.com/s/1rwwEpATlPo9RFzTv6D7lBw?pwd=DGLF>). (For each methods, we provide the MSE-SR, GAN-SR, DeSRA-Mask, LR, and human-labeled GT-Mask)

---

## ⚔️ Quick Inference

1. Detect the artifacts between the MSE-SR results and GAN-SR results. We store many intermediate results and the final detected binary artifact map are stored in *Final_Artifact_Map* folder.

    ```bash
    python demo/artifact_detection.py --mse_root="./LDL/MSE_types" --gan_root="./LDL/GAN_types" --save_root="./results/LDL/DeSRA"
    ```

2. Evaluate the performance. As mentioned in our paper, we provide three scripts to calculate **IOU**, **Precision** and **Recall**, respectively. You can find these scripts in *metrics* folder.

    ```bash
    python metrics/calc_iou.py
    python metrics/calc_precision.py
    python metrics/calc_recall.py
    ```

---

## 📜 License and Acknowledgement

DeSRA is released under Apache License Version 2.0.

## BibTeX

    @article{xie2023desra,
        title={DeSRA: Detect and Delete the Artifacts of GAN-based Real-World Super-Resolution Models},
        author={Xie, Liangbin and Wang, Xintao and Chen, Xiangyu and Li, Gen and Shan, Ying and Zhou, Jiantao and Dong, Chao},
        year={2023}
    }

## 📧 Contact

If you have any question, please email `lb.xie@siat.ac.cn`.
