# A Cross-Attention Multi-Scale Performer with Gaussian Bit-Flips for File Fragment Classification
Authors: Sisung Liu*, Jeong Gyu Park*, Hyeongsik Kim, Je Hyeong Hong
( * : joint first authors)

IEEE Transactions on Information Forensics and Security (TIFS, IF 6.3, Categorical JCR < 8.7%)

## Abstract
File fragment classification is a crucial task in digital forensics and cybersecurity, and has recently achieved significant improvement through the deployment of convolutional neural networks (CNNs) compared to traditional handcrafted feature-based methods. However, CNN-based models exhibit inherent biases that can limit their effectiveness for larger datasets. To address this limitation, we propose the Cross-Attention Multi-Scale Performer (XMP) model, which integrates the attention mechanisms of transformer encoders with the feature extraction capabilities of CNNs. Compared to our conference work, we additionally introduce a novel Gaussian Bit-Flip (GBFlip) method for binary data augmentation, largely inspired by simulating real-world bit flipping errors, improving the model performance. Furthermore, we incorporate a fine-tuning approach and demonstrate XMP adapts more effectively to diverse datasets than other CNN-based competitors without extensive hyperparameter tuning. Our experimental results on two public file fragment classification datasets show XMP surpassing other CNN-based and RCNN-based models, achieving state-of-the-art performance in file fragment classification both with and without fine-tuning.

![image](./images/XMP_architecture.png)

## Prerequisites:
````
python=3.8
torch==1.10.0+cu111
torchvision==0.11.0+cu111
torchaudio==0.10.0
pytorch-fast-transformers
performer-pytorch
vit-pytorch
pykernel
wandb
tqdm
````
This code has been tested with Ubuntu 20.04, A6000 GPUs with CUDA 12.2, Python 3.8, Pytorch 1.10.

Earlier versions may also work~ :)

## 🏃 How to run our code!
### How to Train
To run our code, use the following command template, adjusting the hyperparameters as needed based on the scenario and dataset size. We have conducted extensive hyperparameter tuning for different scenarios and dataset sizes to ensure optimal performance. The details of these configurations are documented in the table. Use these values to modify the config.json file accordingly.

![image](./images/XMP_parameter.png)

Example command:
````
python train.py 
````

### How to Fine-tune
For fine-tuning, we provide pretrained models for the XMP and XMP+GBFlip architectures on the FFT75 scenario 1. You can access these pretrained files from our repository. 

- Full fine-tuning or linear probing: Set mode to org
- VPT: Set mode to vpt
- AdaptFormer: Set mode to adapt

Ensure you adjust other hyperparameters in config.json based on your specific use case and dataset requirements.

Example command:
````
python fine-tuning.py 
````

[XMP weight (512_scen1)](https://drive.google.com/file/d/1pEuiTjLMWueNjK2sr0VdZK9hteYBK83Z/view?usp=drive_link)

[XMP weight (4k_scen1)](https://drive.google.com/file/d/1gVfZ7Y2zi7ywHpTJMibystvYPAeTWx8f/view?usp=drive_link)

[XMP+GBFlip weight (512_scen1)](https://drive.google.com/file/d/1h_BeEQfPjSPC6kv9S7x749nRPlrNhXbg/view?usp=drive_link)

[XMP+GBFlip weight (4k_scen1)](https://drive.google.com/file/d/1dntU9YbGi0Sn4DtDOnMsWrsyl5fxIgS1/view?usp=drive_link)

## Citation
````
@inproceedings{park2024xmp,
  title={XMP: A Cross-Attention Multi-Scale Performer for File Fragment Classification},
  author={Park, Jeong Gyu and Liu, Sisung and Hong, Je Hyeong},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={4505--4509},
  year={2024},
  organization={IEEE}
}
````
## License
A patent application for XMP has been submitted and is under review for registration. XMP is licensed under the CC-BY-NC-SA-4.0 license limiting any commercial use.

## Acknowledgement
Our code is based on [performer](https://github.com/lucidrains/performer-pytorch) repository. We thank the authors for releasing their code. 
> This work was supported by the Korea Research Institute for defense Technology planning and advancement (KRIT) grant funded by the Korea government (DAPA (Defense Acquisition Program Administration)) (No. KRIT-CT-22-021, Space Signal Intelligence Research Laboratory, 2022).
