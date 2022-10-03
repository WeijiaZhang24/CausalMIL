# Multi-Instance Causal Representation Learning for Instance Label Prediction and Out-of-Distribution Generalization
## Weijia Zhang, Xuanhui Zhang, Han-Wen Deng, Min-Ling Zhang

Paper to appear in NeurIPS 2022.

For questions regarding the code, please contact weijia.zhang.xh@gmail.com

Requirements: PyTorch 1.12

### To reproduce the results in the paper:

#### For MNIST, FashionMNIST, KuzushijiMNIST multi-instance bags, please use MNIST_bags.ipynb for training, testing and visulization.


#### For Out-of-Distribution (OOD) generalization results, please use ColorMNIST_OOD.ipynb for training, testing and visulization.


#### For Colon Cancer results, please use colon_cancer.py. 
This piece of code provides a dataloader for processing MIL bags organized as folders of images.

The dataset can be obtained (credit to Dr. Jiawen Yao) from https://drive.google.com/file/d/1RcNlwg0TwaZoaFO0uMXHFtAo_DCVPE6z/view?usp=sharing
