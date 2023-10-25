**Spiking-Diffusion: Vector Quantized Discrete Diffusion Model with Spiking Neural Networks** [https://arxiv.org/abs/2308.10187]

### Abstract

Spiking neural networks (SNNs) have tremendous potential for energy-efficient neuromorphic chips due to their binary and event-driven architecture. SNNs have been primarily used in classification tasks, but limited exploration on image generation tasks. To fill the gap, we propose a Spiking-Diffusion model, which is based on the vector quantized discrete diffusion model. First, we develop a vector quantized variational autoencoder with SNNs (VQ-SVAE) to learn a discrete latent space for images. In VQ-SVAE, image features are encoded using both the spike firing rate and postsynaptic potential, and an adaptive spike generator is designed to restore embedding features in the form of spike trains. Next, we perform absorbing state diffusion in the discrete latent space and construct a spiking diffusion image decoder (SDID) with SNNs to denoise the image. Our work is the first to build the diffusion model entirely from SNN layers. Experimental results on MNIST, FMNIST, KMNIST, Letters, and Cifar10 demonstrate that Spiking-Diffusion outperforms the existing SNN-based generation model. We achieve FIDs of 37.50, 91.98, 59.23, 67.41, and 120.5 on the above datasets respectively, with reductions of 58.60%, 18.75%, 64.51%, 29.75%, and 44.88% in FIDs compared with the state-of-art work.

### Prepare

1. unzip spikingjelly.zip
2. prepare a suitable conda environment 

### Training

```python
python -u main.py --dataset_name MNIST --model snn-vq-vae
```

### Testing

```python
python -u main.py --dataset_name MNIST --model snn-vq-vae --checkpoint [path to VQ-SVAE, such as:/data/liumingxuan/Spiking-Diffusion/result/MNIST/snn-vq-vae/model.pth]
```



### Contact Information

```
@misc{liu2023spikingdiffusion,
      title={Spiking-Diffusion: Vector Quantized Discrete Diffusion Model with Spiking Neural Networks}, 
      author={Mingxuan Liu and Jie Gan and Rui Wen and Tao Li and Yongli Chen and Hong Chen},
      year={2023},
      eprint={2308.10187},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
```





