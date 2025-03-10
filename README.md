# AICT
This is the official repository of "High-Resolution Image Harmonization with Adaptive-Interval Color Transformation".
- [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/192956d4857000578f626c5193b34419-Abstract-Conference.html)

## Requirements

To train our model, a GPU with CUDA support is required.

## Environment

We built the code using Python 3.9 on Linux with NVIDIA GPUs and CUDA 11.6. The required packages can be installed using the `requirements.txt` file.

```
pip install -r requirements.txt
```
We utilize the [iHarmony4](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4) dataset for both training and testing. To use the dataset, the directory path must be updated in `config.yaml` and `config_test_FR.yml`. 
Since some images in the dataset have extremely high resolutions, we resize the HAdobe5k subset so that its largest dimension does not exceed 2048 pixels using `./notebooks/resize_dataset`.

## Training

To start training, simply run the shell file. 

```
runs/train_AICT.sh
```

## Testing

Our pretrained models are available in `weights`.
To evaluate our model, simply set `pretrain_path` in `runs/test_AICT.sh` and execute the following command:

```
runs/test_AICT.sh
```

## Citation
If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@InProceedings{meng2024high,
    author    = {Meng, Quanling and Qinglin, Liu and Li, Zonglin and Lan, Xiangyuan and Zhang, Shengping and Nie, Liqiang},
    title     = {High-Resolution Image Harmonization with Adaptive-Interval Color Transformation},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    volume    = {37},
    year      = {2024},
    pages     = {13769--13793}
}
```

