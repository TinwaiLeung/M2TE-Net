# M2TE-Net

**Multimodal Multi-Scale Temporal Enhancement Network for Audio-Visual Scene Classification**

Ming Yin, Tianwei Liang, Xianxian Zeng, and Daming Shi  
*IEEE Transactions on Audio, Speech and Language Processing*, vol. 34, 2026.  
[Paper (DOI)](https://doi.org/10.1109/TASLPRO.2026.3678801)

M2TE-Net combines multi-scale temporal modeling with hierarchical audio-visual interaction for video-level classification. This repository contains the implementation linked in the paper.

## Method

- **MSRM:** multi-scale residual temporal modeling with dilated convolutions and self-attention.
- **LTEM:** masked local temporal attention to capture nearby temporal dynamics.
- **UMIM:** U-shape multimodal interaction with hybrid attention to combine complementary audio and visual information.

Audio features combine Wavegram and Log-Mel representations. Visual features are processed by the ResNet50 or CLIP implementation.

## Paper-reported results

| Dataset | Visual encoder in paper | Accuracy (%) |
| --- | --- | ---: |
| TAU | ResNet50 | 94.6 |
| TAU | CLIP/ResNet50 | 95.4 |
| AVE | ResNet50 | 93.3 |

These values are transcribed from Section IV-C of the paper, not newly reproduced measurements. The paper reports averages over five independent seeds. See [reproduction notes](docs/REPRODUCIBILITY.md) for differences between the published settings and this code release.

## Repository layout

```text
Dataset/                   # Split lists and class labels; media are not included
  TAU/                     # 10 scene categories
  AVE/                     # 28 event categories
train/
  train_with_ResNet50.py    # ResNet50 training entry point
  train_with_CLIP.py        # CLIP training entry point
  UMITENet_with_*.py        # Network implementations (historical filenames)
  CrossModal_Interaction_Module_V2.py
  dataloader/               # Audio and extracted-frame loaders
requirements.txt           # Original exported Windows environment
requirements_torch.txt     # Original exported Linux Conda environment
docs/REPRODUCIBILITY.md     # Setup, data layout, known differences and checklist
```

## Getting started

The repository is a research release and still requires environment and path configuration. Full paper reproduction has not been verified for this revision.

1. Clone the repository:

   ```bash
   git clone https://github.com/TinwaiLeung/M2TE-Net.git
   cd M2TE-Net
   ```

2. Prepare a CUDA-enabled PyTorch environment. Read the [environment notes](docs/REPRODUCIBILITY.md#environment) before using either exported requirements file; neither is a portable pip requirements file.
3. Obtain the datasets and extract video frames as described in the [data preparation guide](docs/REPRODUCIBILITY.md#data-preparation).
4. Configure paths in the selected training script and loader. The released scripts currently select **AVE**; use 28 classes for AVE and 10 for TAU. Switching to TAU requires changing all three dataset constructors and the classifier class count.
5. From the training directory, run one entry point:

   ```bash
   cd train
   python train_with_ResNet50.py --n_epoch 100 --batch_size 32
   # Or, after configuring the CLIP checkpoint:
   python train_with_CLIP.py --n_epoch 100 --batch_size 32
   ```

Both scripts require CUDA. Training selects the best checkpoint using the validation split. A separate held-out test evaluation is still needed after training; these commands do not report a final test score.

## Citation

```bibtex
@article{yin2026m2tenet,
  author  = {Yin, Ming and Liang, Tianwei and Zeng, Xianxian and Shi, Daming},
  title   = {Multimodal Multi-Scale Temporal Enhancement Network for Audio-Visual Scene Classification},
  journal = {IEEE Transactions on Audio, Speech and Language Processing},
  volume  = {34},
  year    = {2026},
  doi     = {10.1109/TASLPRO.2026.3678801}
}
```

## Reuse and contributions

No code license has been declared in this repository. Please contact the authors about reuse terms. Dataset and pretrained-model terms apply independently. The publisher PDF is not redistributed here; use the DOI link above.

For bug reports, include the entry point, dataset split, Python/PyTorch/CUDA versions, configuration, and full traceback. Exclude private paths and dataset media.
