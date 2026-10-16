# Reproduction guide and release limitations

## Environment

The training code imports PyTorch, torchvision, torchaudio, NumPy, pandas, Pillow, librosa, scikit-learn, matplotlib, seaborn, tqdm, h5py and natsort. The CLIP path also imports the `clip` package. Use the OpenAI CLIP implementation, not an unrelated package with the same import name.

The two environment exports describe different historical machines:

- `requirements.txt` contains local `file:///C:/...` build paths and Conda management packages. `pip install -r requirements.txt` is not a portable installation procedure.
- `requirements_torch.txt` is a Linux Conda export with Python 3.7, PyTorch 1.2 and torchvision 0.4. The current source also uses newer APIs such as `InterpolationMode`, so this export is not evidence of a working environment for all current files.

A tested, complete environment lock is not available yet. Install mutually compatible PyTorch/torchvision/torchaudio builds for the target CUDA system, then validate the remaining dependencies in a clean environment. Do not assume the two exports can be combined. Dependency modernization requires a full forward/backward smoke test and training validation.

## Data preparation

### TAU Urban Audio-Visual Scenes 2021

Download the [development dataset](https://zenodo.org/records/4477542), including `TAU-urban-audio-visual-scenes-2021-development.audio.[1-8].zip` and `TAU-urban-audio-visual-scenes-2021-development.video.[1-16].zip`.

The loaders read extracted frames, not MP4 files. The paper uses 2 fps and 224 x 224 visual inputs. Preserve the same frame count and chronological order for each 10-second clip. The exact crop/augmentation pipeline should be confirmed against the paper before claiming reproduction.

```text
Dataset/TAU/
  train.csv
  val.csv
  test.csv
  label.csv
  audio/<clip-id>.wav
  video frame/<scene-class>/<clip-id>/<ordered-frame>.jpg
```

### AVE

Use the [dataset link supplied in the original README](https://drive.google.com/open?id=1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK) and the split lists already in `Dataset/AVE/`.

```text
Dataset/AVE/
  train.txt
  val.txt
  test.txt
  AVE_label.csv
  audio/<clip-id>.wav
  keyframe/<clip-id>/<ordered-frame>.jpg
```

The paper describes TAU splits of 7,907 / 739 / 3,645 samples and AVE splits of 3,339 / 402 / 402. Confirm local media IDs and split counts before training. Do not move examples between splits to compensate for missing media.

## Paths and checkpoints

Configure `dataset_root` and, for TAU, `audio_root` in the selected `train/dataloader/` file. The `classes` properties also contain absolute label CSV paths. Configure log CSV and model output paths in the entry point and create the log directory before running.

The CLIP entry point currently loads a local `ViT-B-16.pt` checkpoint. The paper reports CLIP/ResNet50, so the current script is not an exact implementation of that reported visual-backbone configuration. Do not substitute a backbone without checking output dimensions and the network's 512-dimensional input assumptions. The custom ResNet50 implementation downloads its pretrained weights through `torch.hub` when requested.

## Paper settings versus released code

| Setting | Paper, Section IV-B | Released source |
| --- | --- | --- |
| Optimizer | Adam, LR 1e-4, weight decay 1e-4 | Same initial values |
| Batch / epochs | 32 / 100 | 32 / 50 by default; pass `--n_epoch 100` |
| LR schedule | Multiply by 0.2 at epochs 3, 30, 50 | CLIP: 0.1 at 3, 30; ResNet50: scheduler step commented out |
| MSRM dilation | 3, 5, 7 | 1, 2, 5 in the network source |
| Visual CLIP backbone | CLIP/ResNet50 | Local ViT-B/16 checkpoint in entry point |
| Audio encoder initialization | Pretrained ResNet18 described | `resnet18()` uses default `pretrained=False` |
| Validation | Separate validation split | Corrected to use `val_loader` in this revision |

These differences are recorded rather than silently changing model architecture or optimization settings. Published scores must not be attributed to an unvalidated revised configuration.

## Training fixes in this revision

- Validation and best-checkpoint selection use the validation loader, leaving test samples out of model selection.
- The active ResNet50 AVE configuration uses 28 output classes.
- The CLIP checkpoint filename has the correct number of format arguments.
- Checkpoint tensors are copied to CPU without moving the live model off its optimizer device.
- Log loss receives the complete classifier label range, including classes absent from an individual evaluation split.

## Verification checklist

- [ ] Establish and record a working clean environment, GPU and CUDA versions.
- [ ] Verify local data paths, frame counts, sample rates and label mappings.
- [ ] Resolve the paper/code configuration differences above with the authors.
- [ ] Run one batch through each selected backbone and check finite loss and gradients.
- [ ] Train with validation-only checkpoint selection.
- [ ] Reload the selected checkpoint and evaluate the held-out test split once.
- [ ] Repeat with five recorded seeds and report mean and variability.

This revision has static syntax and regression checks only. No dataset/GPU training or paper-score reproduction has been performed.
