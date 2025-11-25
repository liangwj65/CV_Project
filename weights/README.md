# Weights Directory

This directory is used to store pretrained model weights.

## Required Files

- `spai.pth`: SPAI pretrained weights
- `effort_clip_l14.pth`: Effort CLIP ViT-L/14 pretrained weights

## Usage

The training and testing scripts will automatically look for weights in this directory.
If weights are not found here, they will also check the parent directory's weights folder.

