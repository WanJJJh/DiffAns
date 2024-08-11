# Conditional Diffusion Model for Open-ended Video Question Answering
This is the implementation of paper Conditional Diffusion Model for Open-ended Video Question Answering.

# Method


# Overview
## Datasets
TGIF-FrameQA: https://github.com/YunseokJANG/tgif-qa

MSVD-QA and MSRVTT-QA: https://github.com/xudejing/video-question-answering
## Backbones
The parameters of backbones are from huggingface.

ViT: https://huggingface.co/openai/clip-vit-large-patch14

RoBERTa: https://huggingface.co/FacebookAI/roberta-base
## Traning Scripts
The traning scripts are in directory: "training_sh/"

# Acknowledgments
Thanks for the following open source works and some codes are borrowed from them:

Scalable Diffusion Models with Transformers: 
https://github.com/facebookresearch/DiT

MomentDiff: Generative Video Moment Retrieval from Random to Real: 
https://github.com/IMCCretrieval/MomentDiff

Multi-Scale Progressive Attention Network for Video Question Answering: 
https://github.com/gzcsudo/MSPAN-VideoQA
