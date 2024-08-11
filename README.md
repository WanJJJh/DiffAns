# Conditional Diffusion Model for Open-ended Video Question Answering
This is the implementation of paper Conditional Diffusion Model for Open-ended Video Question Answering.

# Method
Open-ended VideoQA presents a significant challenge due to the absence of fixed options, requiring the identification of the correct answer from a vast pool of candidate answers. Previous approaches typically utilize classifier or similarity comparison on fusion feature to yield prediction directly, lacking coarse-to-fine filtering on numerous candidates. Gradual refining the probability distribution of candidates can achieve more precise prediction. Thus, we propose the DiffAns model, which integrates the diffusion model to handle open-ended VideoQA task, simulating the gradual process by which humans answer open-ended question. Specifically, we first diffuse the true answer label into a random distribution (forward process). And under the guidance of answer-aware condition generated from video and question, the model iteratively denoises to obtain the correct probability distribution (backward process). This equips the model with the capability to progressively refine the random probability distribution of candidates, ultimately predicting the correct answer. We conduct experiments on three challenging open-ended VideoQA datasets, surpassing existing SoTA methods. Extensive experiments further explore and analyse the impact of each modules, as well as the design of diffusion model, demonstrating the effectiveness of DiffAns.
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
