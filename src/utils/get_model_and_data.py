import clip
from src.models.transformer import *
from src.models.get_model import get_model
from src.datasets.get_datasets import get_datasets

def get_model_and_data(parameters):

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=parameters['device'])
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    datasets = get_datasets(parameters, clip_preprocess=clip_preprocess)
    model = get_model(parameters, clip_model)

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1e6))
    print('Trainable params: %.2fM' % (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6))

    return model, datasets

