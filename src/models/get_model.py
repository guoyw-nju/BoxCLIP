from src.models.transformer import *
from src.models.boxclip import BOXCLIP

def get_model(parameters, clip_model):

    encoder = Encoder_TRANSFORMER(num_layers=parameters['num_attentionLayer'])
    decoder = Decoder_TRANSFORMER(num_layers=parameters['num_attentionLayer'])
    model = BOXCLIP(encoder=encoder, decoder=decoder, clip_model=clip_model, device=parameters['device'], lambdas=parameters['lambdas']).to(device=parameters['device'])

    return model

