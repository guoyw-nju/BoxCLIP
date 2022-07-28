from src.models.transformer import *
from src.models.boxclip import BOXCLIP

def get_model(parameters, clip_model):

    encoder = Encoder_TRANSFORMER(num_layers=4)
    decoder = Decoder_TRANSFORMER(num_layers=4)
    model = BOXCLIP(encoder=encoder, decoder=decoder, clip_model=clip_model, 
                    device=parameters['device']).to(device=parameters['device'])

    return model

