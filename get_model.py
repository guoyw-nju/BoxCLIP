from transformer import *
from boxclip import BOXCLIP

def get_model(parameters, clip_model, categories):

    encoder = Encoder_TRANSFORMER(num_layers=4)
    decoder = Decoder_TRANSFORMER(num_layers=4)
    model = BOXCLIP(encoder=encoder, decoder=decoder, clip_model=clip_model, 
                    categories=categories, device=parameters['device']).to(device=parameters['device'])


    return model

