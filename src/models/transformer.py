import torch
import torch.nn as nn
import numpy as np
import clip


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) # (5000, 512)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (5000, 1)
#         print('Position shape: {}'.format(position.shape))
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
#         print('pe shape: {}'.format(pe.shape))
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

    
class Encoder_TRANSFORMER(nn.Module):
    def __init__(self, bbox_dim=4, latent_dim=512, num_layers=4):
        super().__init__()
        self.bbox_dim = bbox_dim
        self.latent_dim = latent_dim
        
        self.num_layers = num_layers
        
        self.muQuery = nn.Parameter(torch.randn(1, self.latent_dim))
        self.sigmaQuery = nn.Parameter(torch.randn(1, self.latent_dim))        
        
        # input embedding
        self.bbox_embedding = nn.Linear(self.bbox_dim, self.latent_dim)
        self.cats_embedding = nn.Linear(self.latent_dim, self.latent_dim)
        self.input_embedding = nn.Sequential(
            nn.Linear(self.bbox_dim + self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        # position encoder
        self.pos_encoder = PositionalEncoding(d_model=self.latent_dim)
        # Encoder: the same configuration as MotionCLIP
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim, 
                                                        nhead=4, 
                                                        dim_feedforward=1024, 
                                                        dropout=0.1, 
                                                        activation='gelu')
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        
    def forward(self, batch):
         # mask: (bs, num_boxes)
        bboxs, cat_feats, masks = batch['bboxs'], batch['cat_feats'], batch['masks']
        bs = bboxs.shape[0]
        
        bboxs = bboxs.permute(1, 0, 2) # (num_boxes, bs, bbox_dim)
        cat_feats = cat_feats.permute(1, 0, 2) # (num_boxes, bs, latent_dim)

        # bboxs = self.bbox_embedding(bboxs) # (num_boxes, bs, latent_dim)
        # cat_feats = self.cats_embedding(cat_feats) # (bs, num_boxes, latent_dim)
        # x = bboxs + cat_feats

        x = torch.cat((bboxs, cat_feats), axis=-1) # (num_boxes, bs, 4 + 512)
        x = self.input_embedding(x) # (num_boxes, bs, 512)
        
        xseq = torch.cat((self.muQuery.repeat(bs, 1)[None], 
                          self.sigmaQuery.repeat(bs, 1)[None], x), axis=0)
        xseq = self.pos_encoder(xseq)

        mu_and_sigma_mask = torch.ones((bs, 2), dtype=bool, device=x.device) # (bs, 2)
        
#         print('mu_and_sigma_mask: {}\nmasks: {}'.format(mu_and_sigma_mask.shape, masks.shape))
        maskseq = torch.cat((mu_and_sigma_mask, masks), axis=1) # (bs, 2+num_boxes)
        
#         print('xseq: {}\nmaskseq: {}'.format(xseq.shape, maskseq.shape))
#         print('xseq: ', xseq)
        final = self.encoder(xseq, src_key_padding_mask=~maskseq) # (2+num_boxes, bs, 512)
        
        mu = final[0] # (bs, 512)
#         print(mu)
#         print('mu: {}'.format(mu.shape))

        return {"mu": mu}
    
    
class Decoder_TRANSFORMER(nn.Module):
    def __init__(self, bbox_dim=4, latent_dim=512, num_layers=4):
        super().__init__()
        
        self.bbox_dim = bbox_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # position encoder
        self.pos_encoder = PositionalEncoding(self.latent_dim)
        
        # Decoder: the same as MotionCLIP
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.latent_dim, 
                                                        nhead=4, 
                                                        dim_feedforward=1024, 
                                                        dropout=0.1, 
                                                        activation='gelu')
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.num_layers)

        # final layer
        self.bbox_embedding = nn.Linear(self.latent_dim, self.bbox_dim)
        self.cats_embedding = nn.Linear(self.latent_dim, self.latent_dim)
        
        
    def forward(self, batch):
        z, masks = batch['z'], batch['masks']
        
        bs, num_boxes = masks.shape
        
        z = z[None]
        
        queries = torch.zeros(num_boxes, bs, self.latent_dim, device=z.device)
        queries = self.pos_encoder(queries) # (num_boxes, bs, latent_dim)
        
        # (num_boxes, bs, latent_dim)
        output = self.decoder(tgt=queries, memory=z, tgt_key_padding_mask=~masks)
        
        bbox_feats = self.bbox_embedding(output)
        cats_feats = self.cats_embedding(output)
        
        bbox_feats[~masks.T] = 0 # mask.T: (num_boxes, bs)
        cats_feats[~masks.T] = 0
        
        # batch['output_bboxs'] = bbox_feats.permute(1, 0, 2) # (bs, num_boxes, 4)
        batch['output_bboxs'] = torch.sigmoid(bbox_feats.permute(1, 0, 2)) # (bs, num_boxes, 4)
        batch['output_cat_feats'] = cats_feats.permute(1, 0, 2) # (bs, num_boxex, 512)
        
        return batch