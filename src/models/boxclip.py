import torch
import torch.nn as nn
from src.models.transformer import *

class BOXCLIP(nn.Module):
    def __init__(self, encoder, decoder, categories, device, clip_model, bbox_dim=4, latent_dim=512):
        super().__init__()
        self.categories = categories
        self.bbox_dim = bbox_dim
        self.latent_dim = latent_dim

        self.encoder = encoder
        self.decoder = decoder
        
        self.device = device
#         self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model = clip_model
        assert self.clip_model.training == False  # make sure clip is frozen
        
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        self.zeroshot_weights = None
        self.cats_map, text_descriptions = [], []
        
        for k, v in categories.items():
            self.cats_map.append(k)
            text_descriptions.append(v)
        self.cats_map = torch.tensor(self.cats_map)
            
        text_tokens = clip.tokenize(text_descriptions).to(device)
        self.zeroshot_weights = self.clip_model.encode_text(text_tokens).float() # (num_cats, 512)
        self.zeroshot_weights /= self.zeroshot_weights.norm(dim=-1, keepdim=True)
        
        
    def feat2cat(self, batch):
        
        cat_feats = batch['output_cat_feats'].clone()
        cat_feats /= cat_feats.norm(dim=-1, keepdim=True) # (bs, num_boxex, 512)
        
        similarity = cat_feats @ self.zeroshot_weights.T # (bs, num_boxex, num_cats)
        similarity = similarity.argmax(dim=-1)
        
        return {'output_cat': self.cats_map[similarity]}
    
        
    def compute_loss(self, batch):
        
        mixed_loss = 0.
        losses = {}
        bbox_mse = self.mse_loss(batch['bbox_feats'], batch['output_bbox'])
        cats_cos = self.cosine_sim(batch['cat_feats'], batch['output_cat_feats'])
        cats_cos = (1 - cats_cos).mean()

        mixed_loss = 10 * bbox_mse + cats_cos
        losses['bbox_mse'] = 10 * bbox_mse
        losses['cats_cos'] = cats_cos
        
        return mixed_loss, losses
        

    def forward(self, batch):
                
        bs, num_boxes = len(batch['bboxes']), len(batch['bboxes'][0])
        batch['bbox_feats'] = torch.zeros(bs, num_boxes, self.bbox_dim).to(batch['images'].device)
        batch['cat_feats'] = torch.zeros(bs, num_boxes, self.latent_dim).to(batch['images'].device)
        for i in range(bs):
            for j in range(num_boxes):
                batch['bbox_feats'][i][j] = torch.tensor(batch['bboxes'][i][j][0])
                
                cat_token = clip.tokenize(self.categories[batch['bboxes'][i][j][1]]).to(self.device)
#                 print(self.categories[batch['bboxes'][i][j][1]])
                with torch.no_grad():
                    batch['cat_feats'][i][j] = self.clip_model.encode_text(cat_token)
                    
        batch.update(self.encoder(batch))
        batch["z"] = batch["mu"]
        batch.update(self.decoder(batch))
        
        batch.update(self.feat2cat(batch))
        
        return batch
    