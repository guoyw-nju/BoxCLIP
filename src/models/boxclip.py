import torch
import torch.nn as nn
from src.models.transformer import *

class BOXCLIP(nn.Module):
    def __init__(self, encoder, decoder, device, clip_model, bbox_dim=4, latent_dim=512):
        super().__init__()
        # self.categories = categories
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
        
        # self.zeroshot_weights = None
        # self.cats_map, text_descriptions = [], []
        
        # for k, v in categories.items():
        #     self.cats_map.append(k)
        #     text_descriptions.append(v)
        # self.cats_map = torch.tensor(self.cats_map)
            
        # text_tokens = clip.tokenize(text_descriptions).to(device)
        # self.zeroshot_weights = self.clip_model.encode_text(text_tokens).float() # (num_cats, 512)
        # self.zeroshot_weights /= self.zeroshot_weights.norm(dim=-1, keepdim=True)
        
        
    def feat2cat(self, batch):
        
        cat_feats = batch['output_cat_feats'].clone()
        cat_feats /= cat_feats.norm(dim=-1, keepdim=True) # (bs, num_boxex, 512)
        
        similarity = cat_feats @ self.zeroshot_weights.T # (bs, num_boxex, num_cats)
        similarity = similarity.argmax(dim=-1)
        
        return {'output_cat': self.cats_map[similarity]}
    
        
    def compute_loss(self, batch):
        mixed_loss = 0.
        losses = {}

        bbox_mse = self.mse_loss(batch['bboxs'], batch['output_bboxs'])
        cats_cos = self.cosine_sim(batch['cat_feats'], batch['output_cat_feats'])
        cats_cos = (1 - cats_cos).mean()

        mixed_loss = bbox_mse + cats_cos
        losses['bbox_mse'] = bbox_mse
        losses['cats_cos'] = cats_cos

        mixed_clip_loss, clip_losses = self.compute_clip_losses(batch)

        mixed_loss += mixed_clip_loss
        losses.update(clip_losses)
        losses.update({'mixed_loss': mixed_loss})

        return mixed_loss, losses
        
        
    def compute_clip_losses(self, batch):
        mixed_clip_loss = 0.
        clip_losses = {}

        for d in ('image', 'text'):
            with torch.no_grad():
                if d == 'image':
                    d_features = self.clip_model.encode_image(batch['clip_images']).float() # preprocess done in collate_fn
                elif d == 'text':
                    d_features = None
                    for tx in batch['clip_texts']:
                        texts = clip.tokenize(tx).to(self.device)
                        tx_feature = self.clip_model.encode_text(texts).mean(dim=0, keepdim=True) # (1, 512)

                        d_features = tx_feature if d_features == None else torch.cat((d_features, tx_feature), dim=0)

            bbox_features = batch['z']
            cos = self.cosine_sim(d_features, bbox_features)
            cosine_loss = (1 - cos).mean()
            clip_losses[f'{d}_cos'] = cosine_loss.item()
            mixed_clip_loss += cosine_loss

        return mixed_clip_loss, clip_losses
        

    def generate(self, clip_features):
        masks = torch.tensor([[True, True]]).to(self.device)
        batch = {
            'z': clip_features, # (bs, 512)
            'masks': masks
        }

        batch = self.decoder(batch)
        return batch


    def forward(self, batch):
        
        bs, num_boxs, _ = batch['bboxs'].shape
        batch['cat_feats'] = torch.zeros(bs, num_boxs, self.latent_dim).to(self.device)
        for i in range(bs):
            cat_token = clip.tokenize(batch['cat_name'][i]).to(self.device)
            batch['cat_feats'][i] = self.clip_model.encode_text(cat_token)
                    
        batch.update(self.encoder(batch))
        batch["z"] = batch["mu"]
        batch.update(self.decoder(batch))
        
        # batch.update(self.feat2cat(batch))
        
        return batch
    