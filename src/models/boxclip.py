import torch
import torch.nn as nn
import clip
from src.models.transformer import *

class BOXCLIP(nn.Module):
    def __init__(self, encoder, decoder, device, clip_model, lambdas, bbox_dim=4, latent_dim=512):
        super().__init__()
        # self.categories = categories
        self.bbox_dim = bbox_dim
        self.latent_dim = latent_dim

        self.lambdas = lambdas

        self.encoder = encoder
        self.decoder = decoder
        
        self.device = device

        self.clip_model = clip_model
        assert self.clip_model.training == False  # make sure clip is frozen
        
        self.mse_loss = nn.MSELoss(reduction='none')
        # self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.cosine_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
        
        # self.zeroshot_weights = None
        # self.cats_map, text_descriptions = [], []
        
        # for k, v in categories.items():
        #     self.cats_map.append(k)
        #     text_descriptions.append(v)
        # self.cats_map = torch.tensor(self.cats_map)
            
        # text_tokens = clip.tokenize(text_descriptions).to(device)
        # self.zeroshot_weights = self.clip_model.encode_text(text_tokens).float() # (num_cats, 512)
        # self.zeroshot_weights /= self.zeroshot_weights.norm(dim=-1, keepdim=True)
        
        
    def feat2cat(self, batch, cat_texts):
        
        tokens = clip.tokenize(cat_texts).to(self.device)
        zeroshot_weights = self.clip_model.encode_text(tokens).float()
        zeroshot_weights /= zeroshot_weights.norm(dim=-1, keepdim=True)
        
        cat_feats = batch['output_cat_feats'].clone()
        cat_feats /= cat_feats.norm(dim=-1, keepdim=True) # (bs, num_boxex, 512)
        
        similarity = cat_feats @ zeroshot_weights.T # (bs, num_boxex, num_cats)
        similarity = similarity.argmax(dim=-1)
        
        return similarity
        return {'output_cat': similarity}
    
        
    def compute_loss(self, batch):
        mixed_loss = 0.
        losses = {}

        # bbox_mse = 0.
        if False:
            bbox_mse += self.mse_loss(batch['bboxs'][:,:,:2], batch['output_bboxs'][:,:,:2]).sum(dim=-1).mean() # (bs, num_boxes, 2)
            # print(bbox_mse)
            bbox_mse += self.mse_loss(torch.sqrt(batch['bboxs'][:,:,2:]), torch.sqrt(batch['output_bboxs'][:,:,2:])).sum(dim=-1).mean()
            # bbox_mse /= batch['bboxs'].shape[0] * batch['bboxs'].shape[1]
            # print(f'bbox_mse: {bbox_mse}')
        else:
            bbox_mse = self.mse_loss(batch['bboxs'], batch['output_bboxs']).sum(dim=-1).mean()


        cats_cos = self.cosine_sim(batch['cat_feats'], batch['output_cat_feats']) # in: (bs, num_boxes, 512) out: (bs, num_boxes)
        # print(f'cats_cos shape: {cats_cos.shape}')
        cats_cos = (1 - cats_cos).mean()

        losses['bbox_mse'] = bbox_mse.item() * self.lambdas['bbox_mse']
        losses['cats_cos'] = cats_cos.item() * self.lambdas['cats_cos']
        mixed_loss += bbox_mse * self.lambdas['bbox_mse'] + cats_cos * self.lambdas['cats_cos']

        # from text encoder to decoder
        # texts = clip.tokenize([t[0] for t in batch['clip_texts']]).to(self.device)
        # texts_feats = self.clip_model.encode_text(texts).float() # (bs, 512)

        # batch_gen = self.generate(texts_feats)
        # bbox_mse_gen = 0.
        # print(f"batch_bbox shape: {batch['bboxs'].shape}, batch_gen_bbox shape: {batch_gen['output_bboxs'].shape}")

        # bbox_mse_gen += self.mse_loss(batch['bboxs'][:,:,:2], batch_gen['output_bboxs'][:,:,:2]).sum(dim=-1).mean()
        # bbox_mse_gen += self.mse_loss(torch.sqrt(batch['bboxs'][:,:,2:]), torch.sqrt(batch_gen['output_bboxs'][:,:,2:])).sum(dim=-1).mean()
        # # bbox_mse_gen /= batch['bboxs'].shape[0]

        # cats_cos_gen = self.cosine_sim(batch['cat_feats'], batch_gen['output_cat_feats'])
        # cats_cos_gen = (1 - cats_cos_gen).sum(dim=-1).mean()
        
        # losses['bbox_mse_gen'] = bbox_mse_gen.item() * self.lambdas['bbox_mse_gen']
        # losses['cats_cos_gen'] = cats_cos_gen.item() * self.lambdas['cats_cos_gen']
        # mixed_loss += bbox_mse_gen * self.lambdas['bbox_mse_gen'] + cats_cos_gen * self.lambdas['cats_cos_gen']


        mixed_clip_loss, clip_losses = self.compute_clip_losses(batch)
        mixed_loss += mixed_clip_loss
        losses.update(clip_losses)
        losses.update({'mixed_loss': mixed_loss.item()})

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
                        # texts = clip.tokenize(tx).to(self.device)
                        texts = clip.tokenize(tx[0]).to(self.device)
                        tx_feature = self.clip_model.encode_text(texts).mean(dim=0, keepdim=True) # (1, 512)

                        d_features = tx_feature if d_features == None else torch.cat((d_features, tx_feature), dim=0)

            bbox_features = batch['z']
            cos = self.cosine_sim(d_features, bbox_features)
            cosine_loss = (1 - cos).mean()
            clip_losses[f'clip_{d}_cosine'] = cosine_loss.item() * self.lambdas[f'clip_{d}_cosine']
            mixed_clip_loss += cosine_loss * self.lambdas[f'clip_{d}_cosine']

        return mixed_clip_loss, clip_losses
        

    def generate(self, clip_features):
        masks = torch.tensor([[True, True]] * clip_features.shape[0]).to(self.device)
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
    
