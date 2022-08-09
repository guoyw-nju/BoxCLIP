import torch
from copy import deepcopy

def eval_loss(dataloader, model, device):
    dict_loss = {}
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            for k, v in batch.items():
                if torch.is_tensor(batch[k]): batch[k] = batch[k].to(device)
                
            batch.update(model(batch))
            _, losses = model.compute_loss(batch)
            if i == 0:
                dict_loss = deepcopy(losses)
            else:
                for k in dict_loss.keys():
                    dict_loss[k] += losses[k]
        for k in dict_loss.keys():
            dict_loss[k] /= len(dataloader)
    
    dict_loss = {'val_'+k: v for k, v in dict_loss.items()}
    return dict_loss