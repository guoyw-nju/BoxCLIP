import sys
sys.path.append('.')

import torch
from src.utils.collate_fn_coco import collate_fn_coco
from src.utils.get_model_and_data import get_model_and_data
from tqdm import tqdm
import time
import os
from src.parser.training import parser

from torch.utils.tensorboard import SummaryWriter


def do_epoch(model, datasets, parameters, optimizer, start_epoch=1):

    # config dataloader
    dataloader = torch.utils.data.DataLoader(datasets['train'], batch_size=parameters['batch_size'], shuffle=True, collate_fn=collate_fn_coco)
    num_batch = len(dataloader)

    model.train()

    os.makedirs(parameters['exp_path'], exist_ok=True)

    writer = SummaryWriter(log_dir=parameters['exp_path'])    

    print('Start training...')
    for eps in range(start_epoch, start_epoch + parameters['num_epochs']):

        for i, batch in enumerate(tqdm(dataloader, desc=f'Epoch {eps}')):
            
            for k in batch.keys():
                if torch.is_tensor(batch[k]): batch[k] = batch[k].to(parameters['device'])

            optimizer.zero_grad()
            batch = model(batch)
            mixed_loss, losses = model.compute_loss(batch)

            mixed_loss.backward()
            optimizer.step()

            writer.add_scalars(f"Loss/Iters", losses, (eps-1)*num_batch + i)

        writer.add_scalar(f"Lr/Epochs", scheduler.get_last_lr()[0], eps)
        scheduler.step()

        writer.flush()

        checkpoint = {
            'epoch': eps,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict()
        }
        # for nms in os.listdir(parameters['checkpoint_path']):
        #     f_name = os.path.join(parameters['checkpoint_path'], nms)
        #     if f_name.endswith('.pth.tar'): os.remove(f_name)

        if (eps % 5) == 0 or eps == start_epoch+parameters['num_epochs']-1:
            torch.save(checkpoint, os.path.join(parameters['exp_path'], f'checkpoint-epoch{eps}.pth.tar'))

    
if __name__ == '__main__':

    parameters = parser()

    # try device
    if 'device' not in parameters:
        parameters['device'] = "cuda" if torch.cuda.is_available else "cpu"        

    model, datasets = get_model_and_data(parameters)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print('Trainable params: %.2fM' % (sum(p.numel() for _, p in model.named_parameters() if p.requires_grad) / 1000000.0))

    if parameters['overfit']:
        overfit_set = [datasets['train'][i] for i in range(parameters['overfit_size'])]
        parameters['batch_size'] = parameters['overfit_size']
        datasets['train'] = overfit_set
        print(f"Overfit on set scale of {len(datasets['train'])}")

    # training setting
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters['lr'])

    # lr scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=parameters['lr_step_size'], gamma=parameters['lr_gamma'], verbose=True)

    if parameters['checkpoint_path'] != None:
        checkpoint = torch.load(parameters['checkpoint_path'])

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Model loaded from checkpoint {parameters['checkpoint_path']}...")

    else:
        start_epoch = 1
        print('Train on new model...')
    
    # print(f'Init lr: {scheduler.get_last_lr()[0]}')
    do_epoch(model=model, datasets=datasets, parameters=parameters, optimizer=optimizer, start_epoch=start_epoch)

