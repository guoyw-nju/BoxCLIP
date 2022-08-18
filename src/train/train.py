from copy import deepcopy
import sys
sys.path.append('.')

import torch
from src.utils.collate_fn_coco import collate_fn_coco
from src.utils.get_model_and_data import get_model_and_data
from tqdm import tqdm
import os
from src.parser.training import parser
from src.utils.eval_loss import eval_loss
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter


def do_epoch(model, datasets, parameters, optimizer, scheduler, start_epoch=1):

    # config dataloader
    train_loader = torch.utils.data.DataLoader(datasets['train'], batch_size=parameters['batch_size'], shuffle=True, collate_fn=collate_fn_coco)
    val_loader = torch.utils.data.DataLoader(datasets['val'], batch_size=20, collate_fn=collate_fn_coco)
    num_batch = len(train_loader)

    model.train()

    os.makedirs(parameters['exp_path'], exist_ok=True)

    writer = SummaryWriter(log_dir=parameters['exp_path'])    

    print('Start training...')
    for eps in range(start_epoch, start_epoch + parameters['num_epochs']):
        epoch_dict_loss = {}

        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch {eps}')):
            
            for k in batch.keys():
                if torch.is_tensor(batch[k]): batch[k] = batch[k].to(parameters['device'])

            optimizer.zero_grad()
            batch = model(batch)
            mixed_loss, losses = model.compute_loss(batch)

            mixed_loss.backward()
            optimizer.step()
            
            if i == 0:
                epoch_dict_loss = deepcopy(losses)
            else:
                for k in epoch_dict_loss.keys():
                    epoch_dict_loss[k] += losses[k]

            if (i % 10 == 0) or (i == num_batch-1):
                writer.add_scalars(f"Loss/Iters", losses, (eps-1)*num_batch + i)
        
        for k in epoch_dict_loss.keys():
            epoch_dict_loss[k] /= len(train_loader)
        writer.add_scalars("Loss/Epoch", epoch_dict_loss, eps)

        if scheduler != None:
            # writer.add_scalar(f"Lr/Epochs", scheduler.get_last_lr()[0], eps)
            writer.add_scalar(f"Lr/Epochs", optimizer.state_dict()['param_groups'][0]['lr'], eps)
            # optimizer.state_dict()['param_groups'][0]['lr']
        writer.flush()

        # update learning rate
        if scheduler != None: scheduler.step(epoch_dict_loss['mixed_loss'])

        # draw loss curve on val set
        if (eps % 5) == 0 or eps == start_epoch+parameters['num_epochs']-1:
            if not parameters['no_val_loss']:
                val_losses = eval_loss(val_loader, model, parameters['device'])
                writer.add_scalars("Loss/Iters", val_losses, (eps-1)*num_batch + i)

        # save checkpoint
        if (eps % parameters['checkpoint_step']) == 0 or eps == start_epoch+parameters['num_epochs']-1:

            checkpoint = {
                'epoch': eps,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'lr_scheduler': scheduler.state_dict()
            }
            if scheduler != None: checkpoint.update({'lr_scheduler': scheduler.state_dict()})
            torch.save(checkpoint, os.path.join(parameters['exp_path'], f'checkpoint-epoch{eps}.pth.tar'))


if __name__ == '__main__':

    parameters = parser()

    model, datasets = get_model_and_data(parameters)

    # training setting
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=parameters['lr'])
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=parameters['lr'])

    # lr scheduler
    if parameters['lr_scheduler'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.1,
            min_lr=1e-7,
            threshold=0,
            patience=5,
            cooldown=5,
            verbose=True)
            
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=parameters['lr_step_size'], gamma=parameters['lr_gamma'], verbose=True)
    else:
        scheduler = None


    if parameters['checkpoint_path'] != None:
        checkpoint = torch.load(parameters['checkpoint_path'])

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        if scheduler != None: scheduler.load_state_dict(checkpoint['lr_scheduler'])
            
        start_epoch = checkpoint['epoch'] + 1
        print(f"Model loaded from checkpoint {parameters['checkpoint_path']}...")

    else:
        start_epoch = 1
        print('Train on new model...')
    
    # print(f'Init lr: {scheduler.get_last_lr()[0]}')
    do_epoch(model=model, datasets=datasets, parameters=parameters, optimizer=optimizer, scheduler=scheduler, start_epoch=start_epoch)

