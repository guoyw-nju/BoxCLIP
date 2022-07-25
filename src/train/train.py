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

    checkpoint_path = './checkpoint/{}'.format(time.strftime("%m-%d-%H-%M", time.localtime()))
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    writer = SummaryWriter(log_dir=checkpoint_path)    

    print('Start training...')
    for eps in range(start_epoch, start_epoch + parameters['num_epochs']):

        for i, batch in enumerate(tqdm(dataloader, desc=f'Epoch {eps}')):
            
            # if eps == start_epoch and i == 0:
            #     with SummaryWriter(comment='boxclip') as w:
            #         w.add_graph(model, (batch, ))

            optimizer.zero_grad()
            batch = model(batch)
            mixed_loss, losses = model.compute_loss(batch)

            mixed_loss.backward()
            optimizer.step()

            writer.add_scalars(f"Loss/Iters", {
                'mixed_loss': mixed_loss,
                'bbox_mse': losses['bbox_mse'],
                'cats_cos': losses['cats_cos']
            }, (eps-1)*num_batch + i)

        writer.add_scalar(f"Lr/Epochs", scheduler.get_last_lr()[0], eps)
        scheduler.step()

        writer.flush()

        checkpoint = {
            'epoch': eps,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict()
        }
        for nms in os.listdir(checkpoint_path):
            f_name = os.path.join(checkpoint_path, nms)
            if f_name.endswith('.pth.tar'):
                os.remove(f_name)
        torch.save(checkpoint, os.path.join(checkpoint_path, f'checkpoint-epoch{eps}.pth.tar'))

    
if __name__ == '__main__':

    parameters = parser()

    # try device
    if 'device' not in parameters:
        parameters['device'] = "cuda" if torch.cuda.is_available else "cpu"

    model, datasets = get_model_and_data(parameters)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print('Trainable params: %.2fM' % (sum(p.numel() for _, p in model.named_parameters() if p.requires_grad) / 1000000.0))

    # training setting
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters['lr'])

    # lr scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)

    if parameters['checkpoint_path'] != None:
        checkpoint = torch.load(parameters['checkpoint_path'])

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        print(f"Model loaded from checkpoint {parameters['checkpoint_path']}...")

    else:
        start_epoch = 1
        print('Train on new model...')
    
    do_epoch(model=model, datasets=datasets, parameters=parameters, optimizer=optimizer, start_epoch=start_epoch)

