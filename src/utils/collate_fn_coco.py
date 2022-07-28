import torch
import torch.nn.functional as functional
import torchvision.transforms as transforms

def collate_fn_coco(batch, num_box=2):
    # input: batch = [(image, target, caption), ...]

    # output: batch = {
    #   'clip_images':  tensor(bs, 3, 224, 224) 
    #   'clip_texts':   list[[5], [5], ...]
    #   'bboxs':        tensor(bs, num_box, 4)
    #   'cat_name':     list[[num_box], [num_box], ...]
    #   'cat_id':       tensor(bs, num_box)
    #   'masks':        tensor(bs, num_box)
    # }

    clip_images, masks, clip_texts, cat_name = None, None, [], []
    bboxs, cat_id= torch.zeros(len(batch), num_box, 4), torch.zeros(len(batch), num_box)

    for n_batch, (_image, _target, _caption) in enumerate(batch):
        
        # clip_images
        if clip_images == None: clip_images = _image.unsqueeze(0)
        else: clip_images = torch.cat((clip_images, _image.unsqueeze(0)), dim=0)

        # clip_texts
        clip_texts.append(_caption)

        # masks
        num_obj = len(_target)
        _mask = torch.tensor([True for _ in range(min(num_box, num_obj))] + 
                             [False for _ in range(max(0, num_box-num_obj))])
        if masks == None: masks = _mask.unsqueeze(0)
        else: masks = torch.cat((masks, _mask.unsqueeze(0)), dim=0)
        
        _cat_name = []
        for n_target, obj in enumerate(_target):
            if n_target == num_box: break
                
            bboxs[n_batch][n_target] = torch.tensor(obj['bbox'])

            cat_id[n_batch][n_target] = obj['category_id']

            _cat_name.append(obj['category_name'])

                
        while len(_cat_name) < num_box:
            _cat_name.append('NaN')
        cat_name.append(_cat_name)

    return {
        'clip_images': clip_images,
        'clip_texts': clip_texts,
        'bboxs': bboxs,
        'cat_name': cat_name,
        'cat_id': cat_id,
        'masks': masks
    }
