import torch
import torch.nn.functional as functional
import torchvision.transforms as transforms

def collate_fn_coco(batch):
    num_boxes = 2
#     num_classes = 91
    coco_resize = 224
    
    # device = "cuda" if torch.cuda.is_available else "cpu"
    device = "cpu"

    # resize_fn = transforms.Resize([coco_resize, coco_resize])
    
    images, indexs, masks, captions = None, [], None, []
    for _image, _index, _caption in batch:
        _, h, w = _image.shape
        
        # if images == None: images = resize_fn(_image.unsqueeze(0))
        # else: images = torch.cat((images, resize_fn(_image.unsqueeze(0))), dim=0)
        if images == None: images = _image.unsqueeze(0)
        else: images = torch.cat((images, _image.unsqueeze(0)), dim=0)
        
        captions.append(_caption)

        num_obj = len(_index)
        _mask = torch.tensor([True for _ in range(min(num_boxes, num_obj))] + 
                             [False for _ in range(max(0, num_boxes-num_obj))])
        _mask = _mask[None]
#         print(_mask.shape)
        
        if masks == None: masks = _mask
        else: masks = torch.cat((masks, _mask), dim=0)
        
        cnt, tmp = 0, []
        for obj in _index:
            if cnt >= num_boxes: break
                
#             print(obj['bbox'])
#             obj['bbox'][0] /= w
#             obj['bbox'][2] /= w
#             obj['bbox'][1] /= h
#             obj['bbox'][3] /= h
#             print(obj['bbox'])
            feat_box = [obj['bbox'][0] / w, obj['bbox'][1] / h, 
                        obj['bbox'][2] / w, obj['bbox'][3] / h]
            
#             feat_box = obj['bbox']
            cat_id = obj['category_id']
            feat = feat_box, cat_id
#             feat_class = functional.one_hot(torch.tensor(obj['category_id']), 
#                                             num_classes=num_classes)            
#             feat = torch.cat((feat_box, feat_class))
            
            cnt += 1
            tmp.append(feat)
#             if tmp == None: tmp = feat.unsqueeze(0)
#             else: tmp = torch.cat((tmp, feat.unsqueeze(0)), dim=0)
                
        while cnt < num_boxes:
            cnt += 1
            if tmp == None: tmp = torch.zeros(1, num_classes+4)
            else: tmp = torch.cat((tmp, torch.zeros(1, num_classes+4)), dim=0)
                
        indexs.append(tmp)
#         if indexs == None: indexs = tmp.unsqueeze(0)
#         else: indexs = torch.cat((indexs, tmp.unsqueeze(0)), dim=0)
        
    data_batch = {
        'clip_images': images.to(device),
        'bboxes': indexs,
        'masks': masks.to(device),
        'clip_texts': captions
    }
    return data_batch # (bs, 3, 224, 224) (bs, num_boxes, 95)