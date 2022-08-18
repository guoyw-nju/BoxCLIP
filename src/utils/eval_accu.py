import torch
import clip
from tqdm import tqdm

def eval_accu(dataloader, model, cat_texts, device, type='reconstruct'):

    bbox_total, bbox_cat_correct, bbox_corrdinate_error = 0, 0, 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            bs, num_bbox = batch['bboxs'].shape[:2]

            # transfer the data to device(cuda)
            for k, v in batch.items():
                if torch.is_tensor(batch[k]): batch[k] = batch[k].to(device)

            batch.update(model(batch)) # batch['output_bboxs'] batch['output_cat_feats']

            if type == 'reconstruct':
                batch.update(model.feat2cat(batch['output_cat_feats'], cat_texts)) # 'output_cat_name'
                eval_bbox_cat = batch['output_cat_name']
                eval_bbox_coordinate = batch['output_bboxs']

            elif type == 'generate':

                text_feats = torch.cat([clip.tokenize(t[0]).to(device) for t in batch['clip_texts']], dim=0)
                # print(text_feats.shape)

                text_feats = model.clip_model.encode_text(text_feats).float()
                # print(text_feats.shape)

                batch_gen = model.generate(text_feats)
                batch_gen.update(model.feat2cat(batch_gen['output_cat_feats'], cat_texts)) # 'output_cat_name'
                eval_bbox_cat = batch_gen['output_cat_name']
                eval_bbox_coordinate = batch_gen['output_bboxs']

            else:
                assert False, 'Wrong type.'


            for i in range(bs):
                for j in range(num_bbox):
                    if batch['masks'][i][j]:

                        bbox_total += 1
                        if batch['cat_name'][i][j] == eval_bbox_cat[i][j]:
                            bbox_cat_correct += 1

                        bbox_corrdinate_error += torch.abs(batch['bboxs'][i][j] - eval_bbox_coordinate[i][j]).sum().cpu().item()

    # print(f"bbox_cat_correct: {bbox_cat_correct}\nbbox_total: {bbox_total}\nbbox_corrdinate_error: {bbox_corrdinate_error}")

    return bbox_cat_correct / bbox_total, bbox_corrdinate_error / bbox_total * 224
