# BoxCLIP: align_v4
### Branch Description:
1.  Bounding Box输出坐标Linear层后增加sigmoid激活函数（参考YOLOv2）；
2.  对坐标回归的损失函数进行修改，(x, y, w, h)分别为中心点坐标和宽高，对w, h取平方根后计算L2回归误差；
3.  修改了坐标encode方式：拼接后过两层FC Layer；
4.  不再原始CLIP Feature进行align，而是CLIP Feature过两层FC Layer后进行align.

### To start training:
```
python -m src.train.train \
--exp_name <EXPRIMENT NAME> \
--num_epoch 10 \
--batch_size 64 \
--lr 0.0001 \
--lr_gamma 0.1 \
--lr_step_size 5
```