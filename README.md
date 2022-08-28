# align_v3
1.  centerizing coordinate
2.  add sigmoid activation

# BoxCLIP
To begin training:
```
python -m src.train.train --exp_name <EXPRIMENT NAME> --num_epoch 10 --batch_size 64 --lr 0.0001 --lr_gamma 0.1 --lr_step_size 5
```