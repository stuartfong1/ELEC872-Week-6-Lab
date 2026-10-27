# Part 0: Reference model
python train.py --exp-name baseline

# Part A: LSTM
python train.py --exp-name model_cnn --model cnn
python train.py --exp-name rnn_gru --rnn gru

# Part B: Convolutions
python train.py --exp-name n_conv_1 --n-conv 1
python train.py --exp-name n_conv_2 --n-conv 2

# Part C: Window 
python train.py --exp-name window_20 --window 20
python train.py --exp-name window_60 --window 60

# Part D: Augmentations
python train.py --exp-name dropout_0 --dropout 0
python train.py --exp-name weight_decay_1e-4 --weight-decay 0.0001
python train.py --exp-name augment --augment

# Part E: Learning rate warmup
python train.py --exp-name warmup --warmup

Part F: Focal loss
python train.py --exp-name focal_loss --focal-loss
