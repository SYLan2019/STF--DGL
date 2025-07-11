for file in /home/aa/zxy/code_flow/Data/input/SMD/ServerMachineDataset/processed/machine*train*
do 
    var=${file##*/}
    # echo $var
    echo ${var%_*}
    CUDA_VISIBLE_DEVICES=0 nohup python3 -u main.py\
        --n_blocks=2\
        --batch_size=256\
        --window_size=60\
        --train_split=0.6\
        --feat_gat_embed_dim=38\
        --time_gat_embed_dim=38\
        --n_features=38\
        --hidden_size=32\
        --state_size=8\
        --name=${var%_*}\
        > OCCmain${var%_*}.log 2>&1 &
    wait
done
