CUDA_VISIBLE_DEVICES=0 python3 main.py\
    --n_blocks=1\
    --batch_size=256\
    --k=10\
    --loss_weight_manifold_ne=5\
    --loss_weight_manifold_po=1\
    --train_split=0.8\
    --n_features=25\
    --d_model=25\
    --window_size=60\
    --feat_gat_embed_dim=25\
    --time_gat_embed_dim=25\
    --hidden_size=32\
    --state_size=8\
    --name=PSM\
    | tee ./Data/log/PSM.log 2>&1