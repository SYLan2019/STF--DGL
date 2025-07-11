CUDA_VISIBLE_DEVICES=0 python3 main.py\
    --n_blocks=1\
    --batch_size=256\
    --k=10\
    --seed=18\
    --loss_weight_manifold_ne=5\
    --loss_weight_manifold_po=1\
    --window_size=40\
    --train_split=0.8\
    --n_features=51\
    --d_model=51\
    --feat_gat_embed_dim=51\
    --time_gat_embed_dim=51\
    --hidden_size=32\
    --state_size=8\
    --name=SWaT\
    | tee ./Data/log/SWAT.log 2>&1

