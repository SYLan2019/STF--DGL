CUDA_VISIBLE_DEVICES=0 python3 main.py\
    --n_blocks=5\
    --batch_size=256\
    --k=10\
    --loss_weight_manifold_ne=5\
    --loss_weight_manifold_po=1\
    --train_split=0.8\
    --n_features=123\
    --d_model=123\
    --window_size=40\
    --feat_gat_embed_dim=123\
    --time_gat_embed_dim=123\
    --hidden_size=32\
    --state_size=8\
    --name=Wadi\
    | tee ./Data/log/WADI.log 2>&1