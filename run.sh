

# ------- Variables --------

si_c=10.0
epsilon=0.001
mode=ema
alpha=1.0
lseg=False
bacc=False

n_rounds=100
n_epochs=1
n_clients=5
dataset=cifar10
non_iid_alpha=0.1
arch=mlp
mlp_hs=1024
trans_pretrained=False
shuffle=True
drop_last=True
image_path=../image_dataset/
num_workers=5
seed_use=1234
lr=0.001
test_ratio=0.2
batch_size=128

gpu=1
save_path=../output/exp-cifar-split-cwc-1/


# --------- Script -----------

CUDA_VISIBLE_DEVICES="$gpu" python scripts/main.py --n_clients "$n_clients" --dataset "$dataset" \
          --save_path "$save_path" --n_rounds "$n_rounds" --n_epochs "$n_epochs" --arch "$arch" \
          --batch_size "$batch_size" --lr "$lr" --num_workers "$num_workers" --epsilon "$epsilon" \
          --seed_use "$seed_use" --si_c "$si_c" --mode "$mode" --alpha "$alpha" --lseg "$lseg" --bacc "$bacc" \
          --shuffle "$shuffle" --drop_last "$drop_last" --image_path "$image_path" --mlp_hs "$mlp_hs" \
          --non_iid_alpha "$non_iid_alpha" --test_ratio "$test_ratio" --trans_pretrained "$trans_pretrained"
