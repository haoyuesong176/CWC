

# ------- Variables --------


dataset=isicd
non_iid_alpha=0.1
arch=resnet
mlp_hs=1024
trans_pretrained=False
bacc=False

n_rounds=3000
n_epochs=1
n_clients=4
image_path=../image_dataset/
shuffle=True
drop_last=True
test_ratio=0.2
num_workers=4
seed_use=1234
lr=0.001
batch_size=64

gpu=3
save_path=../output/tmp/
# save_path=../output/exp-cifar-split-fedavg-1/

# --------- Script -----------

CUDA_VISIBLE_DEVICES="$gpu" python scripts/fedavg.py --n_clients "$n_clients"  \
          --save_path "$save_path" --n_rounds "$n_rounds" --n_epochs "$n_epochs" \
          --batch_size "$batch_size" --lr "$lr" --num_workers "$num_workers" \
          --seed_use "$seed_use" --arch "$arch" --dataset "$dataset" --mlp_hs "$mlp_hs" \
          --shuffle "$shuffle" --drop_last "$drop_last" --image_path "$image_path" --bacc "$bacc" \
          --non_iid_alpha "$non_iid_alpha" --test_ratio "$test_ratio" --trans_pretrained "$trans_pretrained"

