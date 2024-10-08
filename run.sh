## MCAP

## AMiner
nohup python -u main.py --seed 2021 --context_hops 1 --dataset aminer --lr 0.001 --l2 0.0005  --att_dropout 0 --pool sum --load_ii_sort 5 --with_sim 1 --with_user 1 --with_uu_co_author 1 --with_ii_co_author 1 --with_uu_co_venue 1 --with_ii_co_venue 1 --gpu_id 0 --gnn MCAP --batch_size 1024 --step 1 > logs/aminer/LightGCN-MCAP.log 2>&1 &

## aminer-wechat
# nohup python -u main.py --seed 2021 --context_hops 1 --dataset aminer-wechat --lr 0.001 --l2 0.0001  --att_dropout 0 --pool sum --load_ii_sort 10 --with_sim 1 --with_user 1 --with_uu_co_author 1 --with_ii_co_author 1 --with_uu_co_venue 1 --with_ii_co_venue 1 --gpu_id 0 --gnn MCAP --batch_size 1024 --step 1 > logs/aminer-wechat/LightGCN-MCAP.log 2>&1 &

## CiteULike
# nohup python -u main.py --seed 2021 --dataset citeulike --att_dropout 1 --step 5 --lr 0.001 --l2 1e-6  --pool sum --load_ii_sort 50 --context_hops 1 --with_sim 1 --with_user 1 --e 0.001 --with_uu_co_author 0 --with_ii_co_author 0 --with_uu_co_venue 0 --with_ii_co_venue 0 --gpu_id 1 --gnn MCAP --batch_size 1024  > logs/citeulike/LightGCN-MCAP.log 2>&1 &

## dblp
# nohup python -u main.py --seed 2021 --context_hops 1 --dataset dblp --lr 0.001 --l2 0.0001  --att_dropout 0 --pool sum --with_sim 0 --with_user 0 --gpu_id 0 --gnn LightGCN --batch_size 1024 --step 1 > logs/dblp/LightGCN.log 2>&1 &
