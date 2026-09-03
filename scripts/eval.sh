set -e

echo "Time to evaluate models!"


goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --pretrain degenerate --backbone ACR2
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5" --task test --pretrain degenerate --backbone ACR2 --ood_param 0.5
goodtg --config_path final_configs/BAColorGVIsol/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --pretrain degenerate --backbone ACR2

goodtg --config_path final_configs/MNIST/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain degenerate --backbone ACR2
goodtg --config_path final_configs/MNIST/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain degenerate --backbone ACR2 --ood_param 0.8 --use_readout_norm bn
goodtg --config_path final_configs/MNIST/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain degenerate --backbone ACR2

goodtg --config_path final_configs/MUTAG/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain degenerate
goodtg --config_path final_configs/MUTAG/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --ood_param 0.5 --pretrain degenerate
goodtg --config_path final_configs/MUTAG/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain degenerate

goodtg --config_path final_configs/SST2Planted/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain degenerate
goodtg --config_path final_configs/SST2Planted/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain degenerate
goodtg --config_path final_configs/SST2Planted/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5" --task test --gpu_idx 1 --pretrain degenerate


echo "DONE all: :)"
