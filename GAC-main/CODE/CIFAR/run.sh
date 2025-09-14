## CIFAR10
python main.py

## CIFAR10 GAC
python /zhengzeqi/Githubs_code/GAC-main/CODE/CIFAR/main.py \
--batch_size 64 \
--lr 5e-4 \
--time 2 \
--epochs 250 \
--GAC \
--data_dir /zhengzeqi/vit_test/ViT_test_experiment/data/ \
--output /zhengzeqi/Githubs_code/GAC-main/CODE/weights/T_2 \
--use_cifar10 \
--model msresnet18_GAC \
--experiment CIFAR10_MSResNet_GAC



## CIFAR10 GAC LIF_spikingjelly
python /zhengzeqi/Githubs_code/GAC-main/CODE/CIFAR/main.py \
--batch_size 64 \
--lr 5e-4 \
--time 2 \
--epochs 250 \
--data_dir /zhengzeqi/vit_test/ViT_test_experiment/data/ \
--output /zhengzeqi/Githubs_code/GAC-main/CODE/weights/T_2 \
--use_cifar10 \
--GAC \
--use_spikingjelly_lif \
--model msresnet18_GAC \
--experiment CIFAR10_MSResNet_GAC_sp_LIF




## CIFAR10 MS ResNet LIF_original
python /zhengzeqi/Githubs_code/GAC-main/CODE/CIFAR/main.py \
--batch_size 64 \
--lr 5e-4 \
--time 2 \
--epochs 250 \
--data_dir /zhengzeqi/vit_test/ViT_test_experiment/data/ \
--output /zhengzeqi/Githubs_code/GAC-main/CODE/weights/T_2 \
--use_cifar10 \
--model msresnet18 \
--experiment CIFAR10_MSResNet_original


## CIFAR10 MS ResNet LIF_spikingjelly
python /zhengzeqi/Githubs_code/GAC-main/CODE/CIFAR/main.py \
--batch_size 64 \
--lr 5e-4 \
--time 4 \
--epochs 250 \
--data_dir /zhengzeqi/vit_test/ViT_test_experiment/data/ \
--output /zhengzeqi/Githubs_code/GAC-main/CODE/weights/T_4 \
--use_cifar10 \
--model msresnet18 \
--use_spikingjelly_lif \
--experiment CIFAR10_MSResNet_sp_LIF

## CIFAR10 MS ResNet LIF_spikingjelly_3d_pe_arch_4
python /zhengzeqi/Githubs_code/GAC-main/CODE/CIFAR/main.py \
--batch_size 64 \
--lr 5e-4 \
--time 2 \
--epochs 250 \
--data_dir /zhengzeqi/vit_test/ViT_test_experiment/data/ \
--output /zhengzeqi/Githubs_code/GAC-main/CODE/weights/T_2 \
--use_cifar10 \
--model msresnet18 \
--recurrent_coding \
--pe_type 3d_pe_arch_4 \
--use_spikingjelly_lif \
--experiment CIFAR10_MSResNet_sp_LIF_3d_pe_arch_4



## CIFAR10 GAC LIF_spikingjelly_3d_pe_arch_4
python /zhengzeqi/Githubs_code/GAC-main/CODE/CIFAR/main.py \
--batch_size 64 \
--lr 5e-4 \
--time 2 \
--epochs 250 \
--data_dir /zhengzeqi/vit_test/ViT_test_experiment/data/ \
--output /zhengzeqi/Githubs_code/GAC-main/CODE/weights/T_2 \
--use_cifar10 \
--GAC \
--use_spikingjelly_lif \
--recurrent_coding \
--pe_type 3d_pe_arch_4 \
--model msresnet18_GAC \
--experiment CIFAR10_MSResNet_GAC_sp_LIF_3d_pe_arch_4