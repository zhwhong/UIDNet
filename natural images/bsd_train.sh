python main.py --ndct ./data/Patch_data/BSD500_CUT_noise15/train/cut_clean/ --ldct ./data/Patch_data/BSD500_CUT_noise15/train/cut_noise/ --model_name wgan-gp --c_dim 3 --epoch 200 --gpu_id 2
# python main.py --ndct ./data/Patch_data/BSD500_CUT_noise25/train/cut_clean/ --ldct ./data/Patch_data/BSD500_CUT_noise25/train/cut_noise/ --model_name wgan-gp --c_dim 3 --epoch 200 --gpu_id 2
# python main.py --ndct ./data/Patch_data/BSD500_CUT_noise50/train/cut_clean/ --ldct ./data/Patch_data/BSD500_CUT_noise50/train/cut_noise/ --model_name wgan-gp --c_dim 3 --epoch 200 --gpu_id 2
# python main.py --ndct ./data/challenge/cut_clean/ --ldct ./data/challenge/cut_noise --model_name dcgan --c_dim 1 --epoch 200 --gpu_id 2
# python main.py --ndct ./data/CT_CUT/cut_clean/ --ldct ./data/CT_CUT/cut_noise/ --model_name dcgan --c_dim 1 --epoch 20 --gpu_id 3
