# python main.py --ndct ./data/challenge/cut_clean/ --ldct ./data/challenge/cut_noise/ --model_name dcgan --c_dim 1 --epoch 50 --gpu_id 1 --is_train False
# python main.py --ndct ./data/CT_CUT/cut_clean/ --ldct ./data/CT_CUT/cut_noise/ --model_name dcgan --c_dim 1 --epoch 50 --gpu_id 2 --is_train False
# python main.py --ndct ../CT_denoising/data/CT_CUT/cut_clean/ --ldct ../CT_denoising/data/CT_CUT/cut_noise/ --model_name wgan-gp --c_dim 1 --epoch 50 --gpu_id 2 --is_train False --checkpoint_dir ckpt_131002
python main.py --ndct ./data/test_1mm/cut_clean/ --ldct ./data/test_1mm/cut_noise/ --model_name wgan-gp --c_dim 1 --epoch 50 --gpu_id 0 --is_train False --checkpoint_dir ckpt_95002
# python main.py --ndct ./data/test_3mm/cut_clean/ --ldct ./data/test_3mm/cut_noise/ --model_name wgan-gp --c_dim 1 --epoch 50 --gpu_id 0 --is_train False --checkpoint_dir ckpt_73002
