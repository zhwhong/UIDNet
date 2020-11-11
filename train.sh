python main.py --ndct ./data/CUT_3mm/cut_clean/ --ldct ./data/CUT_1mm/cut_noise/ --model_name wgan-gp --c_dim 1 --epoch 50 --gpu_id 0
# python main.py --ndct ./data/challenge/cut_clean/ --ldct ./data/challenge/cut_noise/ --model_name dcgan --c_dim 1 --epoch 200 --gpu_id 3
# python main.py --ndct ./data/CT_CUT/cut_clean/ --ldct ./data/CT_CUT/cut_noise/ --model_name dcgan --c_dim 1 --epoch 20 --gpu_id 3
# python main.py --ndct ./data/CT_CUT/cut_clean/ --ldct ./data/CT_CUT/cut_noise/ --model_name wgan-gp --c_dim 1 --epoch 50 --gpu_id 2