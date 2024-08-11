import argparse
import yaml
import numpy as np
import random
import torch
import torch.nn as nn
import wandb
# from torchvision.models import vit_b_16
from torchmetrics.classification import MulticlassAccuracy
from torchsummary import summary

import data, engine, model, utils


# def parse_args():
#    parser = argparse.ArgumentParser(description="Original Architecture of VIT")
#    parser.add_argument("--exp_name", type=str, default="org", required=False)
#    parser.add_argument("--dataset_name", type=str, default="cifar10", required=False, help="write dataset name in small case i.e. for MNIST --> mnist, CIFAR10 --> cifar10")
#    parser.add_argument("--seed", type=int, default=64)
#    parser.add_argument("--batch_size", type=int, default=64)
#    parser.add_argument("--num_epoch", type=int, default=100)
#    parser.add_argument("--learning_rate", type=float, default=3e-4)
#    parser.add_argument("--input_channel", type=int, default=3)
#    parser.add_argument("--patch_size", type=int, default=16)
#    parser.add_argument("--embedding_size", type=int, default=768)
#    parser.add_argument("--input_image_size", type=int, default=224)
#    parser.add_argument("--vit_depth", type=int, default=12)
#    parser.add_argument("--num_class", type=int, default=10, help="number of classes in dataset")
#    parser.add_argument("--wandb_project", type=str, default="vit-small-data")
#    parser.add_argument("--wandb_runname", type=str, required=True, help="wandb run name that you want to see in wandb log")
#    parser.add_argument("--output_dir", type=str, default="result")
   
#    args = parser.parse_args()
#    return args

# args=parse_args()

def main(args):
   print("\n")
   print(f"Experiment Name: {args['exp_name']}")
   print(f"Experiment Details: {args['details']}")
   print("\n")
   print(f"Dataset Name: {args['dataset_name']}")
   print(f"Seed: {args['seed']}")
   print(f"Batch Size: {args['batch_size']}")
   print(f"Number of Epochs: {args['num_epoch']}")
   print(f"Learning Rate: {args['learning_rate']}")
   print(f"Input Channel: {args['input_channel']}")
   print(f"Patch Size: {args['patch_size']}")
   print(f"Embedding Size: {(config['patch_size'] ** 2) * 3}")
   print(f"Input Image Size: {args['input_image_size']}")
   print(f"ViT Depth: {args['vit_depth']}")
   print(f"Number of Classes: {args['num_class']}")
   print(f"WandB Project: {args['wandb_project']}")
   print(f"WandB Run Name: {args['wandb_runname']}")
   print(f"Output Directory: {args['output_dir']}")
   print("\n")
   
   # HYPERPARAMETERS
   EXP_NAME = args['exp_name']
   DATASET = args['dataset_name']
   EXP = EXP_NAME + "_" + DATASET
   SEED = args['seed']
   BATCH_SIZE = args['batch_size']
   NUM_EPOCH = args['num_epoch']
   LEARNIGN_RATE = float(args['learning_rate'])#3e-4 # 3e-4, 4e-5, 7e-6, 5e-7, 3e-9
   INCHANNELS = args['input_channel']
   PATCH_SIZE = args['patch_size']
   EMBEDDING_SIZE = (PATCH_SIZE ** 2) * INCHANNELS # args['embedding_size']
   IMG_SIZE = args['input_image_size']
   DEPTH = args['vit_depth']
   NUM_CLASS = args['num_class']
   # CUDA_LAUNCH_BLOCKING=1

   DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
   # print(device)
   # summary
   print("\n",summary(model.ViT(INCHANNELS, PATCH_SIZE, EMBEDDING_SIZE, IMG_SIZE, DEPTH, NUM_CLASS), (INCHANNELS, IMG_SIZE, IMG_SIZE), device = DEVICE),"\n")
   
   
   train_dataloader, test_dataloader = data.prepare_dataloader(args)
   print("\n")
   print(f"EXP {EXP_NAME}: Original VIT on {DATASET} with depth {DEPTH} and LEARNIGN_RATE {LEARNIGN_RATE}")
   print("\n\n")
   random.seed(SEED)
   np.random.seed(SEED)
   torch.manual_seed(SEED)
   torch.cuda.manual_seed(SEED)
   vit_model = model.ViT(in_channels = INCHANNELS, patch_size = PATCH_SIZE,
                        embedding_size = EMBEDDING_SIZE, img_size = IMG_SIZE,
                        depth = DEPTH, n_classes = NUM_CLASS).to(DEVICE)

   # torch_vit = vit_b_16().to(device)

   # vit_model= nn.DataParallel(vit_model).to(device)

   loss_fn = torch.nn.CrossEntropyLoss()
   accuracy_fn = MulticlassAccuracy(num_classes = NUM_CLASS).to(DEVICE)
   optimizer = torch.optim.SGD(vit_model.parameters(), lr = LEARNIGN_RATE, weight_decay=0.03)
   # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.1, end_factor = 0.07,
   #                                               total_iters = NUM_EPOCH - 10)

   # print(f"lr: {scheduler.optimizer.param_groups[0]['lr']}")

   train_model, train_loss, test_loss, train_acc, test_acc = engine.train(model = vit_model, 
                                                                        train_dataloader = train_dataloader,
                                                                        test_dataloader = test_dataloader, 
                                                                        optimizer = optimizer,
                                                                     #    scheduler = scheduler,
                                                                        loss_fn = loss_fn, 
                                                                        accuracy_fn = accuracy_fn, 
                                                                        epochs = NUM_EPOCH, 
                                                                        device = DEVICE,
                                                                        args = args)

   utils.save_model(model = train_model, target_dir = "./save_model", model_name = f"vit_model_{EXP}_{LEARNIGN_RATE}_{BATCH_SIZE}.pth")

   utils.plot_graph(train_losses = train_loss, test_losses = test_loss, train_accs = train_acc, 
                  test_accs = test_acc, fig_name = f"plots/vit_Loss_and_accuracy_plot_{EXP}_{LEARNIGN_RATE}_{BATCH_SIZE}.jpg")
   

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Original Architecture of VIT")
   parser.add_argument("--config", type=str, required=True, help="Path to the config file")
   
   args = parser.parse_args()
   
   # Load config file
   with open(args.config, 'r') as file:
      config = yaml.safe_load(file)

   # Automatically generate wandb_runname
   config['wandb_runname'] = f"{config['exp_name']}_{config['dataset_name']}_Lr_{config['learning_rate']}_EMB_{(config['patch_size'] ** 2) * 3}_patch_{config['patch_size']}_depth_{config['vit_depth']}"
   
   main(config)