"""
old vit training script with adaptation to i-jepa
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import logging
import time
from tqdm import tqdm

from config import config
from models import build_model
from losses.ijepa_loss import IJEPALoss
from data.dataset import create_dataloader
from utils.misc import (
    set_seed, setup_logging, save_checkpoint, load_checkpoint,
    adjust_learning_rate, AverageMeter, ProgressMeter,
    init_distributed_mode, is_main_process, get_world_size, get_rank, all_reduce_mean
)


def parse_args():
    parser = argparse.ArgumentParser(description='I-JEPA Training')
    
    # Basic settings
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--device', default='cuda', type=str, help='Device to use')
    
    # Data settings
    parser.add_argument('--data-path', default='./data', type=str, help='Path to dataset')
    parser.add_argument('--dataset', default='imagenet', type=str, help='Dataset name')
    parser.add_argument('--image-size', default=224, type=int, help='Image size')
    
    # Training settings
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--batch-size', default=256, type=int, help='Batch size')
    parser.add_argument('--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight-decay', default=0.04, type=float, help='Weight decay')
    parser.add_argument('--warmup-epochs', default=10, type=int, help='Warmup epochs')
    parser.add_argument('--min-lr', default=1e-6, type=float, help='Minimum learning rate')
    parser.add_argument('--lr-scheduler', default='cosine', type=str, help='LR scheduler')
    
    # Model settings
    parser.add_argument('--encoder-type', default='vit', type=str, help='Encoder type')
    parser.add_argument('--embed-dim', default=768, type=int, help='Embedding dimension')
    parser.add_argument('--num-layers', default=12, type=int, help='Number of layers')
    parser.add_argument('--num-heads', default=12, type=int, help='Number of attention heads')
    parser.add_argument('--patch-size', default=16, type=int, help='Patch size')
    
    # Loss settings
    parser.add_argument('--loss-type', default='smoothl1', type=str, help='Loss type')
    parser.add_argument('--normalize', default=True, type=bool, help='Normalize features')
    
    # Masking settings
    parser.add_argument('--num-context-blocks', default=4, type=int, help='Number of context blocks')
    parser.add_argument('--num-target-blocks', default=6, type=int, help='Number of target blocks')
    parser.add_argument('--context-block-size', default=2, type=int, help='Context block size')
    parser.add_argument('--target-block-size', default=2, type=int, help='Target block size')
    
    # Logging and checkpointing
    parser.add_argument('--log-dir', default='./logs', type=str, help='Log directory')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', type=str, help='Checkpoint directory')
    parser.add_argument('--checkpoint-freq', default=5, type=int, help='Checkpoint frequency')
    parser.add_argument('--resume', default='', type=str, help='Resume from checkpoint')
    
    # Distributed training
    parser.add_argument('--world-size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--dist-url', default='env://', type=str, help='URL used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='Distributed backend')
    parser.add_argument('--num-workers', default=8, type=int, help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config.data.data_path = args.data_path
    config.data.train_dataset = args.dataset
    config.data.val_dataset = args.dataset
    config.data.image_size = args.image_size
    
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.weight_decay = args.weight_decay
    config.training.warmup_epochs = args.warmup_epochs
    config.training.min_lr = args.min_lr
    config.training.lr_scheduler = args.lr_scheduler
    config.training.num_context_blocks = args.num_context_blocks
    config.training.num_target_blocks = args.num_target_blocks
    config.training.context_block_size = args.context_block_size
    config.training.target_block_size = args.target_block_size
    
    config.model.encoder_type = args.encoder_type
    config.model.embed_dim = args.embed_dim
    config.model.num_layers = args.num_layers
    config.model.num_heads = args.num_heads
    config.model.patch_size = args.patch_size
    
    config.log_dir = args.log_dir
    config.checkpoint_dir = args.checkpoint_dir
    config.checkpoint_freq = args.checkpoint_freq
    
    config.device = args.device
    config.num_workers = args.num_workers
    config.distributed = args.world_size > 1
    config.world_size = args.world_size
    config.dist_url = args.dist_url
    config.dist_backend = args.dist_backend
    
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.world_size > 1:
        init_distributed_mode(args)
    if is_main_process():
        setup_logging(config.log_dir)
        logging.info(f"Config: {config}")

    model = build_model(config)
    model = model.to(config.device)
    
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )

    criterion = IJEPALoss(loss_type=args.loss_type, normalize=args.normalize)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(config.training.beta1, config.training.beta2),
    )
    
    train_loader = create_dataloader(
        dataset_name=config.data.train_dataset,
        data_path=config.data.data_path,
        batch_size=config.training.batch_size,
        img_size=config.data.image_size,
        split="train",
        num_workers=config.num_workers,
        use_color_jitter=config.data.use_color_jitter,
        use_gaussian_blur=config.data.use_gaussian_blur,
        use_random_resized_crop=config.data.use_random_resized_crop,
        use_random_horizontal_flip=config.data.use_random_horizontal_flip,
        distributed=config.distributed,
    )
    
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info(f"Loading checkpoint '{args.resume}'")
            start_epoch, best_loss = load_checkpoint(args.resume, model, optimizer)
            logging.info(f"Loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            logging.error(f"No checkpoint found at '{args.resume}'")
    
    cudnn.benchmark = True
    

    for epoch in range(start_epoch, config.training.epochs):
        if config.distributed:
            train_loader.sampler.set_epoch(epoch)
        

        adjust_learning_rate(optimizer, epoch, args)
        
        train_loss = train_one_epoch(
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
        )
        
        if is_main_process() and (epoch + 1) % config.checkpoint_freq == 0:
            is_best = train_loss < best_loss
            best_loss = min(train_loss, best_loss)
            
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict() if config.distributed else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                },
                is_best,
                config.checkpoint_dir,
                filename=f'checkpoint_epoch{epoch+1}.pth',
            )
    
    logging.info("Training completed!")


def train_one_epoch(model, criterion, train_loader, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    mse = AverageMeter('MSE', ':.4f')
    cos_sim = AverageMeter('CosSim', ':.4f')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, mse, cos_sim],
        prefix=f"Epoch: [{epoch}]"
    )
    model.train()
    
    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        images = images.to(args.device, non_blocking=True)
        
        outputs = model(
            x=images,
            num_context_blocks=config.training.num_context_blocks,
            num_target_blocks=config.training.num_target_blocks,
            context_block_size=config.training.context_block_size,
            target_block_size=config.training.target_block_size,
        )
        
        loss, loss_dict = criterion(
            pred_target_feats=outputs["pred_target_feats"],
            target_feats=outputs["target_feats"],
            num_target_per_batch=outputs["num_target_per_batch"],
        )
        
        losses.update(loss.item(), images.size(0))
        mse.update(loss_dict["mse"].item(), images.size(0))
        cos_sim.update(loss_dict["cos_sim"].item(), images.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.module.momentum_update() if config.distributed else model.momentum_update()
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % 20 == 0 and is_main_process():
            progress.display(i)
    
    if config.distributed:
        losses_tensor = torch.tensor([losses.avg], device=args.device)
        losses_tensor = all_reduce_mean(losses_tensor)
        losses.avg = losses_tensor.item()
    
    if is_main_process():
        logging.info(f"Epoch: {epoch} | Loss: {losses.avg:.4f} | MSE: {mse.avg:.4f} | CosSim: {cos_sim.avg:.4f}")
    
    return losses.avg


if __name__ == "__main__":
    main() 