#!/usr/bin/env python3
"""
Main training script for the Holistic Diagnostic Platform.

This script serves as the primary entry point for training medical image
segmentation models with comprehensive configuration management, HIPAA
compliance, and production-grade features.
"""

import os
import sys
import argparse
import logging
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import monai
from monai.utils import set_determinism

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import get_config, PlatformConfig
from src.models.segmentation import MedicalSwinUNETR
from src.models.loss import create_loss_function
from src.training import MedicalTrainer, ModelCheckpointManager
from src.data import (
    MedicalImageDataset,
    create_medical_dataloader,
    get_training_transforms,
    get_validation_transforms
)
from src.utils import (
    AuditEventType, 
    AuditSeverity, 
    log_audit_event,
    setup_logging
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train medical image segmentation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=f"medical_segmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Name for the experiment"
    )
    
    # Data paths
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to training data directory"
    )
    parser.add_argument(
        "--train-list",
        type=str,
        help="Path to training data list JSON file"
    )
    parser.add_argument(
        "--val-list",
        type=str,
        help="Path to validation data list JSON file"
    )
    
    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="swin_unetr",
        choices=["swin_unetr"],
        help="Model architecture to use"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained weights"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of output classes"
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision"
    )
    
    # Data augmentation
    parser.add_argument(
        "--augmentation-prob",
        type=float,
        default=0.2,
        help="Probability for data augmentation"
    )
    parser.add_argument(
        "--spatial-size",
        nargs=3,
        type=int,
        default=[96, 96, 96],
        help="Spatial size for training patches"
    )
    
    # Validation and checkpointing
    parser.add_argument(
        "--val-interval",
        type=int,
        default=1,
        help="Validation interval in epochs"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
        help="Checkpoint saving interval in epochs"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience"
    )
    
    # Hardware and performance
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--cache-rate",
        type=float,
        default=0.0,
        help="Cache rate for data loading (0.0-1.0)"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic training for reproducibility"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    # Output and logging
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments",
        help="Output directory for experiments"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from"
    )
    
    return parser.parse_args()


def setup_data_loaders(
    args: argparse.Namespace,
    config: PlatformConfig,
    session_id: str,
    user_id: str
) -> Tuple[DataLoader, DataLoader]:
    """Set up training and validation data loaders."""
    
    # Define transforms
    train_transforms = get_training_transforms(
        modality="CT",  # Default to CT, could be made configurable
        spatial_size=tuple(args.spatial_size),
        augmentation_prob=args.augmentation_prob,
        session_id=session_id,
        user_id=user_id
    )
    
    val_transforms = get_validation_transforms(
        modality="CT",
        spatial_size=tuple(args.spatial_size),
        session_id=session_id,
        user_id=user_id
    )
    
    # Create datasets
    if args.train_list and args.val_list:
        # Use provided data lists
        train_dataset = MedicalImageDataset(
            data_list_file=args.train_list,
            transform=train_transforms,
            cache_rate=args.cache_rate,
            session_id=session_id,
            user_id=user_id
        )
        
        val_dataset = MedicalImageDataset(
            data_list_file=args.val_list,
            transform=val_transforms,
            cache_rate=args.cache_rate,
            session_id=session_id,
            user_id=user_id
        )
    else:
        # Auto-discover data in directory
        # This would need to be implemented based on your data structure
        raise NotImplementedError("Auto-discovery of data not yet implemented. Please provide --train-list and --val-list")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Use batch size 1 for validation to handle varying sizes
        shuffle=False,
        num_workers=min(args.num_workers, 2),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0
    )
    
    return train_loader, val_loader


def setup_model(
    args: argparse.Namespace,
    config: PlatformConfig,
    session_id: str,
    user_id: str
) -> torch.nn.Module:
    """Set up the model."""
    
    if args.model_name == "swin_unetr":
        model = MedicalSwinUNETR(
            img_size=tuple(args.spatial_size),
            in_channels=1,
            out_channels=args.num_classes,
            feature_size=48,
            use_checkpoint=True,
            session_id=session_id,
            user_id=user_id
        )
    else:
        raise ValueError(f"Unknown model: {args.model_name}")
    
    # Load pretrained weights if specified
    if args.pretrained:
        # This would load pretrained weights
        # Implementation depends on your pretrained model source
        log_audit_event(
            event_type=AuditEventType.MODEL_TRAINING,
            severity=AuditSeverity.INFO,
            message="Pretrained weights requested but not implemented",
            user_id=user_id,
            session_id=session_id
        )
    
    return model


def setup_optimizer_and_scheduler(
    model: torch.nn.Module,
    args: argparse.Namespace
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """Set up optimizer and learning rate scheduler."""
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Create scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-7
    )
    
    return optimizer, scheduler


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(level=getattr(logging, args.log_level.upper()))
    logger = logging.getLogger(__name__)
    
    # Generate session and user IDs for audit logging
    session_id = str(uuid.uuid4())
    user_id = os.getenv("USER", "unknown")
    
    # Set deterministic behavior if requested
    if args.deterministic:
        set_determinism(args.seed)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    
    # Load configuration
    config = get_config()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Log training start
    log_audit_event(
        event_type=AuditEventType.MODEL_TRAINING,
        severity=AuditSeverity.INFO,
        message="Training session started",
        user_id=user_id,
        session_id=session_id,
        additional_data={
            "experiment_name": args.experiment_name,
            "model_name": args.model_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": str(device),
            "monai_version": monai.__version__,
            "torch_version": torch.__version__
        }
    )
    
    try:
        # Setup data loaders
        logger.info("Setting up data loaders...")
        train_loader, val_loader = setup_data_loaders(
            args, config, session_id, user_id
        )
        
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        # Setup model
        logger.info("Setting up model...")
        model = setup_model(args, config, session_id, user_id)
        model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Setup optimizer and scheduler
        logger.info("Setting up optimizer and scheduler...")
        optimizer, scheduler = setup_optimizer_and_scheduler(model, args)
        
        # Setup loss function
        loss_function = create_loss_function(
            loss_type="dice_ce",
            session_id=session_id,
            user_id=user_id
        )
        
        # Setup experiment directory
        experiment_dir = Path(args.output_dir) / args.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup checkpoint manager
        checkpoint_manager = ModelCheckpointManager(
            checkpoint_dir=str(experiment_dir / "checkpoints"),
            max_versions=10,
            encrypt_checkpoints=True,
            session_id=session_id,
            user_id=user_id
        )
        
        # Setup trainer
        logger.info("Setting up trainer...")
        trainer = MedicalTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            use_amp=args.use_amp,
            max_epochs=args.epochs,
            val_interval=args.val_interval,
            early_stopping_patience=args.early_stopping_patience,
            save_dir=str(experiment_dir),
            experiment_name=args.experiment_name,
            session_id=session_id,
            user_id=user_id
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Start training
        logger.info("Starting training...")
        training_summary = trainer.train()
        
        # Log training completion
        log_audit_event(
            event_type=AuditEventType.MODEL_TRAINING,
            severity=AuditSeverity.INFO,
            message="Training completed successfully",
            user_id=user_id,
            session_id=session_id,
            additional_data=training_summary
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Best metric: {training_summary['best_metric']:.4f} at epoch {training_summary['best_epoch']}")
        logger.info(f"Total training time: {training_summary['total_time']:.2f} seconds")
        
    except Exception as e:
        # Log training failure
        log_audit_event(
            event_type=AuditEventType.MODEL_TRAINING,
            severity=AuditSeverity.ERROR,
            message=f"Training failed: {str(e)}",
            user_id=user_id,
            session_id=session_id,
            additional_data={"error": str(e)}
        )
        
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()