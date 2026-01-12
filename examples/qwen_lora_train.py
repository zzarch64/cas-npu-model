#!/usr/bin/env python
"""
Qwen LoRA 微调训练脚本

这个脚本用于对 Qwen 模型进行 LoRA 微调训练。

运行方式:
    python examples/qwen_lora_train.py [--model MODEL_NAME] [--data DATA_PATH] [--output OUTPUT_DIR] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--lr LR] [--seed SEED]
"""

import sys
import os
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

# 添加扩展路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入echo_npu扩展
try:
    import echo_npu
    print("✓ ECHO-NPU extension imported successfully")
except ImportError as e:
    print(f"✗ Failed to import ECHO-NPU extension: {e}")
    sys.exit(1)

# 检查 transformers 是否安装
try:
    from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
    print("✓ transformers library found")
except ImportError:
    print("✗ transformers library not found")
    print("  Please install: pip install transformers")
    sys.exit(1)

# 检查 peft 是否安装
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    print("✓ peft library found")
except ImportError:
    print("✗ peft library not found")
    print("  Please install: pip install peft")
    sys.exit(1)


class TextDataset(Dataset):
    """简单的文本数据集"""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # 编码文本
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }


def set_seed(seed):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.echo_npu, 'manual_seed'):
        torch.echo_npu.manual_seed(seed)
    print(f"✓ Random seed set to {seed}")


def load_model_and_tokenizer(model_name, device):
    """加载模型和tokenizer"""
    print(f"\nLoading model: {model_name}")
    print("  (This may take a while on first run...)")
    
    try:
        # 尝试从缓存加载
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.float32,
            local_files_only=True,
        )
        print("  ✓ Model loaded (from cache)")
    except Exception as e:
        print(f"  ✗ Failed to load from cache: {e}")
        print("  Trying to download...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                dtype=torch.float32,
            )
            print("  ✓ Model loaded")
        except Exception as e2:
            print(f"  ✗ Failed to load model: {e2}")
            # 尝试替代模型
            try:
                model_name = "Qwen/Qwen2-0.5B"
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    trust_remote_code=True, 
                    local_files_only=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    dtype=torch.float32,
                    local_files_only=True,
                )
                print("  ✓ Alternative model loaded (from cache)")
            except Exception as e3:
                print(f"  ✗ Failed to load alternative model: {e3}")
                raise
    
    # 设置 pad_token（如果不存在）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  ✓ Set pad_token to eos_token")
    
    return model, tokenizer


def setup_lora(model, lora_r=8, lora_alpha=16, lora_dropout=0.1):
    """配置并应用 LoRA"""
    print("\nConfiguring LoRA...")
    
    # 确定目标模块（根据模型类型）
    # 对于 Qwen 模型，通常是 q_proj, k_proj, v_proj, o_proj
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 用于因果语言模型
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    
    print(f"  LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}, dropout={lora_config.lora_dropout}")
    
    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    print("  ✓ LoRA adapter applied")
    
    # 显示参数统计
    param_stats = model.num_parameters()
    if isinstance(param_stats, dict):
        trainable = param_stats.get('trainable', 0)
        total = param_stats.get('all', 0)
    else:
        total = param_stats if isinstance(param_stats, int) else 0
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if total == 0:
            total = sum(p.numel() for p in model.parameters())
    
    frozen = total - trainable
    trainable_ratio = (trainable / total * 100) if total > 0 else 0
    
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters (LoRA only): {trainable:,}")
    print(f"  Frozen parameters (original model): {frozen:,}")
    print(f"  Trainable ratio: {trainable_ratio:.2f}%")
    
    return model


def prepare_training_data(data_path, tokenizer, max_length=512):
    """准备训练数据"""
    print("\nPreparing training data...")
    
    if data_path and os.path.exists(data_path):
        # 从文件读取数据
        print(f"  Loading data from: {data_path}")
        texts = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
        print(f"  Loaded {len(texts)} samples from file")
    else:
        # 使用示例数据
        print("  Using example training data")
        texts = [
            "Hello, how are you? I'm doing well, thank you.",
            "What is the weather today? It's sunny and warm.",
            "Can you help me with this task? Of course, I'd be happy to help.",
            "Tell me a joke. Why did the chicken cross the road? To get to the other side!",
            "What is machine learning? Machine learning is a subset of artificial intelligence.",
            "Explain quantum computing. Quantum computing uses quantum mechanical phenomena.",
            "How does a neural network work? A neural network consists of interconnected nodes.",
            "What is the capital of France? The capital of France is Paris.",
            "Describe the process of photosynthesis. Photosynthesis converts light energy into chemical energy.",
            "What are the benefits of exercise? Exercise improves physical and mental health.",
        ]
        print(f"  Using {len(texts)} example samples")
    
    # 创建数据集
    dataset = TextDataset(texts, tokenizer, max_length=max_length)
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Max sequence length: {max_length}")
    
    return dataset


def train_epoch(model, dataloader, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        # 将数据移到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # 前向传播
        # 对于因果语言模型，labels 就是 input_ids（shifted）
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids  # 使用 input_ids 作为 labels 进行自回归训练
        )
        
        loss = outputs.loss
        
        # 检查 NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  ⚠ Warning: NaN/Inf loss detected at batch {batch_idx}")
            continue
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 清理内存
        del outputs, loss
        if hasattr(torch.echo_npu, 'empty_cache'):
            torch.echo_npu.empty_cache()
        
        # 打印进度
        if (batch_idx + 1) % 5 == 0:
            avg_loss = total_loss / num_batches
            print(f"  Epoch {epoch}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {avg_loss:.6f}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def save_checkpoint(model, tokenizer, output_dir, epoch, loss):
    """保存检查点"""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 保存 LoRA 适配器
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    
    # 保存训练信息
    info_file = os.path.join(checkpoint_dir, "training_info.txt")
    with open(info_file, 'w') as f:
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Loss: {loss:.6f}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
    
    print(f"  ✓ Checkpoint saved to {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Qwen LoRA Fine-tuning Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用默认参数训练
  python examples/qwen_lora_train.py
  
  # 指定模型和数据
  python examples/qwen_lora_train.py --model Qwen/Qwen2.5-0.5B --data train.txt --output ./output
  
  # 自定义训练参数
  python examples/qwen_lora_train.py --epochs 10 --batch-size 2 --lr 1e-4 --seed 42
        """
    )
    
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-0.5B",
                        help='Model name or path (default: Qwen/Qwen2.5-0.5B)')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to training data file (one text per line). If not provided, uses example data.')
    parser.add_argument('--output', type=str, default='./qwen_lora_output',
                        help='Output directory for checkpoints (default: ./qwen_lora_output)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size (default: 2)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length (default: 512)')
    parser.add_argument('--lora-r', type=int, default=8,
                        help='LoRA rank (default: 8)')
    parser.add_argument('--lora-alpha', type=int, default=16,
                        help='LoRA alpha (default: 16)')
    parser.add_argument('--lora-dropout', type=float, default=0.1,
                        help='LoRA dropout (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='echo_npu:0',
                        help='Device to use (default: echo_npu:0)')
    parser.add_argument('--save-steps', type=int, default=5,
                        help='Save checkpoint every N steps (default: 5)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Qwen LoRA Fine-tuning Training")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"ECHO-NPU available: {torch.echo_npu.is_available()}")
    print(f"ECHO-NPU device count: {torch.echo_npu.device_count()}")
    print()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    print(f"Output directory: {args.output}")
    
    # 加载模型和tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer(args.model, device)
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 配置 LoRA
    try:
        model = setup_lora(
            model, 
            lora_r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout
        )
    except Exception as e:
        print(f"✗ Failed to setup LoRA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 将模型移到设备
    print(f"\nMoving model to {device}...")
    try:
        model = model.to(device)
        print("  ✓ Model moved to device")
    except Exception as e:
        print(f"  ✗ Failed to move model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 准备训练数据
    try:
        dataset = prepare_training_data(args.data, tokenizer, max_length=args.max_length)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=0  # 设置为0以避免多进程问题
        )
    except Exception as e:
        print(f"✗ Failed to prepare data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 创建优化器（只对可训练参数）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    print(f"\n✓ Optimizer created (learning rate: {args.lr})")
    
    # 训练循环
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    all_losses = []
    
    try:
        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            print("-" * 60)
            
            avg_loss = train_epoch(model, dataloader, optimizer, device, epoch)
            all_losses.append(avg_loss)
            
            print(f"\nEpoch {epoch} completed. Average loss: {avg_loss:.6f}")
            
            # 保存检查点
            if epoch % args.save_steps == 0 or epoch == args.epochs:
                save_checkpoint(model, tokenizer, args.output, epoch, avg_loss)
        
        # 保存最终模型
        final_dir = os.path.join(args.output, "final_model")
        os.makedirs(final_dir, exist_ok=True)
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"\n✓ Final model saved to {final_dir}")
        
        # 打印训练总结
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"Total epochs: {args.epochs}")
        print(f"Final loss: {all_losses[-1]:.6f}")
        print(f"Best loss: {min(all_losses):.6f}")
        print(f"Model saved to: {args.output}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✓ Training completed successfully!")


if __name__ == "__main__":
    main()
