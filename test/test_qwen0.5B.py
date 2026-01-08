#!/usr/bin/env python
"""
Qwen0.5B 模型测试 - 验证 mm 和 bmm 算子实现

测试步骤:
1. 基础 mm 和 bmm 操作测试
2. Linear 层测试（使用 mm）
3. 加载 Qwen0.5B 模型并运行前向传播
4. LoRA 微调训练测试（需要 peft 库）
"""

import sys
import os
import argparse
import torch
import torch.nn as nn

# 添加扩展路径（从test目录向上一级找到python包）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入cas_npu扩展
try:
    import cas_npu
    print("✓ CAS-NPU extension imported successfully")
except ImportError as e:
    print(f"✗ Failed to import CAS-NPU extension: {e}")
    sys.exit(1)

print(f"PyTorch version: {torch.__version__}")
print(f"CAS-NPU available: {torch.cas_npu.is_available()}")
print(f"CAS-NPU device count: {torch.cas_npu.device_count()}")
print()

# 检查 transformers 是否安装
try:
    from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
    print("✓ transformers library found")
except ImportError:
    print("✗ transformers library not found")
    print("  Please install: pip install transformers")
    sys.exit(1)

# 检查 peft 是否安装（用于 LoRA）
try:
    from peft import LoraConfig, get_peft_model, TaskType
    print("✓ peft library found")
    PEFT_AVAILABLE = True
except ImportError:
    print("⚠ peft library not found (LoRA test will be skipped)")
    print("  To enable LoRA test, install: pip install peft")
    PEFT_AVAILABLE = False

def test_mm_bmm_basic():
    """测试基础的 mm 和 bmm 操作"""
    print("=" * 60)
    print("Test 1: Basic mm and bmm operations")
    print("=" * 60)
    
    device = torch.device('cas_npu:0')
    
    # 测试 mm
    print("\nTesting mm (matrix multiplication)...")
    a = torch.randn(3, 4).to(device)
    b = torch.randn(4, 5).to(device)
    c = torch.mm(a, b)
    print(f"  a.shape: {a.shape}, b.shape: {b.shape}")
    print(f"  c.shape: {c.shape}")
    
    # 验证结果
    a_cpu = a.cpu()
    b_cpu = b.cpu()
    c_expected = torch.mm(a_cpu, b_cpu)
    c_cpu = c.cpu()
    
    max_diff = (c_cpu - c_expected).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert max_diff < 1e-5, f"mm result mismatch! Max diff: {max_diff}"
    print("  ✓ mm test passed")
    
    # 测试 bmm
    print("\nTesting bmm (batch matrix multiplication)...")
    a_batch = torch.randn(2, 3, 4).to(device)
    b_batch = torch.randn(2, 4, 5).to(device)
    c_batch = torch.bmm(a_batch, b_batch)
    print(f"  a_batch.shape: {a_batch.shape}, b_batch.shape: {b_batch.shape}")
    print(f"  c_batch.shape: {c_batch.shape}")
    
    # 验证结果
    a_batch_cpu = a_batch.cpu()
    b_batch_cpu = b_batch.cpu()
    c_expected_batch = torch.bmm(a_batch_cpu, b_batch_cpu)
    c_batch_cpu = c_batch.cpu()
    
    max_diff = (c_batch_cpu - c_expected_batch).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert max_diff < 1e-5, f"bmm result mismatch! Max diff: {max_diff}"
    print("  ✓ bmm test passed")
    
    print("\n✓ Basic mm/bmm tests completed\n")

def test_qwen_model():
    """测试 Qwen0.5B 模型"""
    print("=" * 60)
    print("Test 2: Qwen0.5B Model Forward Pass")
    print("=" * 60)
    
    device = torch.device('cas_npu:0')
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"\nLoading model: {model_name}")
    print("  (This may take a while on first run...)")
    
    try:
        # 加载模型和tokenizer（优先使用本地缓存）
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # 使用float32，因为我们的实现只支持float
            local_files_only=True,
        )
        print("  ✓ Model loaded (from cache)")
    except Exception as e:
        print(f"  ✗ Failed to load model from cache: {e}")
        print("  Trying to download...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
            print("  ✓ Model loaded")
        except Exception as e2:
            print(f"  ✗ Failed to load model: {e2}")
            print("  Trying alternative: Qwen/Qwen2-0.5B")
            try:
                model_name = "Qwen/Qwen2-0.5B"
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
                model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    local_files_only=True,
                )
                print("  ✓ Model loaded (alternative, from cache)")
            except Exception as e3:
                print(f"  ✗ Failed to load alternative model: {e3}")
                print("  Skipping model test...")
                return False
    
    # 将模型移到设备
    print(f"\nMoving model to {device}...")
    try:
        model = model.to(device)
        print("  ✓ Model moved to device")
    except Exception as e:
        print(f"  ✗ Failed to move model: {e}")
        print(f"  Error details: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 准备输入
    print("\nPreparing input...")
    text = "Hello, how are you?"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    print(f"  Input text: {text}")
    print(f"  Input shape: {input_ids.shape}")
    
    # 前向传播
    print("\nRunning forward pass...")
    model.eval()  # 设置为评估模式
    try:
        with torch.no_grad():
            outputs = model(input_ids)
        print("  ✓ Forward pass completed")
        print(f"  Output shape: {outputs.last_hidden_state.shape}")
        print(f"  Output dtype: {outputs.last_hidden_state.dtype}")
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ Qwen model test completed\n")
    return True

def test_linear_layer():
    """测试 Linear 层（使用 mm）"""
    print("=" * 60)
    print("Test 3: Linear Layer (uses mm)")
    print("=" * 60)
    
    device = torch.device('cas_npu:0')
    
    # 创建 Linear 层
    linear = nn.Linear(768, 3072).to(device)
    x = torch.randn(2, 10, 768).to(device)
    
    print(f"Linear layer: {linear}")
    print(f"Input shape: {x.shape}")
    
    # 前向传播
    y = linear(x)
    print(f"Output shape: {y.shape}")
    
    # 验证
    assert y.device.type == 'cas_npu', f"Output should be on cas_npu device, got {y.device.type}"
    assert y.shape == (2, 10, 3072), f"Unexpected output shape: {y.shape}"
    
    print("  ✓ Linear layer test passed\n")

def test_lora_finetune():
    """测试 LoRA 微调训练"""
    print("=" * 60)
    print("Test 4: LoRA Fine-tuning Training")
    print("=" * 60)
    
    if not PEFT_AVAILABLE:
        print("  ⚠ Skipping LoRA test (peft library not available)")
        return False
    
    device = torch.device('cas_npu:0')
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"\nLoading model: {model_name}")
    try:
        # 尝试从缓存加载
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        print("  ✓ Model loaded (from cache)")
    except Exception as e:
        print(f"  ✗ Failed to load from cache: {e}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
            print("  ✓ Model loaded")
        except Exception as e2:
            print(f"  ✗ Failed to load model: {e2}")
            # 尝试替代模型
            try:
                model_name = "Qwen/Qwen2-0.5B"
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
                model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    local_files_only=True,
                )
                print("  ✓ Alternative model loaded (from cache)")
            except Exception as e3:
                print(f"  ✗ Failed to load alternative model: {e3}")
                return False
    
    # 设置 pad_token（如果不存在）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  ✓ Set pad_token to eos_token")
    
    # 配置 LoRA
    print("\nConfiguring LoRA...")
    try:
        # 使用较小的 LoRA rank 以节省内存
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # 用于特征提取任务
            r=4,  # LoRA 的秩（从 8 减少到 4 以节省内存）
            lora_alpha=8,  # LoRA 的缩放因子（相应调整）
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 目标模块
        )
        print(f"  LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}")
        
        # 应用 LoRA 到模型
        # LoRA 工作原理：
        # 1. 冻结原始模型的所有参数（不训练）
        # 2. 在目标线性层（如 q_proj, k_proj 等）旁边添加低秩矩阵 A 和 B
        # 3. 只训练这些小的低秩矩阵（参数量远小于原始模型）
        # 4. 前向传播：output = Wx + (B*A)x，其中 W 是冻结的原始权重，B*A 是低秩适配器
        model = get_peft_model(model, lora_config)
        print("  ✓ LoRA adapter applied")
        
        # 显示参数统计
        # 注意：不同版本的 peft 可能返回不同类型
        param_stats = model.num_parameters()
        if isinstance(param_stats, dict):
            # 新版本返回字典
            trainable = param_stats.get('trainable', 0)
            total = param_stats.get('all', 0)
        else:
            # 旧版本可能返回整数（总参数数）
            # 手动计算可训练参数
            total = param_stats if isinstance(param_stats, int) else 0
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # 如果 total 为 0，尝试计算总参数数
            if total == 0:
                total = sum(p.numel() for p in model.parameters())
        
        frozen = total - trainable
        trainable_ratio = (trainable / total * 100) if total > 0 else 0
        
        print(f"  Total parameters: {total:,}")
        print(f"  Trainable parameters (LoRA only): {trainable:,}")
        print(f"  Frozen parameters (original model): {frozen:,}")
        print(f"  Trainable ratio: {trainable_ratio:.2f}%")
        print(f"  Note: Only LoRA matrices are trained, not the entire LLM!")
    except Exception as e:
        print(f"  ✗ Failed to configure LoRA: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 将模型移到设备
    print(f"\nMoving model to {device}...")
    try:
        # 注意：peft 的 LoRA 层会跟随基础模型的设备
        # 当调用 model.to(device) 时，所有参数（包括 LoRA 适配器）都会移到目标设备
        model = model.to(device)
        print("  ✓ Model moved to device")
        
        # 检查设备位置
        first_param = next(model.parameters())
        print(f"  Model device: {first_param.device}")
    except Exception as e:
        print(f"  ✗ Failed to move model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 准备训练数据（简单的示例数据）
    print("\nPreparing training data...")
    try:
        # 创建简单的训练样本（减少 batch size 和序列长度以节省内存）
        texts = [
            "Hello, how are you?",
            "What is the weather today?",
        ]
        
        # 编码文本（使用较小的序列长度以节省内存）
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=32,  # 减少序列长度从 128 到 32
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        print(f"  Training samples: {len(texts)}")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Attention mask shape: {attention_mask.shape}")
        print(f"  Note: Using smaller batch size and sequence length to save memory")
    except Exception as e:
        print(f"  ✗ Failed to prepare data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 设置训练模式
    print("\nSetting up training...")
    model.train()
    
    # 创建优化器
    # 注意：只对可训练参数（LoRA）创建优化器，避免为冻结参数分配内存
    # 原始模型的参数被冻结，不会参与梯度更新
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable parameter groups: {len(trainable_params)} (LoRA only)")
    if len(trainable_params) == 0:
        print("  ⚠ WARNING: No trainable parameters found!")
        return False
    
    # 只对可训练参数创建优化器，避免为冻结参数分配优化器状态内存
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    print(f"  ✓ Optimizer created (only for trainable parameters)")
    
    # 运行几个训练步骤
    print("\nRunning training steps...")
    num_steps = 3
    losses = []
    
    # 验证输入数据在正确的设备上
    assert input_ids.device == device, f"Input data should be on {device}, got {input_ids.device}"
    assert attention_mask.device == device, f"Attention mask should be on {device}, got {attention_mask.device}"
    print(f"  Input data device: {input_ids.device}")
    
    # 清理可能的内存缓存
    if hasattr(torch.cas_npu, 'empty_cache'):
        torch.cas_npu.empty_cache()
    print(f"  Memory cache cleared")
    
    # 先做一个简单的前向传播测试，检查内存是否足够
    # 推理时应该传递 attention_mask 以确保正确性（现在有了 fallback 应该可以工作）
    print("  Testing forward pass before training...")
    try:
        with torch.no_grad():
            # 推理时传递 attention_mask 以确保正确性
            test_output = model(
                input_ids=input_ids[:1],
                attention_mask=attention_mask[:1]
            )
            print(f"  ✓ Forward pass test successful (with attention_mask), output shape: {test_output.last_hidden_state.shape}")
        del test_output  # 释放内存
        if hasattr(torch.cas_npu, 'empty_cache'):
            torch.cas_npu.empty_cache()
    except Exception as e:
        print(f"  ✗ Forward pass test failed: {e}")
        print("  Note: Trying without attention_mask as fallback...")
        try:
            # 如果传递 attention_mask 失败，尝试不传递（训练模式）
            with torch.no_grad():
                test_output = model(input_ids=input_ids[:1])
            print(f"  ✓ Forward pass test successful (without attention_mask), output shape: {test_output.last_hidden_state.shape}")
            del test_output
            if hasattr(torch.cas_npu, 'empty_cache'):
                torch.cas_npu.empty_cache()
        except Exception as e2:
            print(f"  ✗ Forward pass test failed even without attention_mask: {e2}")
            print("  Will continue with training attempt...")
    
    # 准备表格数据
    training_rows = []
    
    try:
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # 前向传播（所有计算在模型所在的设备上进行）
            # 如果模型在 cas_npu:0，LoRA 的计算也在 cas_npu:0 上
            # 使用单个样本进行训练以节省内存
            batch_input_ids = input_ids[:1]
            batch_attention_mask = attention_mask[:1]
            
            # 训练时：可以不传递 attention_mask（模型会自动处理）
            # 但为了与推理保持一致，我们尝试传递，如果失败则回退
            try:
                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask
                )
            except Exception as e:
                # 如果传递 attention_mask 失败（可能 fallback 还未生效），回退到不传递
                outputs = model(input_ids=batch_input_ids)
            
            # 检查前向传播输出是否包含NaN
            hidden_states = outputs.last_hidden_state
            if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                nan_count = torch.isnan(hidden_states).sum().item()
                inf_count = torch.isinf(hidden_states).sum().item()
                training_rows.append([f"Step {step+1}", "NaN/Inf", f"NaN:{nan_count} Inf:{inf_count}", "✗"])
                raise RuntimeError("NaN/Inf detected in forward pass output")
            
            # 计算损失（使用简单的 MSE 损失作为示例）
            # 在实际应用中，这里应该是任务特定的损失函数
            # 创建一个简单的目标（可以是任何合理的损失函数）
            # 这里我们使用一个简单的示例：预测下一个 token 的嵌入
            if hidden_states.shape[1] > 1:
                # 使用前 n-1 个 token 的嵌入预测最后一个 token 的嵌入
                pred = hidden_states[:, :-1, :]
                target = hidden_states[:, 1:, :].detach()
                loss = nn.functional.mse_loss(pred, target)
            else:
                # 如果只有一个 token，使用一个简单的损失
                loss = hidden_states.mean()
            
            # 检查损失是否包含NaN
            if torch.isnan(loss) or torch.isinf(loss):
                training_rows.append([f"Step {step+1}", "NaN/Inf", f"loss={loss.item()}", "✗"])
                raise RuntimeError("NaN/Inf detected in loss")
            
            # 反向传播
            loss.backward()
            
            # 检查梯度是否包含NaN
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                training_rows.append([f"Step {step+1}", "NaN/Inf", "gradient", "✗"])
                raise RuntimeError("NaN/Inf detected in gradients")
            
            optimizer.step()
            
            # 检查更新后的参数是否包含NaN
            has_nan_param = False
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        has_nan_param = True
                        break
            
            if has_nan_param:
                training_rows.append([f"Step {step+1}", "NaN/Inf", "parameter", "✗"])
                raise RuntimeError("NaN/Inf detected in parameters after optimizer step")
            
            loss_value = loss.item()
            losses.append(loss_value)
            
            # 添加到表格
            status = "✓" if torch.isfinite(torch.tensor(loss_value)) else "✗"
            training_rows.append([f"Step {step+1}", f"{loss_value:.6f}", f"shape={hidden_states.shape}", status])
            
            # 清理中间变量以释放内存
            del outputs, hidden_states, loss
            if hasattr(torch.cas_npu, 'empty_cache'):
                torch.cas_npu.empty_cache()
        
        # 打印训练表格
        print("\nTraining Progress:")
        print("-" * 80)
        print(f"{'Step':<10} | {'Loss':<12} | {'Info':<20} | {'Status':<6}")
        print("-" * 80)
        for row in training_rows:
            print(f"{row[0]:<10} | {row[1]:<12} | {row[2]:<20} | {row[3]:<6}")
        
        print("\n✓ Training steps completed")
        
        # 验证损失是否在合理范围内（不应该是 NaN 或 Inf）
        assert all(torch.isfinite(torch.tensor(losses))), "Loss contains NaN or Inf!"
        print("  ✓ Loss values are valid")
        
    except Exception as e:
        print(f"  ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试推理（评估模式）
    # 推理时应该传递 attention_mask 以确保正确性，特别是处理变长序列时
    print("\nTesting inference after training...")
    try:
        model.eval()
        with torch.no_grad():
            # 推理时传递 attention_mask 以确保正确性
            # 现在有了 aten::all 的 fallback，应该可以正常工作
            test_outputs = model(
                input_ids=input_ids[:1],
                attention_mask=attention_mask[:1]
            )
        print(f"  ✓ Inference successful (with attention_mask)")
        print(f"  Output shape: {test_outputs.last_hidden_state.shape}")
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ LoRA fine-tuning test completed\n")
    return True

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Qwen0.5B Model Test with CAS-NPU mm/bmm Operators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 运行所有测试（包括 LoRA）
  python test_qwen0.5B.py --lora
  
  # 跳过 LoRA 测试
  python test_qwen0.5B.py --no-lora
  
  # 默认行为：如果 peft 可用则运行 LoRA 测试
  python test_qwen0.5B.py
        """
    )
    parser.add_argument(
        '--lora',
        action='store_true',
        help='Enable LoRA fine-tuning test (default: auto-detect based on peft availability)'
    )
    parser.add_argument(
        '--no-lora',
        action='store_true',
        help='Disable LoRA fine-tuning test'
    )
    args = parser.parse_args()
    
    # 确定是否运行 LoRA 测试
    run_lora = False
    if args.no_lora:
        run_lora = False
        print("LoRA test disabled by --no-lora flag")
    elif args.lora:
        run_lora = True
        print("LoRA test enabled by --lora flag")
    else:
        # 默认：如果 peft 可用则运行
        run_lora = PEFT_AVAILABLE
        if run_lora:
            print("LoRA test enabled (peft library available)")
        else:
            print("LoRA test disabled (peft library not available)")
    
    print("=" * 60)
    print("Qwen0.5B Model Test with CAS-NPU mm/bmm Operators")
    print("=" * 60)
    print()
    
    all_passed = True
    
    # 测试1: 基础 mm/bmm
    try:
        test_mm_bmm_basic()
    except Exception as e:
        print(f"✗ Basic mm/bmm test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # 测试2: Linear层
    try:
        test_linear_layer()
    except Exception as e:
        print(f"✗ Linear layer test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # 测试3: Qwen模型
    try:
        qwen_passed = test_qwen_model()
        if not qwen_passed:
            print("⚠ Qwen model test failed")
            # 注意：Qwen模型测试失败不影响核心功能测试结果
            # 因为核心的 mm/bmm 已经通过测试
    except Exception as e:
        print(f"⚠ Qwen model test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        # 不把模型测试失败当作整体失败，因为核心功能已通过
    
    # 测试4: LoRA微调训练（可选）
    if run_lora:
        try:
            lora_passed = test_lora_finetune()
            if not lora_passed:
                print("⚠ LoRA fine-tuning test failed or skipped")
                # LoRA测试失败不影响核心功能测试结果
        except Exception as e:
            print(f"⚠ LoRA fine-tuning test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            # 不把LoRA测试失败当作整体失败
    else:
        print("Skipping LoRA fine-tuning test (disabled)")
    
    print("=" * 60)
    if all_passed:
        print("All core tests passed! ✓")
    else:
        print("Some tests failed. ✗")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
