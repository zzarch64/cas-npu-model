#!/usr/bin/env python
"""
Qwen 模型推理脚本

这个脚本用于加载 Qwen 模型并进行文本生成和自然语言理解。

运行方式:
    python examples/qwen_inference.py [--model MODEL_NAME] [--prompt PROMPT] [--interactive] [--seed SEED]
"""

import sys
import os
import argparse
import random
import torch
from typing import Optional, List

# 添加扩展路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入cas_npu扩展
try:
    import cas_npu
    print("✓ CAS-NPU extension imported successfully")
except ImportError as e:
    print(f"✗ Failed to import CAS-NPU extension: {e}")
    sys.exit(1)

# ============ Monkey Patch: 修复 masked_fill_ 内存问题 ============
# cpu_fallback 处理 inplace 操作时可能与 functorch 的 dynamic layer 冲突
# 这里用非 inplace 版本替代 inplace 版本
_original_masked_fill_ = torch.Tensor.masked_fill_

def _patched_masked_fill_(self, mask, value):
    """使用非 inplace 版本实现 masked_fill_，避免内存问题"""
    # 检查是否在 cas_npu 设备上
    if self.device.type == 'privateuseone':
        # 使用非 inplace 版本，然后 copy_ 结果
        result = torch.masked_fill(self, mask, value)
        self.copy_(result)
        return self
    else:
        # 其他设备使用原始实现
        return _original_masked_fill_(self, mask, value)

torch.Tensor.masked_fill_ = _patched_masked_fill_
print("✓ Patched masked_fill_ for CAS-NPU compatibility")

# 检查 transformers 是否安装
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    print("✓ transformers library found")
except ImportError:
    print("✗ transformers library not found")
    print("  Please install: pip install transformers")
    sys.exit(1)

# 检查 peft 是否安装（用于加载 LoRA 模型）
try:
    from peft import PeftModel
    print("✓ peft library found (can load LoRA models)")
    PEFT_AVAILABLE = True
except ImportError:
    print("⚠ peft library not found (cannot load LoRA models)")
    PEFT_AVAILABLE = False


def set_seed(seed):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.cas_npu, 'manual_seed'):
        torch.cas_npu.manual_seed(seed)
    print(f"✓ Random seed set to {seed}")


def load_model_and_tokenizer(model_path: str, lora_path: Optional[str] = None, device: str = 'cas_npu:0'):
    """加载模型和tokenizer"""
    print(f"\nLoading model: {model_path}")
    print("  (This may take a while on first run...)")
    
    try:
        # 尝试从缓存加载
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
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
                model_path, 
                trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=torch.float32,
            )
            print("  ✓ Model loaded")
        except Exception as e2:
            print(f"  ✗ Failed to load model: {e2}")
            # 尝试替代模型
            try:
                model_path = "Qwen/Qwen2-0.5B"
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, 
                    trust_remote_code=True, 
                    local_files_only=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    dtype=torch.float32,
                    local_files_only=True,
                )
                print("  ✓ Alternative model loaded (from cache)")
            except Exception as e3:
                print(f"  ✗ Failed to load alternative model: {e3}")
                raise
    
    # 如果提供了 LoRA 路径，加载 LoRA 适配器
    if lora_path and PEFT_AVAILABLE:
        print(f"\nLoading LoRA adapter from: {lora_path}")
        try:
            model = PeftModel.from_pretrained(model, lora_path)
            print("  ✓ LoRA adapter loaded")
        except Exception as e:
            print(f"  ⚠ Failed to load LoRA adapter: {e}")
            print("  Continuing with base model...")
    
    # 设置 pad_token（如果不存在）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  ✓ Set pad_token to eos_token")
    
    # 将模型移到设备
    print(f"\nMoving model to {device}...")
    try:
        model = model.to(device)
        model.eval()  # 设置为评估模式
        print("  ✓ Model moved to device and set to eval mode")
    except Exception as e:
        print(f"  ✗ Failed to move model: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    return model, tokenizer


def generate_text(
    model, 
    tokenizer, 
    prompt: str, 
    device: str = 'cas_npu:0',
    max_length: int = 512,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    repetition_penalty: float = 1.1,
    show_progress: bool = True
):
    """生成文本"""
    # 编码输入，显式创建 attention_mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    else:
        # 如果没有 attention_mask，创建一个全1的mask（所有token都是有效的）
        attention_mask = torch.ones_like(input_ids).to(device)
    
    # 【调试信息】打印输入和 attention_mask 信息
    print(f"\n[DEBUG] Input information:")
    print(f"  Prompt: {prompt}")
    print(f"  Input IDs shape: {input_ids.shape}, dtype: {input_ids.dtype}")
    print(f"  Input IDs (first 10): {input_ids[0, :min(10, input_ids.shape[1])].cpu().tolist()}")
    print(f"  Attention mask shape: {attention_mask.shape}, dtype: {attention_mask.dtype}")
    print(f"  Attention mask (first 10): {attention_mask[0, :min(10, attention_mask.shape[1])].cpu().tolist()}")
    print(f"  Attention mask sum (should be > 0): {attention_mask.sum().item()}")
    # 使用 decode 正确显示中文，而不是 convert_ids_to_tokens（会显示 byte-level BPE 编码）
    print(f"  Decoded input: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
    
    # 生成配置
    generation_config = GenerationConfig(
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # 生成文本
    if show_progress:
        print("Generating (this may take a while)...", flush=True)
    
    # 【重要】Hook 模型的 forward 方法，验证 attention_mask 的使用
    original_forward = model.forward
    forward_call_count = [0]
    attention_mask_used = [False]
    
    def hooked_forward(*args, **kwargs):
        forward_call_count[0] += 1
        if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
            attention_mask_used[0] = True
            am = kwargs['attention_mask']
            if forward_call_count[0] <= 3:  # 只打印前几次
                print(f"[HOOK] Forward call #{forward_call_count[0]}: attention_mask shape={am.shape}, sum={am.sum().item()}")
        elif forward_call_count[0] <= 3:
            print(f"[HOOK] Forward call #{forward_call_count[0]}: WARNING - no attention_mask!")
        return original_forward(*args, **kwargs)
    
    model.forward = hooked_forward
    
    # 执行生成
    with torch.no_grad():
        try:
            generate_kwargs = {
                "input_ids": input_ids,
                "generation_config": generation_config,
            }
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask
                print(f"[DEBUG] Passing attention_mask to model.generate()")
            else:
                print(f"[DEBUG] WARNING: attention_mask is None!")
            outputs = model.generate(**generate_kwargs)
            
            # 恢复原始 forward
            model.forward = original_forward
            
            print(f"[DEBUG] Total forward calls: {forward_call_count[0]}")
            print(f"[DEBUG] Attention mask was used: {attention_mask_used[0]}")
        except Exception as e:
            print(f"  ⚠ Generation error: {e}")
            print("  Trying with simpler generation config...")
            # 回退到更简单的配置
            try:
                generate_kwargs = {
                    "input_ids": input_ids,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "temperature": temperature if do_sample else None,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                if attention_mask is not None:
                    generate_kwargs["attention_mask"] = attention_mask
                outputs = model.generate(**generate_kwargs)
            except Exception as e2:
                print(f"  ⚠ Second generation attempt also failed: {e2}")
                raise  # 重新抛出异常，让调用者处理
    
    # 【调试信息】打印输出信息
    print(f"\n[DEBUG] Output information:")
    print(f"  Output shape: {outputs.shape}")
    print(f"  Input length: {input_ids.shape[1]}")
    print(f"  Output length: {outputs.shape[1]}")
    
    # 解码输出
    # 只解码新生成的部分（去掉输入部分）
    generated_ids = outputs[0][input_ids.shape[1]:]
    
    # 使用 decode 正确显示中文
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"  Full output: {full_text[:200]}...")
    print(f"  Generated only: {generated_text[:200]}...")
    
    return generated_text


def interactive_mode(model, tokenizer, device: str = 'cas_npu:0', **generation_kwargs):
    """交互式对话模式"""
    print("\n" + "=" * 60)
    print("Interactive Mode")
    print("=" * 60)
    print("Enter your prompts (type 'quit' or 'exit' to exit)")
    print("-" * 60)
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # 构建提示（可以包含对话历史）
            if conversation_history:
                # 使用对话历史
                prompt = "\n".join(conversation_history) + f"\nUser: {user_input}\nAssistant:"
            else:
                # 首次输入
                prompt = f"User: {user_input}\nAssistant:"
            
            print("Assistant: ", end="", flush=True)
            
            # 生成回复
            response = generate_text(
                model, 
                tokenizer, 
                prompt, 
                device=device,
                **generation_kwargs
            )
            
            print(response)
            
            # 更新对话历史（限制长度以避免过长）
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Assistant: {response}")
            
            # 限制历史长度（保留最近5轮对话）
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


def batch_inference(model, tokenizer, prompts: List[str], device: str = 'cas_npu:0', **generation_kwargs):
    """批量推理模式"""
    print("\n" + "=" * 60)
    print("Batch Inference")
    print("=" * 60)
    
    results = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Prompt: {prompt}")
        print("-" * 60)
        
        try:
            response = generate_text(
                model, 
                tokenizer, 
                prompt, 
                device=device,
                **generation_kwargs
            )
            
            print(f"Response: {response}")
            results.append({
                'prompt': prompt,
                'response': response
            })
        
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                'prompt': prompt,
                'response': f"Error: {e}"
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Qwen Model Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 单次推理
  python examples/qwen_inference.py --prompt "Hello, how are you?"
  
  # 交互式模式
  python examples/qwen_inference.py --interactive
  
  # 使用 LoRA 微调后的模型
  python examples/qwen_inference.py --model Qwen/Qwen2.5-0.5B --lora ./qwen_lora_output/final_model --interactive
  
  # 自定义生成参数
  python examples/qwen_inference.py --prompt "Tell me a joke" --temperature 0.9 --max-new-tokens 100
        """
    )
    
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-0.5B",
                        help='Model name or path (default: Qwen/Qwen2.5-0.5B)')
    parser.add_argument('--lora', type=str, default=None,
                        help='Path to LoRA adapter (optional)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Input prompt for single inference')
    parser.add_argument('--interactive', action='store_true',
                        help='Enable interactive mode')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cas_npu:0',
                        help='Device to use (default: cas_npu:0)')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length (default: 512)')
    parser.add_argument('--max-new-tokens', type=int, default=256,
                        help='Maximum number of new tokens to generate (default: 256)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature (default: 0.7)')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Top-p (nucleus) sampling (default: 0.9)')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k sampling (default: 50)')
    parser.add_argument('--do-sample', action='store_true', default=True,
                        help='Use sampling (default: True)')
    parser.add_argument('--no-sample', dest='do_sample', action='store_false',
                        help='Disable sampling (use greedy decoding)')
    parser.add_argument('--repetition-penalty', type=float, default=1.1,
                        help='Repetition penalty (default: 1.1)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Qwen Model Inference")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CAS-NPU available: {torch.cas_npu.is_available()}")
    print(f"CAS-NPU device count: {torch.cas_npu.device_count()}")
    print()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 加载模型和tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer(
            args.model, 
            lora_path=args.lora,
            device=str(device)
        )
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 准备生成参数
    generation_kwargs = {
        'max_length': args.max_length,
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'do_sample': args.do_sample,
        'repetition_penalty': args.repetition_penalty,
    }
    
    print(f"\nGeneration config:")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  temperature: {args.temperature}")
    print(f"  top_p: {args.top_p}")
    print(f"  top_k: {args.top_k}")
    print(f"  do_sample: {args.do_sample}")
    print(f"  repetition_penalty: {args.repetition_penalty}")
    
    # 运行推理
    try:
        if args.interactive:
            # 交互式模式
            interactive_mode(model, tokenizer, device=str(device), **generation_kwargs)
        elif args.prompt:
            # 单次推理
            print("\n" + "=" * 60)
            print("Single Inference")
            print("=" * 60)
            print(f"Prompt: {args.prompt}")
            print("-" * 60)
            
            response = generate_text(
                model, 
                tokenizer, 
                args.prompt, 
                device=str(device),
                **generation_kwargs
            )
            
            print(f"Response: {response}")
        else:
            # 默认：使用示例提示
            example_prompts = [
                "Hello, how are you?",
                "What is machine learning?",
                "Tell me a joke.",
            ]
            
            batch_inference(model, tokenizer, example_prompts, device=str(device), **generation_kwargs)
    
    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✓ Inference completed!")


if __name__ == "__main__":
    main()
