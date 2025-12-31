import torch
from transformers import AutoModel
import os

def format_params(num):
    """将数字转换为 M (百万) 或 B (十亿) 格式"""
    if num >= 1e9:
        return f"{num / 1e9:.2f} B (十亿)"
    elif num >= 1e6:
        return f"{num / 1e6:.2f} M (百万)"
    else:
        return f"{num}"

def count_parameters(model, prefix=""):
    """统计模型参数的具体函数"""
    # 总参数
    total_params = sum(p.numel() for p in model.parameters())
    # 训练参数 (如果是加载的预训练模型，默认通常都是 True，除非你手动冻结了)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"[{prefix}] 统计结果:")
    print(f"  - 总参数量: {format_params(total_params)} ({total_params:,})")
    print(f"  - 可训练参数: {format_params(trainable_params)}")
    print("-" * 40)
    return total_params

def main():
    # ==========================================
    # 在这里修改你的本地模型路径
    # ==========================================
    model_path = "/ldap_shared/home/s_rlx/hllm/img_text_retrieval/code/local_models/siglip2-base-patch16-224/" 
    
    # 如果你在当前目录下，可以直接写文件夹名，例如：
    # model_path = "./siglip2-base-patch16-224"

    print(f"正在加载模型: {model_path} ...")
    
    try:
        # 加载模型
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请检查路径是否正确，或 transformers 库是否已更新到最新版本 (pip install -U transformers)")
        return

    print("=" * 40)
    print(f"模型结构: {model.__class__.__name__}")
    print("=" * 40)

    # 1. 统计整个模型的参数
    total = count_parameters(model, prefix="整体模型 (Total)")

    # 2. 尝试分别统计 Vision 和 Text 模块 (SigLIP 结构通常包含这两部分)
    if hasattr(model, "vision_model"):
        count_parameters(model.vision_model, prefix="视觉编码器 (Vision Encoder)")
    
    if hasattr(model, "text_model"):
        count_parameters(model.text_model, prefix="文本编码器 (Text Encoder)")

if __name__ == "__main__":
    main()