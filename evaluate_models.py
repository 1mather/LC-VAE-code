#!/usr/bin/env python
"""
完整的模型对比评估脚本
用法: python evaluate_models.py --wfvae_ckpt path/to/wfvae.pth --v1_ckpt path/to/v1.pth
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
import json
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# 导入模型
from causalvideovae.model.vae.modeling_wfvae import WFVAEModel
from causalvideovae.model.vae.modeling_latent_wfvae import LatentWFVAEModelV1


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model, model_name, device='cuda'):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        
    def evaluate_quality(self, data_loader):
        """评估质量指标"""
        self.model.eval()
        
        metrics = {
            'psnr': [],
            'ssim': [],
            'mse': [],
            'mae': []
        }
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"[{self.model_name}] Quality"):
                if isinstance(batch, dict):
                    inputs = batch['video'].to(self.device)
                else:
                    inputs = batch.to(self.device)
                
                # 重建
                output = self.model(inputs, sample_posterior=False)
                recon = output.sample
                
                # 计算指标
                mse = F.mse_loss(recon, inputs).item()
                mae = F.l1_loss(recon, inputs).item()
                psnr = -10 * np.log10(mse) if mse > 0 else 100.0
                
                metrics['psnr'].append(psnr)
                metrics['mse'].append(mse)
                metrics['mae'].append(mae)
                
                # SSIM（简化版）
                ssim = self._calculate_simple_ssim(recon, inputs)
                metrics['ssim'].append(ssim)
        
        # 统计
        return {
            'psnr': {
                'mean': np.mean(metrics['psnr']),
                'std': np.std(metrics['psnr']),
                'min': np.min(metrics['psnr']),
                'max': np.max(metrics['psnr'])
            },
            'ssim': {
                'mean': np.mean(metrics['ssim']),
                'std': np.std(metrics['ssim'])
            },
            'mse': {
                'mean': np.mean(metrics['mse']),
                'std': np.std(metrics['mse'])
            },
            'mae': {
                'mean': np.mean(metrics['mae']),
                'std': np.std(metrics['mae'])
            }
        }
    
    def evaluate_efficiency(self, input_shape=(1, 3, 25, 256, 256)):
        """评估效率指标"""
        print(f"[{self.model_name}] Evaluating Efficiency...")
        
        # 1. 参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # 2. 模型大小
        temp_path = f'temp_{self.model_name}.pth'
        torch.save(self.model.state_dict(), temp_path)
        model_size_mb = Path(temp_path).stat().st_size / (1024 ** 2)
        Path(temp_path).unlink()
        
        # 3. 推理速度
        latency = self._measure_latency(input_shape, num_runs=100, warmup=10)
        
        # 4. 显存占用
        memory = self._measure_memory(input_shape)
        
        return {
            'parameters': {
                'total_m': total_params / 1e6,
                'trainable_m': trainable_params / 1e6,
            },
            'model_size_mb': model_size_mb,
            'latency': latency,
            'memory': memory
        }
    
    def _measure_latency(self, input_shape, num_runs=100, warmup=10):
        """测量推理延迟"""
        self.model.eval()
        input_tensor = torch.randn(*input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(input_tensor, sample_posterior=False)
        
        # 测量
        torch.cuda.synchronize()
        times = []
        
        with torch.no_grad():
            for _ in tqdm(range(num_runs), desc=f"[{self.model_name}] Latency"):
                start = time.time()
                _ = self.model(input_tensor, sample_posterior=False)
                torch.cuda.synchronize()
                times.append((time.time() - start) * 1000)  # ms
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'median_ms': np.median(times),
            'fps': input_shape[2] * 1000.0 / np.mean(times)  # 视频帧率
        }
    
    def _measure_memory(self, input_shape):
        """测量显存占用"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        input_tensor = torch.randn(*input_shape).to(self.device)
        
        # 推理显存
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor, sample_posterior=False)
        
        inference_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        # 训练显存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        self.model.train()
        output = self.model(input_tensor, sample_posterior=True)
        loss = F.mse_loss(output.sample, input_tensor)
        loss.backward()
        
        training_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        return {
            'inference_mb': inference_memory_mb,
            'training_mb': training_memory_mb,
            'ratio': training_memory_mb / inference_memory_mb if inference_memory_mb > 0 else 0
        }
    
    def _calculate_simple_ssim(self, pred, target, window_size=11):
        """简化的SSIM计算"""
        # 在帧级别计算
        B, C, T, H, W = pred.shape
        ssims = []
        
        for t in range(T):
            pred_frame = pred[:, :, t]
            target_frame = target[:, :, t]
            
            # 简化：使用MSE的逆作为近似
            mse = F.mse_loss(pred_frame, target_frame)
            ssim_approx = 1.0 / (1.0 + mse)
            ssims.append(ssim_approx.item())
        
        return np.mean(ssims)


def compare_models(results_wfvae, results_v1):
    """对比两个模型的结果"""
    comparison = {}
    
    # Quality对比
    quality_diff = {
        'psnr_diff': results_wfvae['quality']['psnr']['mean'] - results_v1['quality']['psnr']['mean'],
        'ssim_diff': results_wfvae['quality']['ssim']['mean'] - results_v1['quality']['ssim']['mean'],
        'mse_diff': results_wfvae['quality']['mse']['mean'] - results_v1['quality']['mse']['mean'],
    }
    
    # 质量赢家
    quality_winner = 'WFVAE' if quality_diff['psnr_diff'] > 0 else 'V1'
    
    comparison['quality'] = {
        'differences': quality_diff,
        'winner': quality_winner,
        'winner_margin_psnr_db': abs(quality_diff['psnr_diff'])
    }
    
    # Efficiency对比
    params_ratio = results_wfvae['efficiency']['parameters']['total_m'] / results_v1['efficiency']['parameters']['total_m']
    speed_ratio = results_wfvae['efficiency']['latency']['mean_ms'] / results_v1['efficiency']['latency']['mean_ms']
    memory_ratio = results_wfvae['efficiency']['memory']['inference_mb'] / results_v1['efficiency']['memory']['inference_mb']
    
    # 效率得分（越高越好）
    eff_score_wfvae = 1.0 / (results_wfvae['efficiency']['parameters']['total_m'] * 
                               results_wfvae['efficiency']['latency']['mean_ms'] * 
                               results_wfvae['efficiency']['memory']['inference_mb'])
    eff_score_v1 = 1.0 / (results_v1['efficiency']['parameters']['total_m'] * 
                           results_v1['efficiency']['latency']['mean_ms'] * 
                           results_v1['efficiency']['memory']['inference_mb'])
    
    efficiency_winner = 'WFVAE' if eff_score_wfvae > eff_score_v1 else 'V1'
    
    comparison['efficiency'] = {
        'params_ratio': params_ratio,
        'speed_ratio': speed_ratio,
        'memory_ratio': memory_ratio,
        'winner': efficiency_winner,
        'efficiency_score_ratio': eff_score_wfvae / eff_score_v1
    }
    
    # Trade-off分析
    quality_eff_wfvae = results_wfvae['quality']['psnr']['mean'] * eff_score_wfvae * 1000
    quality_eff_v1 = results_v1['quality']['psnr']['mean'] * eff_score_v1 * 1000
    
    comparison['tradeoff'] = {
        'quality_efficiency_score': {
            'wfvae': quality_eff_wfvae,
            'v1': quality_eff_v1
        },
        'overall_winner': 'WFVAE' if quality_eff_wfvae > quality_eff_v1 else 'V1'
    }
    
    return comparison


def generate_report(results_wfvae, results_v1, comparison, save_path):
    """生成Markdown报告"""
    
    report = f"""# Model Comparison Report

Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

### Overall Winner: **{comparison['tradeoff']['overall_winner']}**

- **Quality Winner**: {comparison['quality']['winner']} (PSNR advantage: {comparison['quality']['winner_margin_psnr_db']:.2f} dB)
- **Efficiency Winner**: {comparison['efficiency']['winner']} (Score ratio: {comparison['efficiency']['efficiency_score_ratio']:.2f}x)

---

## 1. Quality Comparison

| Metric | WFVAE | V1 | Difference | Winner |
|--------|-------|-------|------------|--------|
| **PSNR (dB)** ↑ | {results_wfvae['quality']['psnr']['mean']:.2f} ± {results_wfvae['quality']['psnr']['std']:.2f} | {results_v1['quality']['psnr']['mean']:.2f} ± {results_v1['quality']['psnr']['std']:.2f} | {comparison['quality']['differences']['psnr_diff']:+.2f} | **{comparison['quality']['winner']}** |
| **SSIM** ↑ | {results_wfvae['quality']['ssim']['mean']:.4f} ± {results_wfvae['quality']['ssim']['std']:.4f} | {results_v1['quality']['ssim']['mean']:.4f} ± {results_v1['quality']['ssim']['std']:.4f} | {comparison['quality']['differences']['ssim_diff']:+.4f} | **{comparison['quality']['winner']}** |
| **MSE** ↓ | {results_wfvae['quality']['mse']['mean']:.6f} | {results_v1['quality']['mse']['mean']:.6f} | {comparison['quality']['differences']['mse_diff']:+.6f} | - |
| **MAE** ↓ | {results_wfvae['quality']['mae']['mean']:.6f} | {results_v1['quality']['mae']['mean']:.6f} | - | - |

---

## 2. Efficiency Comparison

### Parameters

| Model | Total (M) | Trainable (M) | Model Size (MB) |
|-------|-----------|---------------|-----------------|
| **WFVAE** | {results_wfvae['efficiency']['parameters']['total_m']:.2f} | {results_wfvae['efficiency']['parameters']['trainable_m']:.2f} | {results_wfvae['efficiency']['model_size_mb']:.2f} |
| **V1** | {results_v1['efficiency']['parameters']['total_m']:.2f} | {results_v1['efficiency']['parameters']['trainable_m']:.2f} | {results_v1['efficiency']['model_size_mb']:.2f} |
| **Ratio** | {comparison['efficiency']['params_ratio']:.2f}x | - | - |

### Speed

| Model | Mean (ms) | Std (ms) | FPS | Winner |
|-------|-----------|----------|-----|--------|
| **WFVAE** | {results_wfvae['efficiency']['latency']['mean_ms']:.2f} | {results_wfvae['efficiency']['latency']['std_ms']:.2f} | {results_wfvae['efficiency']['latency']['fps']:.1f} | - |
| **V1** | {results_v1['efficiency']['latency']['mean_ms']:.2f} | {results_v1['efficiency']['latency']['std_ms']:.2f} | {results_v1['efficiency']['latency']['fps']:.1f} | - |
| **Ratio** | {comparison['efficiency']['speed_ratio']:.2f}x | - | - | **{comparison['efficiency']['winner']}** |

### Memory

| Model | Inference (MB) | Training (MB) | Ratio |
|-------|----------------|---------------|-------|
| **WFVAE** | {results_wfvae['efficiency']['memory']['inference_mb']:.2f} | {results_wfvae['efficiency']['memory']['training_mb']:.2f} | {results_wfvae['efficiency']['memory']['ratio']:.2f}x |
| **V1** | {results_v1['efficiency']['memory']['inference_mb']:.2f} | {results_v1['efficiency']['memory']['training_mb']:.2f} | {results_v1['efficiency']['memory']['ratio']:.2f}x |
| **Ratio** | {comparison['efficiency']['memory_ratio']:.2f}x | - | - |

---

## 3. Trade-off Analysis

### Quality-Efficiency Score

- **WFVAE**: {comparison['tradeoff']['quality_efficiency_score']['wfvae']:.2f}
- **V1**: {comparison['tradeoff']['quality_efficiency_score']['v1']:.2f}
- **Winner**: **{comparison['tradeoff']['overall_winner']}**

*Score = PSNR × (1 / (Params × Latency × Memory)) × 1000*

---

## 4. Recommendations

### Use WFVAE if:
- Quality is the top priority
- Working with high-fidelity video compression
- Explicit frequency domain modeling is beneficial
- Computational resources are abundant

### Use V1 if:
- Need better speed/efficiency
- Working on resource-constrained devices
- Prefer end-to-end learning
- Want simpler architecture

---

## Appendix: Detailed Statistics

### WFVAE Quality Range
- PSNR: [{results_wfvae['quality']['psnr']['min']:.2f}, {results_wfvae['quality']['psnr']['max']:.2f}] dB

### V1 Quality Range
- PSNR: [{results_v1['quality']['psnr']['min']:.2f}, {results_v1['quality']['psnr']['max']:.2f}] dB

### Speed Statistics
- WFVAE Latency: {results_wfvae['efficiency']['latency']['min_ms']:.2f} (min) - {results_wfvae['efficiency']['latency']['max_ms']:.2f} (max) ms
- V1 Latency: {results_v1['efficiency']['latency']['min_ms']:.2f} (min) - {results_v1['efficiency']['latency']['max_ms']:.2f} (max) ms

---

*End of Report*
"""
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(f"\n✅ Report saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate WFVAE vs Latent_WFVAE_V1')
    parser.add_argument('--wfvae_ckpt', type=str, required=True, help='Path to WFVAE checkpoint')
    parser.add_argument('--v1_ckpt', type=str, required=True, help='Path to V1 checkpoint')
    parser.add_argument('--data_path', type=str, help='Path to test data')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=100, help='Number of test samples')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("MODEL COMPARISON EVALUATION")
    print("="*80)
    
    # 加载模型
    print("\n[1/5] Loading Models...")
    model_wfvae = WFVAEModel()
    model_wfvae.init_from_ckpt(args.wfvae_ckpt)
    
    model_v1 = LatentWFVAEModelV1()
    model_v1.init_from_ckpt(args.v1_ckpt)
    
    # 创建评估器
    evaluator_wfvae = ModelEvaluator(model_wfvae, 'WFVAE', args.device)
    evaluator_v1 = ModelEvaluator(model_v1, 'V1', args.device)
    
    # 准备数据
    print("\n[2/5] Preparing Data...")
    if args.data_path:
        # 加载真实数据
        # test_loader = create_dataloader(args.data_path, args.batch_size)
        print("⚠️  Real data loading not implemented, using synthetic data")
        # 使用合成数据
        test_loader = [torch.randn(1, 3, 25, 256, 256) for _ in range(args.num_samples)]
    else:
        # 使用合成数据
        print("Using synthetic data for testing...")
        test_loader = [torch.randn(1, 3, 25, 256, 256) for _ in range(args.num_samples)]
    
    # 评估质量
    print("\n[3/5] Evaluating Quality...")
    results_wfvae_quality = evaluator_wfvae.evaluate_quality(test_loader)
    results_v1_quality = evaluator_v1.evaluate_quality(test_loader)
    
    # 评估效率
    print("\n[4/5] Evaluating Efficiency...")
    results_wfvae_efficiency = evaluator_wfvae.evaluate_efficiency()
    results_v1_efficiency = evaluator_v1.evaluate_efficiency()
    
    # 汇总结果
    results_wfvae = {
        'quality': results_wfvae_quality,
        'efficiency': results_wfvae_efficiency
    }
    
    results_v1 = {
        'quality': results_v1_quality,
        'efficiency': results_v1_efficiency
    }
    
    # 对比分析
    print("\n[5/5] Generating Comparison...")
    comparison = compare_models(results_wfvae, results_v1)
    
    # 保存结果
    results_all = {
        'wfvae': results_wfvae,
        'v1': results_v1,
        'comparison': comparison
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_all, f, indent=2)
    print(f"✅ Results saved to: {output_dir / 'results.json'}")
    
    # 生成报告
    generate_report(results_wfvae, results_v1, comparison, output_dir / 'report.md')
    
    # 打印摘要
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nQuality Winner: {comparison['quality']['winner']}")
    print(f"  PSNR Difference: {comparison['quality']['differences']['psnr_diff']:+.2f} dB")
    print(f"\nEfficiency Winner: {comparison['efficiency']['winner']}")
    print(f"  Params Ratio: {comparison['efficiency']['params_ratio']:.2f}x")
    print(f"  Speed Ratio: {comparison['efficiency']['speed_ratio']:.2f}x")
    print(f"\nOverall Winner: {comparison['tradeoff']['overall_winner']}")
    print("="*80)


if __name__ == '__main__':
    main()

