# Masked Coeffs Compression Usage Guide

## 概述

在 `WVAE_Compressed_TopK_multi_wavelet` 中实现了根据 `masks_5d` 压缩 `masked_coeffs` 的功能，可以删除被 mask 掉的 channel，显著减少内存占用。

## 功能特性

- ✅ 自动检测所有 batch 的 mask 是否相同
- ✅ 统一压缩模式：当所有 batch mask 相同时，统一压缩
- ✅ 分离压缩模式：当不同 batch mask 不同时，分别压缩
- ✅ 提供解压缩函数，可以恢复原始形状
- ✅ 打印详细的压缩信息

## API 说明

### 1. `_compress_masked_coeffs(masked_coeffs, masks_5d)`

**功能：** 根据 mask 压缩系数

**输入：**
- `masked_coeffs`: `(B, Ctot, T, H, W)` - 被 mask 的系数
- `masks_5d`: `(B, Ctot, 1, 1, 1)` - mask 矩阵

**输出：** 字典，包含：

#### 统一模式 (所有 batch mask 相同)
```python
{
    'coeffs': torch.Tensor,              # (B, K, T, H, W) 压缩后的系数
    'keep_indices': torch.Tensor,        # (K,) 保留的 channel 索引
    'original_shape': tuple,             # (B, Ctot, T, H, W)
    'compressed_shape': tuple,           # (B, K, T, H, W)
    'compression_ratio': float,          # K / Ctot
    'mode': 'batch_unified'
}
```

#### 分离模式 (不同 batch mask 不同)
```python
{
    'coeffs_list': List[torch.Tensor],         # 每个 batch 的压缩结果
    'keep_indices_list': List[torch.Tensor],   # 每个 batch 保留的索引
    'original_shape': tuple,                   # (B, Ctot, T, H, W)
    'mode': 'batch_separate',
    'avg_compression_ratio': float             # 平均压缩比
}
```

### 2. `_decompress_masked_coeffs(compressed_result, device=None)`

**功能：** 将压缩的系数恢复为原始形状（被 mask 的位置填充0）

**输入：**
- `compressed_result`: `_compress_masked_coeffs` 的返回结果
- `device`: 目标设备（可选）

**输出：**
- `decompressed_coeffs`: `(B, Ctot, T, H, W)` - 恢复后的系数

### 3. `print_compression_info(compressed_result)`

**功能：** 打印压缩信息

**示例输出：**
```
================================================================================
Masked Coeffs Compression Info
================================================================================
Original shape: (2, 64, 8, 32, 32)
Mode: batch_unified
Compressed shape: (2, 32, 8, 32, 32)
Channels kept: 32/64 (50.00%)
Memory reduction: 50.00%
Element count: 524,288 / 1,048,576
Memory usage: 50.00% of original
================================================================================
```

## 使用示例

### 基本使用

```python
# 在 encode 函数中自动调用
result = vae.encode(x, return_dict=True)
posterior = result.latent_dist
extra_output = result.extra_output

# extra_output 包含：
masked_coeffs, masks_5d, meta, z_teacher, lowfreq_consistency_loss, reshaped_masked_coeffs = extra_output

# reshaped_masked_coeffs 就是压缩后的结果
print(f"Original shape: {masked_coeffs.shape}")
if reshaped_masked_coeffs is not None:
    if reshaped_masked_coeffs['mode'] == 'batch_unified':
        print(f"Compressed shape: {reshaped_masked_coeffs['compressed_shape']}")
        print(f"Compression ratio: {reshaped_masked_coeffs['compression_ratio']:.2%}")
```

### 打印压缩信息

```python
# 在训练过程中查看压缩信息
if step % 100 == 0:
    vae.print_compression_info(reshaped_masked_coeffs)
```

### 手动压缩和解压

```python
# 手动压缩
compressed = vae._compress_masked_coeffs(masked_coeffs, masks_5d)

# 使用压缩后的数据
if compressed['mode'] == 'batch_unified':
    compressed_coeffs = compressed['coeffs']  # (B, K, T, H, W)
    # 在这里使用 compressed_coeffs 进行计算
    # 可以显著节省内存和计算量
    
# 恢复原始形状（如果需要）
decompressed = vae._decompress_masked_coeffs(compressed)
# decompressed 的形状为 (B, Ctot, T, H, W)，被 mask 的位置为 0
```

### 在损失计算中使用

```python
# 如果只想对保留的 channel 计算损失
if reshaped_masked_coeffs['mode'] == 'batch_unified':
    compressed_coeffs = reshaped_masked_coeffs['coeffs']
    keep_indices = reshaped_masked_coeffs['keep_indices']
    
    # 获取对应的 teacher 数据
    z_teacher_compressed = z_teacher[:, keep_indices, :, :, :]
    
    # 计算损失（只在保留的 channel 上）
    loss = F.mse_loss(compressed_coeffs, z_teacher_compressed)
```

## 性能优势

### 内存节省

假设：
- Batch size: 4
- 原始 channel: 64
- 保留 channel: 32 (50%)
- 时间维度: 8
- 空间维度: 32x32

**原始内存：**
```
4 × 64 × 8 × 32 × 32 = 2,097,152 elements
≈ 8 MB (float32)
```

**压缩后内存：**
```
4 × 32 × 8 × 32 × 32 = 1,048,576 elements
≈ 4 MB (float32)
```

**节省：** 50% 内存

### 计算加速

对于后续的计算操作（如卷积、注意力等），压缩后的 tensor 可以：
- 减少 50% 的计算量
- 提升计算速度
- 降低显存压力

## 注意事项

### 1. Mask 模式

当前实现支持两种模式：
- **固定 mask 模式**：所有 batch 使用相同的 mask（最常用）
- **动态 mask 模式**：每个 batch 可以有不同的 mask

大多数情况下使用固定 mask 模式，压缩效率更高。

### 2. 梯度处理

如果需要通过压缩的 tensor 反向传播：
```python
# 压缩操作保留梯度
compressed = vae._compress_masked_coeffs(masked_coeffs, masks_5d)
compressed_coeffs = compressed['coeffs']  # 保留梯度

# 在压缩的 coeffs 上计算损失
loss = some_loss_function(compressed_coeffs)
loss.backward()  # 梯度会正确传播回 masked_coeffs 的对应位置
```

### 3. 保存和加载

如果需要保存压缩后的结果：
```python
# 保存
torch.save({
    'compressed_coeffs': compressed['coeffs'],
    'keep_indices': compressed['keep_indices'],
    'original_shape': compressed['original_shape'],
}, 'compressed.pt')

# 加载
data = torch.load('compressed.pt')
# 恢复
decompressed = torch.zeros(data['original_shape'])
decompressed[:, data['keep_indices'], :, :, :] = data['compressed_coeffs']
```

## 示例输出

```python
# 示例 1: 统一模式
vae.print_compression_info(reshaped_masked_coeffs)
```

输出：
```
================================================================================
Masked Coeffs Compression Info
================================================================================
Original shape: (4, 64, 8, 32, 32)
Mode: batch_unified
Compressed shape: (4, 32, 8, 32, 32)
Channels kept: 32/64 (50.00%)
Memory reduction: 50.00%
Element count: 1,048,576 / 2,097,152
Memory usage: 50.00% of original
================================================================================
```

```python
# 示例 2: 分离模式（不同 batch 不同 mask）
vae.print_compression_info(reshaped_masked_coeffs)
```

输出：
```
================================================================================
Masked Coeffs Compression Info
================================================================================
Original shape: (4, 64, 8, 32, 32)
Mode: batch_separate
Average compression ratio: 48.44%
  Batch 0: 30/64 channels kept (46.88%)
  Batch 1: 32/64 channels kept (50.00%)
  Batch 2: 31/64 channels kept (48.44%)
  Batch 3: 31/64 channels kept (48.44%)
================================================================================
```

## 扩展用途

### 1. 可视化 mask 分布

```python
if reshaped_masked_coeffs['mode'] == 'batch_unified':
    keep_indices = reshaped_masked_coeffs['keep_indices']
    
    # 创建可视化
    import matplotlib.pyplot as plt
    mask_vis = torch.zeros(64)
    mask_vis[keep_indices] = 1
    
    plt.figure(figsize=(12, 2))
    plt.imshow(mask_vis.reshape(1, -1), cmap='RdYlGn', aspect='auto')
    plt.colorbar()
    plt.title('Channel Mask Visualization')
    plt.xlabel('Channel Index')
    plt.show()
```

### 2. 分析压缩比随训练的变化

```python
compression_ratios = []

for epoch in range(num_epochs):
    for batch in dataloader:
        result = vae.encode(batch)
        compressed = result.extra_output[-1]
        if compressed is not None:
            ratio = compressed['compression_ratio']
            compression_ratios.append(ratio)
    
    # 每个 epoch 后分析
    avg_ratio = sum(compression_ratios) / len(compression_ratios)
    print(f"Epoch {epoch}: Average compression ratio = {avg_ratio:.2%}")
```

## 总结

这个压缩功能可以：
- 🔥 显著减少内存占用（通常 30-70%）
- ⚡ 加速后续计算
- 📊 提供详细的压缩统计信息
- 🔄 支持无损恢复（填充0）
- 🎯 适用于稀疏化训练场景

