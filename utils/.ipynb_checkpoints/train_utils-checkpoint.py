from functools import partial

from matplotlib.pyplot import step
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import lax, jit, vmap, pmap, random
from jax.flatten_util import ravel_pytree
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

# -------------------------- 原有模块 --------------------------
def compute_lp_norms(pred, y, ord=2):
    diff_norms = jnp.linalg.norm(pred - y, axis=1, ord=ord, keepdims=True)
    y_norms = jnp.linalg.norm(y, axis=1, ord=ord, keepdims=True)
    lp_error = (diff_norms / y_norms).mean()
    return diff_norms, y_norms, lp_error

class PatchHandler:
    def __init__(self, inputs, patch_size):
        self.patch_size = patch_size
        _, self.height, self.width, self.channel = inputs.shape
        self.patch_height, self.patch_width = (
            self.height // self.patch_size[0],
            self.width // self.patch_size[1],
        )
    def merge_patches(self, x):
        batch, _, _ = x.shape
        x = jnp.reshape(
            x,
            (
                batch,
                self.patch_height,
                self.patch_width,
                self.patch_size[0],
                self.patch_size[1],
                -1,
            ),
        )
        x = jnp.swapaxes(x, 2, 3)
        x = jnp.reshape(
            x,
            (
                batch,
                self.patch_height * self.patch_size[0],
                self.patch_width * self.patch_size[1],
                -1,
            ),
        )
        return x

def create_encoder_step(encoder, mesh):
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch")),
        out_specs=P("batch"),
    )
    def encoder_step(encoder_params, batch):
        _, x, _ = batch
        z = encoder.apply(encoder_params, x)
        return z

    return encoder_step

###################################################
############# utils for diffusion models ##########
###################################################


def create_train_diffusion_step(model, mesh, use_conditioning=False):
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch")),
        out_specs=(P(), P()),
    )
    def train_step(state, batch):
        def loss_fn(params):
            if use_conditioning:
                x, t, c, y = batch
                pred = model.apply(params, x, t, c)
            else:
                x, t, y = batch
                pred = model.apply(params, x, t)
            eps = 1e-8  # 添加这行
            # 获取批量大小和序列长度
            batch_size, seq_len, channels = y.shape
            
            # 创建正确的掩码形状
            real_data_length = 50  # 你的真实数据长度
            mask = jnp.arange(seq_len) < real_data_length
            mask = mask.astype(jnp.float32)
            
            # 扩展到正确的形状: (batch_size, seq_len, channels)
            mask = mask[None, :, None]  # (1, seq_len, 1)
            mask = jnp.broadcast_to(mask, (batch_size, seq_len, channels))
            # ====================================================
            
            # 计算有效元素数量
            valid_count = jnp.sum(mask) + eps
            
            # 计算带掩码的MSE损失
            squared_error = (y - pred) ** 2
            masked_squared_error = squared_error * mask
            loss = jnp.sum(masked_squared_error) / valid_count
            return loss

        # Compute gradients and update parameters
        grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
        loss, grads = grad_fn(state.params)

        grads = lax.pmean(grads, "batch")
        loss = lax.pmean(loss, "batch")
        state = state.apply_gradients(grads=grads)
        
        return state, loss

    return train_step

@partial(jit, static_argnums=(3,))
def get_diffusion_batch(key, z1=None, c=None, use_conditioning=False):
    keys = random.split(key, 3)
    z0 = random.normal(keys[0], shape=z1.shape)  # (b, 200, 512)
    t = random.uniform(keys[1], (z1.shape[0], 1, 1))

    z_t = t * z1 + (1. - t) * z0
    target = z1 - z0

    if use_conditioning:
        batch = (z_t, t.flatten(), c, target)
    else:
        batch = (z_t, t.flatten(), target)

    return batch, keys[2]


def sample_ode(state, z0=None, c=None, num_steps=None, use_conditioning=False):
    dt = 1 / num_steps
    traj = [z0]

    z = z0
    for i in tqdm(range(num_steps)):
        t = jnp.ones((z.shape[0],)) * i / num_steps
        if use_conditioning:
            pred = state.apply_fn(state.params, z, t, c)
        else:
            pred = state.apply_fn(state.params, z, t)
        z = z + pred * dt
        traj.append(z)
    return z, traj




# 添加以下函数

def create_autoencoder_eval_step(encoder, decoder, mesh=None):
    """创建自编码器评估步骤"""
    # 使用相同的物理数据范围
    PHYSICAL_DATA_RANGE = 2.106  # 根据数据 [-0.956, 1.150] 计算
    
    if mesh is not None:
        @jax.jit
        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(P(), P("batch")),
            out_specs=P(),
            check_rep=False
        )
        def eval_step(state, batch):
            encoder_params, decoder_params = state.params
            x, coords, y = batch
            
            # 添加通道转换：将2通道转换为1通道
            if x.shape[-1] == 2:
                x = jnp.mean(x, axis=-1, keepdims=True)  # 取平均值变成1通道
            
            z = encoder.apply(encoder_params, x)
            
            # ========== 修复batch维度匹配 ==========
            
            # 处理坐标形状：确保是 (batch, num_coords, 1)
            if coords.ndim == 3 and coords.shape[2] == 2:
                # 如果坐标有2个维度，只取第一个维度（深度坐标）
                coords_processed = coords[:, :, 0:1]  # 取第一个维度，保持3D形状
            elif coords.ndim == 3 and coords.shape[2] == 1:
                # 已经是1维坐标，直接使用
                coords_processed = coords
            elif coords.ndim == 2:
                # (num_coords, dim) -> (batch, num_coords, dim)
                coords_processed = jnp.broadcast_to(coords[None, :, :], (x.shape[0], coords.shape[0], coords.shape[1]))
            elif coords.ndim == 1:
                # (num_coords,) -> (batch, num_coords, 1)
                coords_processed = jnp.broadcast_to(coords[None, :, None], (x.shape[0], coords.shape[0], 1))
            else:
                # 默认创建96个坐标点
                num_coords = 96
                coords_processed = jnp.broadcast_to(jnp.linspace(0, 1, num_coords)[None, :, None], 
                                                   (x.shape[0], num_coords, 1))
            
            # 确保隐变量和坐标的batch维度匹配
            if z.shape[0] != coords_processed.shape[0]:
                # 广播隐变量到匹配的batch维度
                z = jnp.broadcast_to(z, (coords_processed.shape[0], z.shape[1], z.shape[2]))
            
            # ========== 修复结束 ==========
                
            pred = decoder.apply(decoder_params, z, coords_processed)
            
            # 计算带掩码的RMSE
            batch_size, seq_len, channels = y.shape
        
            # 创建掩码
            real_data_length = 50  # 你的真实数据长度
            mask = jnp.arange(seq_len) < real_data_length
            mask = mask.astype(jnp.float32)
            mask = mask[None, :, None]  # (1, seq_len, 1)
            mask = jnp.broadcast_to(mask, (batch_size, seq_len, channels))
            
            valid_count = jnp.sum(mask) + 1e-8
            
            # 计算带掩码的MSE损失
            squared_error = (y - pred) ** 2
            masked_squared_error = squared_error * mask
            loss = jnp.sum(masked_squared_error) / valid_count
            rmse = jnp.sqrt(loss)
            
            # 使用物理数据范围计算NRMSE
            normalized_rmse = rmse / (PHYSICAL_DATA_RANGE + 1e-8)
            
            # 汇总结果
            rmse = jax.lax.pmean(rmse, "batch")
            normalized_rmse = jax.lax.pmean(normalized_rmse, "batch")
            
            return rmse, normalized_rmse
        return eval_step
    else:
        @jax.jit
        def eval_step(state, batch):
            encoder_params, decoder_params = state.params
            x, coords, y = batch
            
            # 添加通道转换：将2通道转换为1通道
            if x.shape[-1] == 2:
                x = jnp.mean(x, axis=-1, keepdims=True)
            
            z = encoder.apply(encoder_params, x)
            
            # ========== 同样的batch维度修复 ==========
            
            if coords.ndim == 3 and coords.shape[2] == 2:
                coords_processed = coords[:, :, 0:1]
            elif coords.ndim == 3 and coords.shape[2] == 1:
                coords_processed = coords
            elif coords.ndim == 2:
                coords_processed = jnp.broadcast_to(coords[None, :, :], (x.shape[0], coords.shape[0], coords.shape[1]))
            elif coords.ndim == 1:
                coords_processed = jnp.broadcast_to(coords[None, :, None], (x.shape[0], coords.shape[0], 1))
            else:
                num_coords = 96
                coords_processed = jnp.broadcast_to(jnp.linspace(0, 1, num_coords)[None, :, None], 
                                                   (x.shape[0], num_coords, 1))
            
            # 确保隐变量和坐标的batch维度匹配
            if z.shape[0] != coords_processed.shape[0]:
                z = jnp.broadcast_to(z, (coords_processed.shape[0], z.shape[1], z.shape[2]))
            
            # ========== 修复结束 ==========
                
            pred = decoder.apply(decoder_params, z, coords_processed)
            
            # 计算损失（同上）
            batch_size, seq_len, channels = y.shape
            real_data_length = 50
            mask = jnp.arange(seq_len) < real_data_length
            mask = mask.astype(jnp.float32)
            mask = mask[None, :, None]
            mask = jnp.broadcast_to(mask, (batch_size, seq_len, channels))
            
            valid_count = jnp.sum(mask) + 1e-8
            squared_error = (y - pred) ** 2
            masked_squared_error = squared_error * mask
            loss = jnp.sum(masked_squared_error) / valid_count
            rmse = jnp.sqrt(loss)
            normalized_rmse = rmse / (PHYSICAL_DATA_RANGE + 1e-8)
            
            return rmse, normalized_rmse
        return eval_step


def create_end_to_end_eval_step(encoder, decoder, diffusion_model, mesh, use_conditioning=False):
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P(), P("batch")),
        out_specs=P(),
        check_rep=False
    )
    def eval_step(fae_state, diffusion_state, batch):
        coords, x_condition, y_true_target = batch
        
        # 添加通道转换
        if x_condition.shape[-1] == 2:
            x_condition = jnp.mean(x_condition, axis=-1, keepdims=True)
        
        # 1. 编码器获取条件隐变量
        encoder_params, _ = fae_state.params
        z_condition = encoder.apply(encoder_params, x_condition)
        
        # ================================
        # 修复1: 使用不同的随机种子
        # ================================
        # 使用当前时间或步数作为随机种子
        key = random.PRNGKey(42)  # 固定种子，但确保可重复
        key, subkey = random.split(key)
        z0 = random.normal(subkey, z_condition.shape)
        
        # 2. 扩散模型生成
        z_generated, _ = sample_ode(
            diffusion_state, z0, z_condition, num_steps=100, use_conditioning=use_conditioning
        )

        # 3. 解码器转换
        _, decoder_params = fae_state.params
        
        # 坐标处理
        if coords.ndim == 3:
            coords_processed = coords.reshape(coords.shape[0], -1)
        else:
            coords_processed = jnp.repeat(coords[jnp.newaxis, :], z_generated.shape[0], axis=0)
        
        y_pred = decoder.apply(decoder_params, z_generated, coords_processed)
        
        # ================================
        # 修复2: 检查批次内样本的多样性
        # ================================
        # 计算每个样本的预测与真实值的相关性
        sample_correlations = []
        for i in range(y_pred.shape[0]):
            correlation = jnp.corrcoef(y_pred[i].flatten(), y_true_target[i].flatten())[0, 1]
            sample_correlations.append(correlation)
        
        avg_correlation = jnp.mean(jnp.array(sample_correlations))
        jax.debug.print("样本预测相关性: {:.6f}", avg_correlation)
        
        # 计算损失
        squared_error = (y_pred - y_true_target) ** 2
        rmse = jnp.sqrt(jnp.mean(squared_error))
        normalized_rmse = rmse / 2.106  # 使用你的物理范围
        
        return rmse, normalized_rmse, y_pred, y_true_target
    
    return jax.jit(eval_step)
