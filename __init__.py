

# JAX 兼容性补丁
import jax
import sys

try:
    from jax.interpreters import partial_eval as pe
    # 创建 linear_util 的兼容实现
    class LinearUtilCompat:
        @staticmethod
        def wrap(name, fun):
            return fun
    jax.linear_util = LinearUtilCompat()
    sys.modules['jax.linear_util'] = jax.linear_util
    print("✅ JAX linear_util 兼容层已激活")
except ImportError as e:
    print(f"❌ 创建兼容层失败: {e}")
