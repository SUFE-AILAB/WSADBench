import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 导入基线方法
try:
    from .Sultani import Sultani
except ImportError as e:
    print(f"Warning: Failed to import Sultani: {e}")
    Sultani = None

try:
    from .CRGAN import CRGAN
except ImportError as e:
    print(f"Warning: Failed to import CRGAN: {e}")
    CRGAN = None

__all__ = ['Sultani', 'CRGAN']