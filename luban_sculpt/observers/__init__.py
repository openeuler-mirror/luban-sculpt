"""
Luban custom llm-compressor observers and schemes.
"""
 
from .luban_ema_absmax import LubanEMAAbsMaxObserver
from .luban_percentile import LubanPercentileObserver

#必须先导入 Observer，才能导入 Scheme。
__all__ = [
    "LubanEMAAbsMaxObserver",
    "LubanPercentileObserver",

]