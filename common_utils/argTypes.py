import argparse


def str2bool(v):
    """将字符串转换为布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("需要布尔值.")


def none_or_str(value):
    """处理None字符串"""
    if value.lower() == "none":
        return None
    return value

def int_or_float(value):
    """如果输入的是整数，则返回整数，否则返回浮点数"""
    if isinstance(value, int) or isinstance(value, float):
        return value
    try:
        if "." in value:
            return float(value)
        else:
            return int(value)
    except:
        return value