import contextlib
import unittest

import torch


# wildcard trick is taken from pythongossss's
class AnyType(str):

    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")


def get_weight_dtype_inputs():
    return {
        "weight_dtype": (
            [
                "default",
                "float32",
                "float64",
                "bfloat16",
                "float16",
                "fp8_e4m3fn",
                "fp8_e4m3fn_fast",
                "fp8_e5m2",
            ],
        ),
    }


def parse_weight_dtype(model_options, weight_dtype):
    dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "fp8_e4m3fn": torch.float8_e4m3fn,
        "fp8_e4m3fn_fast": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
    }.get(weight_dtype, None)
    if dtype is not None:
        model_options["dtype"] = dtype
    if weight_dtype == "fp8_e4m3fn_fast":
        model_options["fp8_optimizations"] = True
    return model_options


@contextlib.contextmanager
def disable_load_models_gpu():
    def foo(*args, **kwargs):
        pass

    from comfy import model_management

    with unittest.mock.patch.object(model_management, "load_models_gpu", foo):
        yield
