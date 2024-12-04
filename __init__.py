from .misc_nodes import EnhancedLoadDiffusionModel
from .velocator_nodes import (
    VelocatorLoadAndQuantizeDiffusionModel,
    VelocatorLoadAndQuantizeClip,
    VelocatorQuantizeModel,
    VelocatorCompileModel,
)


def patch_cast_to():

    def cast_to(weight, dtype=None, device=None, non_blocking=False, copy=False):
        if device is None or weight.device == device:
            if not copy:
                if dtype is None or weight.dtype == dtype:
                    return weight
            return weight.to(dtype=dtype, copy=copy)

        # torch.empty_like does not work with tensor subclasses well
        # r = torch.empty_like(weight, dtype=dtype, device=device)
        # r.copy_(weight, non_blocking=non_blocking)
        r = weight.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
        return r

    from comfy import model_management

    model_management.cast_to = cast_to


patch_cast_to()

NODE_CLASS_MAPPINGS = {
    "EnhancedLoadDiffusionModel": EnhancedLoadDiffusionModel,
    "VelocatorLoadAndQuantizeDiffusionModel": VelocatorLoadAndQuantizeDiffusionModel,
    "VelocatorLoadAndQuantizeClip": VelocatorLoadAndQuantizeClip,
    "VelocatorQuantizeModel": VelocatorQuantizeModel,
    "VelocatorCompileModel": VelocatorCompileModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnhancedLoadDiffusionModel": "Load Diffusion Model+",
    "VelocatorLoadAndQuantizeDiffusionModel": "🚀Load & Quantize Diffusion Model",
    "VelocatorLoadAndQuantizeClip": "🚀Load & Quantize CLIP",
    "VelocatorQuantizeModel": "🚀Quantize Model",
    "VelocatorCompileModel": "🚀Compile Model",
}
