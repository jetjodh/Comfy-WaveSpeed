import contextlib
import unittest
import torch

from . import utils
from . import first_block_cache


class ApplyFBCacheOnModel:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (utils.any_typ, ),
                "object_to_patch": (
                    "STRING",
                    {
                        "default": "diffusion_model",
                    },
                ),
                "residual_diff_threshold": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.005,
                    },
                ),
            }
        }

    RETURN_TYPES = (utils.any_typ, )
    FUNCTION = "patch"

    CATEGORY = "wavespeed"

    def patch(
        self,
        model,
        object_to_patch,
        residual_diff_threshold,
    ):
        prev_timestep = None

        model = model.clone()
        diffusion_model = model.get_model_object(object_to_patch)
        cached_transformer_blocks = torch.nn.ModuleList([
            first_block_cache.CachedTransformerBlocks(
                diffusion_model.transformer_blocks if hasattr(
                    diffusion_model, "transformer_blocks") else
                diffusion_model.double_blocks,
                diffusion_model.single_blocks if hasattr(
                    diffusion_model, "single_blocks") else None,
                residual_diff_threshold=residual_diff_threshold,
            )
        ])
        dummy_single_transformer_blocks = torch.nn.ModuleList()

        def model_unet_function_wrapper(model_function, kwargs):
            nonlocal prev_timestep

            input = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]
            t = timestep[0].item()

            if prev_timestep is None or t >= prev_timestep:
                prev_timestep = t
                first_block_cache.set_current_cache_context(
                    first_block_cache.create_cache_context())

            with unittest.mock.patch.object(
                    diffusion_model,
                    "transformer_blocks"
                    if hasattr(diffusion_model, "transformer_blocks") else
                    "double_blocks",
                    cached_transformer_blocks,
            ), unittest.mock.patch.object(
                    diffusion_model,
                    "single_blocks",
                    dummy_single_transformer_blocks,
            ) if hasattr(diffusion_model,
                         "single_blocks") else contextlib.nullcontext():
                return model_function(input, timestep, **c)

        model.set_model_unet_function_wrapper(model_unet_function_wrapper)
        return (model, )
