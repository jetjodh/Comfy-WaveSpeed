import torch
import folder_paths
import comfy.sd


class EnhancedDiffusionModelLoader:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
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
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "wavespeed"

    def load_unet(self, unet_name, weight_dtype):
        model_options = {}
        if weight_dtype == "float32":
            model_options["dtype"] = torch.float32
        elif weight_dtype == "float64":
            model_options["dtype"] = torch.float64
        elif weight_dtype == "bfloat16":
            model_options["dtype"] = torch.bfloat16
        elif weight_dtype == "float16":
            model_options["dtype"] = torch.float16
        elif weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return (model,)
