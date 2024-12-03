import importlib
import json
import torch
import comfy.model_management

HAS_VELOCATOR = importlib.util.find_spec("xelerate") is not None


class VelocatorQuantizeModel:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "object_to_patch": ("STRING", {
                    "default": "diffusion_model",
                }),
                "quant_type": ([
                    "int8_dynamic",
                    "e4m3_e4m3_dynamic",
                    "e4m3_e4m3_dynamic_per_tensor",
                    "int8_weightonly",
                    "e4m3_weightonly",
                    "e4m3_e4m3_weightonly",
                    "e4m3_e4m3_weightonly_per_tensor",
                    "nf4_weightonly",
                    "int4_weightonly",
                ],),
                "filter_fn": ("STRING", {
                    "default": "fnmatch_matches_fqn",
                }),
                "filter_fn_kwargs": ("STRING", {
                    "multiline": True,
                    "default": '{"pattern": ["*"]}',
                }),
                "kwargs": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                }),
            }
        }

    RETURN_TYPES = ("MODEL", )
    FUNCTION = "patch"

    CATEGORY = "wavespeed"

    def __init__(self):
        self._patched = False

    def patch(self, model, object_to_patch, quant_type, filter_fn, filter_fn_kwargs, kwargs):
        if not self._patched:
            assert HAS_VELOCATOR, "velocator is not installed"

            from xelerate.ao.quant import quantize

            comfy.model_management.unload_all_models()
            comfy.model_management.load_models_gpu([model], force_patch_weights=True, force_full_load=True)

            filter_fn_kwargs = json.loads(filter_fn_kwargs)
            kwargs = json.loads(kwargs)

            model = model.clone()
            model.add_object_patch(
                object_to_patch,
                quantize(
                    model.get_model_object(object_to_patch),
                    quant_type=quant_type,
                    filter_fn=filter_fn,
                    filter_fn_kwargs=filter_fn_kwargs,
                    **kwargs,
                ))

            self._patched = True

        return (model, )


class VelocatorCompileModel:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "object_to_patch": ("STRING", {
                    "default": "diffusion_model",
                }),
                "memory_format": (["channels_last", "contiguous_format", "preserve_format"],),
                "fullgraph": ("BOOLEAN", {
                    "default": False,
                }),
                "dynamic": (["None", "True", "False"],),
                "mode": ("STRING", {
                    "multiline": True,
                    "default": "cache-all:max-autotune:low-precision",
                }),
                "options": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                }),
                "disable": ("BOOLEAN", {
                    "default": False,
                }),
                "backend": ("STRING", {
                    "default": "velocator",
                })
            }
        }

    RETURN_TYPES = ("MODEL", )
    FUNCTION = "patch"

    CATEGORY = "wavespeed"

    def __init__(self):
        self._compiled = False

    def patch(self, model, object_to_patch, memory_format, fullgraph, dynamic, mode, options, disable, backend):
        if not self._compiled:
            assert HAS_VELOCATOR, "velocator is not installed"

            from xelerate.compilers.xelerate_compiler import xelerate_compile
            from xelerate.utils.memory_format import apply_memory_format

            memory_format = getattr(torch, memory_format)

            dynamic = eval(dynamic)
            options = json.loads(options)
            if backend == "velocator":
                backend = "xelerate"

            model = model.clone()
            model.add_object_patch(
                object_to_patch,
                xelerate_compile(
                    apply_memory_format(model.get_model_object(object_to_patch), memory_format=memory_format),
                    fullgraph=fullgraph,
                    dynamic=dynamic,
                    mode=mode,
                    options=options,
                    disable=disable,
                    backend=backend,
                ))

            self._compiled = True

        return (model, )
