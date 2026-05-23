"""
CellViT TorchScript Conversion Implementation
Based on best practices for complex medical imaging models
"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.abspath(current_dir))
sys.path.append(current_dir)
sys.path.append(project_root)

import torch.nn as nn
import torch.jit
import traceback
from pathlib import Path
from typing import Dict, Tuple, Literal, Union, Optional
from tqdm import tqdm
import json
import time

from cellvit.models.cell_segmentation.cellvit import CellViT
from cellvit.models.cell_segmentation.cellvit_256 import CellViT256
from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
from cellvit.models.cell_segmentation.cellvit_uni import CellViTUNI
from cellvit.models.cell_segmentation.cellvit_virchow import CellViTVirchow
from cellvit.models.classifier.linear_classifier import LinearClassifier



# FACTORY_KINDS = {
#     "aten::zeros", "aten::ones", "aten::full", "aten::empty",
#     "aten::eye", "aten::arange", "aten::linspace", "aten::rand",
#     "aten::randn", "aten::tensor"  # 視圖中實際出現的為準
# }
#
# def patch_factories_move_to_input_device(tsm: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
#     g = tsm.graph
#     ref = g.inputs()  # 假設 forward 的第 1 個參數是你的輸入 tensor
#
#     # 方便建立常數
#     def c_bool(val: bool):
#         n = g.create("prim::Constant")
#         n.f_("value", float(val)) if isinstance(val, float) else n.i_("value", int(val))
#         g.insert_node(n)
#         return n.output()
#
#     def c_none():
#         n = g.create("prim::Constant")
#         n.setattr("value", None)
#         g.insert_node(n)
#         return n.output()
#
#     # PyTorch 內部有多種 aten::to 的 overload，這裡選「以 tensor 作為參考」的那個：
#     # schema: aten::to(Tensor self, Tensor other, bool non_blocking, bool copy, MemoryFormat? memory_format) -> Tensor
#
#     for n in list(g.nodes()):
#         if n.kind() in FACTORY_KINDS:
#             with g.inserting_after(n):
#                 to_node = g.create(
#                     "aten::to",
#                     [n.output(), ref, c_bool(False), c_bool(False), c_none()]
#                 )
#                 g.insert_node(to_node)
#                 n.output().replaceAllUsesWith(to_node.output())
#
#     tsm.compile()
#     return tsm




from collections import deque




FACTORY_KINDS = {
    "aten::arange","aten::zeros","aten::ones","aten::full","aten::empty",
    "aten::eye","aten::linspace","aten::rand","aten::randn","aten::tensor",
}
SINGLE_TENSOR_PRODUCERS = set(FACTORY_KINDS) | {
    "prim::Constant","prim::GetAttr","prim::NumToTensor",
}

# ----------------------------- type helpers -----------------------------

def is_tensor_type(t: torch._C.Type) -> bool:
    return isinstance(t, torch._C.TensorType) or t.isSubtypeOf(torch._C.TensorType.get())

def is_optional_tensor_type(t: torch._C.Type) -> bool:
    try:
        from torch._C import OptionalType
        return isinstance(t, OptionalType) and is_tensor_type(t.getElementType())
    except Exception:
        return ("Optional" in str(t)) and ("Tensor" in str(t))

def is_tensor_or_optional_tensor(v: torch._C.Value) -> bool:
    t = v.type()
    return is_tensor_type(t) or is_optional_tensor_type(t)

def first_tensor_input(g: torch._C.Graph):
    for v in g.inputs():
        if is_tensor_type(v.type()):
            return v
    return None

# ---------------------- anchor-based insertion utils --------------------

def _set_const_value(n: torch._C.Node, value):
    # n is prim::Constant
    try:
        if value is None:
            # NoneType constant: set output type to NoneType, no value attribute
            n.output().setType(torch._C.NoneType.get())
        elif isinstance(value, bool):
            n.i_("value", int(value))
            n.output().setType(torch._C.BoolType.get())
        elif isinstance(value, int):
            n.i_("value", value)
        elif isinstance(value, float):
            n.f_("value", float(value))
        elif isinstance(value, str):
            n.s_("value", value)
    except Exception:
        pass

def insert_const_before(anchor: torch._C.Node, value, g: torch._C.Graph):
    n = g.create("prim::Constant", [], 1)
    n.insertBefore(anchor)
    _set_const_value(n, value)
    return n.output()

def insert_const_after(anchor: torch._C.Node, value, g: torch._C.Graph):
    n = g.create("prim::Constant", [], 1)
    n.insertAfter(anchor)
    _set_const_value(n, value)
    return n.output()

def make_to_after_producer(producer: torch._C.Node, ref: torch._C.Value, g: torch._C.Graph):
    """
    在 producer 之後串一個 to(out, ref, False, False, None)
    並把 producer 的輸出用途全部改到 to 的輸出。
    """
    outs = list(producer.outputs())
    if len(outs) != 1 or not is_tensor_type(outs[0].type()):
        return None
    out_v = outs[0]
    # 連鎖插入：確保順序 = 常數們 -> to
    v_nb = insert_const_after(producer, False, g)
    v_cp = insert_const_after(v_nb.node(), False, g)
    v_mf = insert_const_after(v_cp.node(), None, g)

    to_node = g.create("aten::to", [out_v, ref, v_nb, v_cp, v_mf], 1)
    to_node.insertAfter(v_mf.node())

    # 避免自循環：先整體替換，再把第 0 輸入還原為原 out_v
    out_v.replaceAllUsesWith(to_node.output())
    to_node.replaceInput(0, out_v)
    return to_node

def wrap_value_before_consumer(val: torch._C.Value, consumer: torch._C.Node, ref: torch._C.Value, g: torch._C.Graph) -> torch._C.Value:
    """
    對 consumer 的某個 Tensor/Optional[Tensor] 輸入，在 consumer 之前包一層 to(...)
    """
    if not is_tensor_or_optional_tensor(val):
        return val
    # 在 consumer 前面依序放置常數，再放 to
    v_mf = insert_const_before(consumer, None, g)
    v_cp = insert_const_before(v_mf.node(), False, g)
    v_nb = insert_const_before(v_cp.node(), False, g)

    to_node = g.create("aten::to", [val, ref, v_nb, v_cp, v_mf], 1)
    to_node.insertBefore(consumer)
    return to_node.output()

# ------------------------------- patchers -------------------------------

def patch_graph(g: torch._C.Graph, tag: str = "", verbose=True):
    """
    - factories / Constant / GetAttr / NumToTensor（單一 Tensor 輸出）→ producer 後插 to(...)
    - ListConstruct：逐一檢查**輸入**是否為 (Optional)Tensor → consumer 前插 to(...)
    """
    ref = first_tensor_input(g)
    if ref is None:
        if verbose:
            print(f"[{tag}] skip (no tensor input)")
        return (0, 0, 0, 0, 0)

    f_cnt = c_cnt = g_cnt = l_cnt = 0
    blocks = 0

    def walk(block: torch._C.Block):
        nonlocal f_cnt, c_cnt, g_cnt, l_cnt, blocks
        blocks += 1

        for n in list(block.nodes()):
            # 遞迴 child blocks
            for b in n.blocks():
                walk(b)

            k = n.kind()

            if k == "prim::ListConstruct":
                changed = False
                for i, iv in enumerate(list(n.inputs())):
                    nv = wrap_value_before_consumer(iv, n, ref, g)
                    if nv is not iv:
                        n.replaceInput(i, nv)
                        changed = True
                if changed:
                    l_cnt += 1
                continue

            if k in SINGLE_TENSOR_PRODUCERS:
                outs = list(n.outputs())
                if len(outs) == 1 and is_tensor_type(outs[0].type()):
                    make_to_after_producer(n, ref, g)
                    if k in FACTORY_KINDS:
                        f_cnt += 1
                    elif k == "prim::Constant":
                        c_cnt += 1
                    elif k == "prim::GetAttr":
                        g_cnt += 1
                    else:  # prim::NumToTensor
                        f_cnt += 1
                continue

    walk(g.block())
    return (f_cnt, c_cnt, g_cnt, l_cnt, blocks)

def iter_method_names(cmod) -> list:
    for attr in ("_method_names","_get_method_names"):
        if hasattr(cmod, attr):
            try:
                return list(getattr(cmod, attr)())
            except Exception:
                pass
    if hasattr(cmod, "methods"):
        try:
            return [m.name for m in cmod.methods()]
        except Exception:
            pass
    return ["forward"]

def get_method_graph(cmod, name: str):
    if hasattr(cmod, "_get_method"):
        return cmod._get_method(name).graph
    if hasattr(cmod, "methods"):
        for m in cmod.methods():
            if getattr(m, "name", None) == name:
                return m.graph
    return getattr(cmod, name).graph

def safe_mod_name(cmod) -> str:
    for attr in ("qualified_name","original_name"):
        v = getattr(cmod, attr, None)
        if isinstance(v, str) and v:
            return v
    try:
        return str(cmod)
    except Exception:
        return "<module>"

def patch_all_methods(tsm: torch.jit.ScriptModule, verbose=True):
    total_f = total_c = total_g = total_l = total_b = 0
    q = deque([tsm._c])

    while q:
        cmod = q.popleft()
        mname = safe_mod_name(cmod)
        for name in iter_method_names(cmod):
            try:
                g = get_method_graph(cmod, name)
            except Exception as e:
                if verbose:
                    print(f"[{mname}::{name}] skip (cannot get graph): {e}")
                continue
            f, c, gattr, lfix, b = patch_graph(g, tag=f"{mname}::{name}", verbose=verbose)
            total_f += f; total_c += c; total_g += gattr; total_l += lfix; total_b += b
            if verbose and (f or c or gattr or lfix):
                print(f"[{mname}::{name}] factories={f}, consts={c}, getattrs={gattr}, listfix={lfix}, blocks={b}")
        try:
            for child in cmod.children():
                q.append(child)
        except Exception:
            pass

    tsm.compile()
    if verbose:
        print(f"[summary] factories={total_f}, consts={total_c}, getattrs={total_g}, listfix={total_l}, blocks={total_b}")
    return tsm




class CellViTWithLinearClassifier(nn.Module):
    """Fused CellViT + LinearClassifier wrapper for TorchScript export.

    The LinearClassifier is applied pixel-wise to the encoder's token map (z4),
    replacing nuclei_type_map with predictions aligned to the custom label set.

    Output nuclei_type_map shape: (B, num_classes, H_tokens, W_tokens)
    e.g. for CellViTSAM-H with 1024x1024 input: (B, num_classes, 64, 64)
    """

    def __init__(self, cellvit: nn.Module, classifier: nn.Module):
        super().__init__()
        self.cellvit = cellvit
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> dict:
        # retrieve_tokens=True to obtain the encoder feature map
        out = self.cellvit(x, retrieve_tokens=True)
        tokens = out["tokens"]  # (B, embed_dim, H_t, W_t)
        B, D, H, W = tokens.shape
        # apply classifier to each spatial token position
        cls_logits = self.classifier(
            tokens.permute(0, 2, 3, 1).reshape(B * H * W, D)
        )  # (B*H*W, num_classes)
        return {
            "tissue_types": out["tissue_types"],
            "nuclei_binary_map": out["nuclei_binary_map"],
            "hv_map": out["hv_map"],
            "nuclei_type_map": cls_logits.reshape(B, H, W, -1).permute(0, 3, 1, 2),
        }


def unflatten_dict(d: dict, sep: str = ".") -> dict:
    """Unflatten a flattened dictionary (created a nested dictionary)

    Args:
        d (dict): Dict to be nested
        sep (str, optional): Seperator of flattened keys. Defaults to '.'.

    Returns:
        dict: Nested dict
    """
    output_dict = {}
    for key, value in d.items():
        keys = key.split(sep)
        d = output_dict
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    return output_dict



    
class CellViTTorchScriptConverter:
    """
    Comprehensive CellViT to TorchScript converter based on proven patterns
    for complex segmentation models
    """
    
    def __init__(self, num_types: int = 6, mode: str = 'fast', device=None, cudnn_enabled=True):
        self.num_types = num_types
        self.mode = mode
        
        torch.backends.cudnn.enabled=cudnn_enabled
        torch.backends.cudnn.benchmark=cudnn_enabled
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            torch.set_default_device(device)
            self.device = torch.device(device)
        
        
    def _get_model(
        self, model_type: Literal["CellViT", "CellViT256", "CellViTSAM", "CellViTUNI"]
    ) -> Union[CellViT, CellViT256, CellViTSAM, CellViTUNI]:
        """Return the trained model for inference

        Args:
            model_type (str): Name of the model. Must either be one of:
                CellViT, CellViT256, CellViTSAM, CellViTUNI

        Returns:
            Union[CellViT, CellViT256, CellViTSAM, CellViTUNI]: Model
        """
        implemented_models = ["CellViT", "CellViT256", "CellViTSAM", "CellViTUNI", "CellViTVirchow"]
        if model_type not in implemented_models:
            raise NotImplementedError(
                f"Unknown model type. Please select one of {implemented_models}"
            )
        if model_type in ["CellViT"]:
            model = CellViT(
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                embed_dim=self.run_conf["model"]["embed_dim"],
                input_channels=self.run_conf["model"].get("input_channels", 3),
                depth=self.run_conf["model"]["depth"],
                num_heads=self.run_conf["model"]["num_heads"],
                extract_layers=self.run_conf["model"]["extract_layers"],
                regression_loss=self.run_conf["model"].get("regression_loss", False),
            )

        elif model_type in ["CellViT256"]:
            model = CellViT256(
                model256_path=None,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                regression_loss=self.run_conf["model"].get("regression_loss", False),
            )
        elif model_type in ["CellViTSAM"]:
            model = CellViTSAM(
                model_path=None,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                vit_structure=self.run_conf["model"]["backbone"],
                regression_loss=self.run_conf["model"].get("regression_loss", False),
            )
        elif model_type == "CellViTUNI":
            model = CellViTUNI(
                model_uni_path=None,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
            )
        elif model_type in ["CellViTVirchow"]:
            model = CellViTVirchow(
                model_virchow_path=None,
                num_nuclei_classes=self.run_conf["data"]
                ["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]
                ["num_tissue_classes"]
            )
        return model
    
    def load_cellvit_model(self, checkpoint_path: str) -> nn.Module:
        """
        Load CellViT model with proper handling of different checkpoint formats.

        Accepts two kinds of checkpoints:
        - A full CellViT checkpoint (arch: CellViTSAM / CellViT256 / etc.)
        - A LinearClassifier checkpoint produced by train_cell_classifier_head.py
          (arch: LinearClassifier). In this case the base CellViT path is read
          from the classifier config's ``cellvit_path`` key and both models are
          fused into a CellViTWithLinearClassifier wrapper.
        """
        try:
            model_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            arch = model_checkpoint["arch"]

            if arch == "LinearClassifier":
                # ---- classifier checkpoint ---------------------------------
                cls_conf = unflatten_dict(model_checkpoint["config"], ".")
                cellvit_path = cls_conf["cellvit_path"]
                print(f"→ Detected LinearClassifier checkpoint.")
                print(f"  Loading base CellViT from: {cellvit_path}")

                base_ckpt = torch.load(cellvit_path, map_location="cpu", weights_only=False)
                self.run_conf = unflatten_dict(base_ckpt["config"], ".")
                cellvit = self._get_model(model_type=base_ckpt["arch"])
                cellvit.load_state_dict(base_ckpt["model_state_dict"])
                cellvit.eval()

                embed_dim = model_checkpoint["model_state_dict"]["fc1.weight"].shape[1]
                classifier = LinearClassifier(
                    embed_dim=embed_dim,
                    hidden_dim=cls_conf["model"].get("hidden_dim", 100),
                    num_classes=cls_conf["data"]["num_classes"],
                    drop_rate=0,
                )
                classifier.load_state_dict(model_checkpoint["model_state_dict"])
                classifier.eval()

                self.model = CellViTWithLinearClassifier(cellvit, classifier)
                self.model_arch = base_ckpt["arch"]
                self.run_conf["model"]["token_patch_size"] = cellvit.patch_size
            else:
                # ---- full CellViT checkpoint (existing behaviour) -----------
                self.run_conf = unflatten_dict(model_checkpoint["config"], ".")
                self.model = self._get_model(model_type=arch)
                self.model.load_state_dict(model_checkpoint["model_state_dict"])
                self.model_arch = arch
                self.run_conf["model"]["token_patch_size"] = self.model.patch_size

            self.model.eval()
            self.model.to(self.device)
        
        
        
            # Initialize model
            # model = HoVerNet(nr_types=self.num_types, freeze=False, mode=self.mode)
            
            # Load checkpoint with multiple format support
            # checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            # if isinstance(checkpoint['desc'], dict):
            #     # Check for common keys
            #     if 'model_state_dict' in checkpoint['desc']:
            #         state_dict = checkpoint['desc']['model_state_dict']
            #     elif 'state_dict' in checkpoint['desc']:
            #         state_dict = checkpoint['desc']['state_dict']
            #     elif 'model' in checkpoint['desc']:
            #         state_dict = checkpoint['desc']['model']
            #     else:
            #         state_dict = checkpoint['desc']
            # else:
            #     state_dict = checkpoint['desc']
            #
            # # Handle DataParallel checkpoints (remove 'module.' prefix)
            # if any(k.startswith('module.') for k in state_dict.keys()):
            #     new_state_dict = {}
            #     for k, v in state_dict.items():
            #         new_key = k.replace('module.', '') if k.startswith('module.') else k
            #         new_state_dict[new_key] = v
            #     state_dict = new_state_dict
            
            # model.load_state_dict(state_dict)
            # model.eval()
            
            print(f"✓ Successfully loaded CellViT model from {checkpoint_path}")
            return self.model
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            raise
    
    def prepare_example_input(self, batch_size: int = 1, num_channel: int = 3, height: int = 1024, width: int = 1024) -> torch.Tensor:
        """
        Prepare example input for tracing (BGR format, 0-255 range)
        """
        # Create example input in BGR format with 0-255 range (CellViT specific)
        example_input = torch.randn(batch_size, 3, height, width, dtype=torch.float32)
        
        # Scale to -1.0-1.0 range (CellViT expects this)
        example_input = torch.clamp(example_input, -1.0, 1.0)
        
        return example_input.to(self.device)
    
    def validate_model_conversion(self, original_model: nn.Module, 
                                traced_model: torch.jit.ScriptModule,
                                test_input: torch.Tensor,
                                tolerance: float = 1e-2) -> bool:
        """
        Validate that traced model produces same outputs as original
        """
        print("🔍 Validating model conversion...")
        
        try:
            original_model.eval()
            traced_model.eval()
            
            with torch.no_grad():
                # Get outputs from both models
                original_output = original_model(test_input)
                traced_output = traced_model(test_input)
                
            if original_output.keys() != traced_output.keys():
                print(f"✗ Output dict format mismatch: {original_output.keys()} vs {traced_output.keys()}")
                return False
            
            # Handle multiple outputs (common in segmentation models)
            
            for k in original_output.keys():
                if isinstance(original_output[k], (list, tuple)):
                    if len(original_output[k]) != len(traced_output):
                        print(f"✗ Output count mismatch: {len(original_output)} vs {len(traced_output)}")
                        return False
                    
                    max_diff = 0
                    for i, (orig, traced) in enumerate(zip(original_output[k], traced_output[k])):
                        diff = torch.abs(orig - traced).max().item()
                        max_diff = max(max_diff, diff)
                        print(f"  Output {i}: max difference = {diff:.2e}")
                        
                        if diff > tolerance:
                            print(f"✗ Large difference in output {i}: {diff}")
                            return False
                else:
                    max_diff = torch.abs(original_output[k] - traced_output[k]).max().item()
                    print(f"  Single output: max difference = {max_diff:.2e}")
                    
                    if max_diff > tolerance:
                        print(f"✗ Large difference: {max_diff}")
                        return False
                
                    print(f"✓ Validation passed (max diff: {max_diff:.2e})")
            return True
                
        except Exception as e:
            print(f"✗ Validation failed: {e}")
            return False
    
    def handle_unsupported_operations(self, model: nn.Module) -> nn.Module:
        """
        Handle common TorchScript incompatibilities in complex models
        """
        print("🔧 Checking for TorchScript incompatibilities...")
        
        # Common fixes for medical imaging models
        def replace_unsupported_activations(module):
            for name, child in module.named_children():
                if isinstance(child, nn.modules.activation.GELU):
                    # Replace GELU with ReLU if causing issues
                    setattr(module, name, nn.ReLU(inplace=True))
                    print(f"  Replaced GELU with ReLU in {name}")
                else:
                    replace_unsupported_activations(child)
        
        # Apply fixes if needed (uncomment if you encounter issues)
        # replace_unsupported_activations(model)
        
        return model
    
    def convert_to_torchscript(self, model_path: str, 
                             output_path: str,
                             input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
                             optimize: bool = False,
                             patch: bool = False,
                             validate: bool = True,
                             save_graph: bool = False) -> torch.jit.ScriptModule:
        """
        Main conversion method with comprehensive error handling
        """
        print(f"🚀 Starting CellViT TorchScript conversion...")
        print(f"   Input: {model_path}")
        print(f"   Output: {output_path}")
        print(f"   Input shape: {input_shape}")
        
        try:
            # Step 1: Load model
            model = self.load_cellvit_model(model_path)
            model = model.to(self.device)
            
            # Step 2: Handle incompatibilities
            model = self.handle_unsupported_operations(model)
            
            # Step 3: Prepare input
            example_input = self.prepare_example_input(*input_shape)
            print(f"✓ Prepared example input: {example_input.shape}, range: [{example_input.min():.1f}, {example_input.max():.1f}]")
            
            # Step 4: Test forward pass
            print("🧪 Testing forward pass...")
            with torch.no_grad():    
                test_output = model(example_input)
            
                for k in test_output.keys():
                    if isinstance(test_output[k], (list, tuple)):
                        print(f"✓ Model outputs {len(test_output)} tensors")
                        for i, out in enumerate(test_output[k]):
                            print(f"  Output {i}: {out.shape}")
                    else:
                        print(f"✓ Model output: {test_output[k].shape}")
            
            # Step 5: Trace the model
            print("📝 Tracing model...")
            start_time = time.time()
            
            with torch.no_grad():
                # Use strict=False to handle dynamic operations better
                traced_model = torch.jit.trace(model, example_input, strict=False)
                
            trace_time = time.time() - start_time
            print(f"✓ Tracing completed in {trace_time:.2f}s")
            
            # Step 6: Validate conversion
            if validate:
                validation_passed = self.validate_model_conversion(
                    model, traced_model, example_input, tolerance=0.02
                )
                if not validation_passed:
                    raise RuntimeError("Model validation failed")
            
            # Step 7: Patch model
            if patch:
                print(f"🔧 Patch TorchScript model")
                # traced_model = patch_factories_move_to_input_device(traced_model)
                traced_model = patch_all_methods(traced_model, verbose=True)

            # Step 8: Optimize model
            # NOTE: optimize_for_inference calls freeze internally and fails when
            # patch is active (alias analysis rejects the injected aten::to nodes)
            if optimize and not patch:
                print("⚡ Optimizing traced model...")
                traced_model = torch.jit.optimize_for_inference(traced_model)
                print("✓ Model optimization completed")

            # Step 9: Freeze for inference
            # optimize_for_inference already calls freeze internally, so only
            # freeze explicitly when we skipped the optimize step.
            if not (optimize and not patch):
                if patch:
                    print("⚠️  Skipping freeze (incompatible with patched aten::to nodes)")
                else:
                    print(f"🥶 Freezing traced model...")
                    traced_model = torch.jit.freeze(traced_model)
            
            # Step 10: Save model
            print(f"💾 Saving TorchScript model to {output_path}")
            torch.jit.save(traced_model, output_path)
            
            # Step 10: Save metadata
            self.save_conversion_metadata(output_path, input_shape, model_path)
            
            # Step 11: Save computation graph (optional)
            if save_graph:
                self.save_computation_graph(traced_model, output_path)
            
            print(f"🎉 Successfully converted CellViT to TorchScript!")
            print(f"   Model saved: {output_path}")
            
            return traced_model
            
        except Exception as e:
            print(f"💥 Conversion failed: {e}")
            traceback.print_exc()
            raise
    
    def save_conversion_metadata(self, output_path: str, input_shape: Tuple, model_path: str):
        """Save metadata about the conversion"""
        metadata = {
            "original_model": model_path,
            "input_shape": list(input_shape),
            "num_types": self.num_types,
            "mode": self.mode,
            "conversion_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pytorch_version": torch.__version__,
            "input_format": "BGR",
            "input_range": "0-255",
            "device_used": str(self.device)
        }
        
        metadata_path = Path(output_path).with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata saved: {metadata_path}")
    
    def save_computation_graph(self, traced_model: torch.jit.ScriptModule, output_path: str):
        """Save the computation graph for inspection"""
        try:
            graph_path = Path(output_path).with_suffix('.graph.txt')
            with open(graph_path, 'w') as f:
                f.write(str(traced_model.graph))
            print(f"✓ Computation graph saved: {graph_path}")
        except Exception as e:
            print(f"⚠️  Could not save graph: {e}")
    
    def load_and_test_torchscript(self, torchscript_path: str, 
                                test_input: Optional[torch.Tensor] = None) -> torch.jit.ScriptModule:
        """
        Load and test a TorchScript model
        """
        print(f"📂 Loading TorchScript model from {torchscript_path}")
        
        try:
            # Load model
            model = torch.jit.load(torchscript_path, map_location=self.device)
            model.eval()
            
            # Test with example input
            if test_input is None:
                test_input = self.prepare_example_input()
            
            print("🧪 Testing loaded model...")
            with torch.no_grad():
                output = model(test_input)
                
            for k in output.keys():
                if isinstance(output[k], (list, tuple)):
                    print(f"✓ Model produces {len(output[k])} outputs")
                    for i, out in enumerate(output[k]):
                        print(f"  Output {i}: {out.shape}, range: [{out.min():.3f}, {out.max():.3f}]")
                else:
                    print(f"✓ Model output: {output[k].shape}, range: [{output[k].min():.3f}, {output[k].max():.3f}]")
            
            return model
            
        except Exception as e:
            print(f"✗ Failed to load or test model: {e}")
            raise

# Performance benchmarking utilities
class TorchScriptBenchmark:
    """Benchmark TorchScript vs original model performance"""
    
    @staticmethod
    def benchmark_models(original_model: nn.Module, 
                        torchscript_model: torch.jit.ScriptModule,
                        test_input: torch.Tensor,
                        num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark both models and return timing results
        """
        print(f"⏱️  Benchmarking models ({num_runs} runs)...")
        
        # Warm up
        for _ in tqdm(range(10)):
            with torch.no_grad():
                _ = original_model(test_input)
                _ = torchscript_model(test_input)
        
        # Benchmark original model
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in tqdm(range(num_runs)):
            with torch.no_grad():
                _ = original_model(test_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        original_time = time.time() - start
        
        # Benchmark TorchScript model
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in tqdm(range(num_runs)):
            with torch.no_grad():
                _ = torchscript_model(test_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        torchscript_time = time.time() - start
        
        results = {
            'original_time': original_time,
            'torchscript_time': torchscript_time,
            'speedup': original_time / torchscript_time,
            'original_fps': num_runs / original_time,
            'torchscript_fps': num_runs / torchscript_time
        }
        
        print(f"📊 Benchmark Results:")
        print(f"   Original model:    {original_time:.3f}s ({results['original_fps']:.1f} FPS)")
        print(f"   TorchScript model: {torchscript_time:.3f}s ({results['torchscript_fps']:.1f} FPS)")
        print(f"   Speedup: {results['speedup']:.2f}x")
        
        return results

# Usage example and main execution
def main():
    """
    Convert a CellViT (or fused CellViT+LinearClassifier) checkpoint to TorchScript.

    Two checkpoint types are accepted:
      (a) A full CellViT checkpoint saved by the upstream pre-training pipeline
          (arch field = CellViTSAM / CellViT256 / CellViTUNI / CellViTVirchow).
      (b) A LinearClassifier checkpoint produced by train_cell_classifier_head.py
          (arch field = LinearClassifier).  The base CellViT path is read from the
          classifier checkpoint's embedded config key 'cellvit_path', and both
          models are fused into a CellViTWithLinearClassifier before tracing.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Convert a CellViT or CellViT+LinearClassifier checkpoint to a frozen "
            "TorchScript (.pt) file suitable for deployment in QuPath / Java / C++ "
            "without a Python runtime."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------ required
    parser.add_argument(
        "--checkpoint",
        required=True,
        metavar="PATH",
        help=(
            "Path to the input checkpoint (.pth).  "
            "Accepted formats: "
            "(a) full CellViT checkpoint (arch = CellViTSAM, CellViT256, etc.); "
            "(b) LinearClassifier checkpoint from train_cell_classifier_head.py "
            "(arch = LinearClassifier) — the base CellViT will be loaded "
            "automatically from the path stored inside the checkpoint config."
        ),
    )

    # ------------------------------------------------------------------ optional
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Destination path for the TorchScript model (.pt).  "
            "If omitted, the output is placed next to the checkpoint with the "
            "extension replaced by '.torchscript.pt' "
            "(e.g. model_best.pth → model_best.torchscript.pt)."
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        metavar="INT",
        help=(
            "Tile height in pixels used as the tracing example input.  "
            "Must match the tile size the model was trained on.  "
            "Typical values: 1024 for SAM-H/SAM-L backbones, 256 for CellViT256."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        metavar="INT",
        help=(
            "Tile width in pixels used as the tracing example input.  "
            "Must match the tile size the model was trained on.  "
            "Typical values: 1024 for SAM-H/SAM-L backbones, 256 for CellViT256."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        metavar="STR",
        help=(
            "PyTorch device string for tracing, e.g. 'cuda', 'cuda:1', or 'cpu'.  "
            "Defaults to 'cuda' when a GPU is available, otherwise 'cpu'.  "
            "Note: the TorchScript model is device-agnostic after conversion; "
            "this only affects which device is used during the tracing step."
        ),
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help=(
            "Skip the post-tracing numerical validation that checks the traced "
            "model output matches the original PyTorch model within a tolerance "
            "of 0.02.  Useful to speed up conversion when you have already "
            "verified correctness, but not recommended for first-time conversions."
        ),
    )
    parser.add_argument(
        "--patch",
        action="store_true",
        help=(
            "Enable experimental graph-level device-portability patches that insert "
            "aten::to() calls after every tensor-factory op (aten::arange, aten::zeros, "
            "aten::ones, etc.) so they follow the device of the model input at runtime.  "
            "NOTE: This is experimental and incompatible with torch.jit.freeze/save on "
            "PyTorch 2.x due to alias-analysis limitations.  On PyTorch 2.x, CPU index "
            "tensors are implicitly copied to the correct device, so this flag is generally "
            "not needed."
        ),
    )

    args = parser.parse_args()

    # Derive output path if not specified
    checkpoint_path = args.checkpoint
    if args.output is not None:
        output_path = args.output
    else:
        cp = Path(checkpoint_path)
        output_path = str(cp.with_name(cp.stem + ".torchscript.pt"))

    # Initialize converter
    converter = CellViTTorchScriptConverter(num_types=6, mode='fast', device=args.device, cudnn_enabled=True)

    try:
        # Convert model
        traced_model = converter.convert_to_torchscript(
            model_path=checkpoint_path,
            output_path=output_path,
            input_shape=(1, 3, args.height, args.width),
            optimize=True,
            patch=args.patch,
            validate=not args.no_validate,
            save_graph=False,
        )

        # Test the converted model
        loaded_model = converter.load_and_test_torchscript(output_path)

        print("\n🎯 Conversion Summary:")
        print(f"   ✓ Original model loaded successfully")
        print(f"   ✓ TorchScript conversion completed")
        if not args.no_validate:
            print(f"   ✓ Model validation passed")
        print(f"   ✓ Model saved and tested")
        print(f"\n📁 Output files:")
        print(f"   🔹 TorchScript model: {output_path}")
        print(f"   🔹 Metadata: {Path(output_path).with_suffix('.json')}")

    except Exception as e:
        print(f"\n💥 Conversion failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Ensure the checkpoint path is correct and the file is readable")
        print("   2. For LinearClassifier checkpoints, verify 'cellvit_path' in the config points to a valid base CellViT checkpoint")
        print("   3. Use --height / --width matching your training tile size (1024 for SAM-H, 256 for CellViT256)")
        print("   4. If tracing fails due to graph patching, remove --patch to disable it (off by default)")
        print("   5. If numerical validation fails, check that the correct checkpoint is being used")

if __name__ == "__main__":
    main()
