# -*- coding: utf-8 -*-
"""
AgriHealth AI – model_runtime.py
Photo model ko chalane ke 2 tareeqe, dono ka jawab bilkul ek jaisa:

  1) ONNX (halka):  model.onnx + head.npz   -> sirf onnxruntime chahiye (PyTorch nahi)
                    Render jaise chhote free server (512 MB RAM) par bhi chal jata hai.
  2) PyTorch:       model.pt                -> torch + timm chahiye (Hugging Face par)

Folder mein model.onnx ho to ONNX chalta hai, warna model.pt.

Heatmap ka hisaab: EfficientNet ke aakhir mein "average pool + ek linear layer" hai.
Is liye Grad-CAM ke gradients har channel ke liye bas classifier ka weight / (h*w) hote hain.
Yahan wahi seedha hisaab hai, is liye heatmap pehle wale (autograd) jaisa hi banta hai,
aur PyTorch ke baghair bhi.
"""
import os

import numpy as np

import engine


class PhotoModel:
    """features (C, h, w) aur logits (classes,) dene wala model."""

    kind = ""

    def __init__(self, weight, bias):
        self.weight = np.asarray(weight, dtype=np.float32)     # (classes, C)
        self.bias = np.asarray(bias, dtype=np.float32)         # (classes,)

    def features(self, x):
        raise NotImplementedError

    def run(self, x):
        """x: (1, 3, H, W) float32  ->  features (C, h, w), logits (classes,)"""
        feats = np.asarray(self.features(x), dtype=np.float32)
        logits = self.weight @ feats.mean(axis=(1, 2)) + self.bias
        return feats, logits

    def cam(self, feats, class_index):
        """Grad-CAM (0..1) us class ke liye jo model ne chuni."""
        h, w = feats.shape[1:]
        grads = np.broadcast_to(self.weight[class_index][:, None, None] / float(h * w), feats.shape)
        return engine.grad_cam(feats, grads)


class OnnxModel(PhotoModel):
    kind = "onnx"

    def __init__(self, folder, threads):
        import onnxruntime as ort
        head = np.load(os.path.join(folder, "head.npz"))
        super().__init__(head["weight"], head["bias"])
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(os.path.join(folder, "model.onnx"), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def features(self, x):
        return self.session.run(None, {self.input_name: np.ascontiguousarray(x, dtype=np.float32)})[0][0]


class TorchModel(PhotoModel):
    kind = "torch"

    def __init__(self, folder, model_name, num_classes, threads):
        import timm
        import torch
        torch.set_num_threads(threads)
        state = torch.load(os.path.join(folder, "model.pt"), map_location="cpu", weights_only=True)
        state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}
        net = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        net.load_state_dict(state)
        net.eval()
        classifier = net.get_classifier()
        super().__init__(classifier.weight.detach().numpy(), classifier.bias.detach().numpy())
        self.torch = torch
        self.net = net

    def features(self, x):
        with self.torch.inference_mode():
            return self.net.forward_features(self.torch.from_numpy(np.ascontiguousarray(x))).numpy()[0]


def load_model(folder, config, threads=2):
    if os.path.exists(os.path.join(folder, "model.onnx")):
        return OnnxModel(folder, threads)
    return TorchModel(folder, config["model_name"], len(config["classes"]), threads)
