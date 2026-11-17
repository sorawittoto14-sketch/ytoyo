"""Export the rumble detector to ONNX (for edge devices / other runtimes).

    python -m elephant_rumble.export_onnx \
        --model models/sincnet_lstm_real_elephant_data.pt \
        --out models/sincnet_lstm_real_elephant_data.onnx

ONNX graph
  input  "audio"        float32 (batch, 750)  - 3 s of audio at 250 Hz, any scale
                                                (the model normalises each window)
  output "frame_probs"  float32 (batch, 15)   - rumble probability every 0.2 s
  output "window_prob"  float32 (batch,)      - max over frames (rumble in window?)

Pre-processing on the device: take the first channel, low-pass + resample the
recording to 250 Hz, cut 3-s windows every 2 s (1-s overlap).
"""

import argparse
import json

import numpy as np
import torch
from torch import nn

from .config import CLIP_SAMPLES, DETECT_THRESHOLD, N_FRAMES, SAMPLE_RATE
from .detect import load_model

# Model description stored in the .pt checkpoint and the ONNX metadata.
REAL_DATA_INFO = {
    "model_name": "ELSENSE SincNet-LSTM",
    "project": "ELSENSE - elephant infrasound detection and early-warning device",
    "task": "elephant rumble (infrasound) detection, 10-60 Hz",
    "training_data": "real elephant recordings with expert labels",
    "test_results": "window acc 96.3%, precision 97.5%, recall 95.1%",
    "data_credit": "Cornell Elephant Listening Project",
}


class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio):
        frame_probs = torch.sigmoid(self.model(audio))
        return frame_probs, frame_probs.amax(-1)


def tag_checkpoint(path, info=REAL_DATA_INFO):
    """Write the data-provenance info into the .pt checkpoint itself."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt["model_info"] = dict(info)  # replaces any earlier description
    torch.save(ckpt, path)


def export(model_path, out_path, info=REAL_DATA_INFO, opset=17):
    import onnx

    model, window_thr = load_model(model_path)
    wrapper = ExportWrapper(model).eval()
    dummy = torch.randn(2, CLIP_SAMPLES)
    torch.onnx.export(
        wrapper, (dummy,), out_path, input_names=["audio"], output_names=["frame_probs", "window_prob"],
        dynamic_axes={"audio": {0: "batch"}, "frame_probs": {0: "batch"}, "window_prob": {0: "batch"}},
        opset_version=opset, dynamo=False,
    )
    m = onnx.load(out_path)
    meta = dict(info) | {
        "sample_rate_hz": str(SAMPLE_RATE), "window_seconds": "3", "window_hop_seconds": "2",
        "frames_per_window": str(N_FRAMES), "frame_seconds": "0.2", "band_hz": "10-60",
        "window_threshold": f"{window_thr:.3f}", "detection_threshold": str(DETECT_THRESHOLD),
    }
    for k, v in meta.items():
        p = m.metadata_props.add()
        p.key, p.value = k, str(v)
    m.doc_string = info["model_name"] + " - " + info["training_data"]
    onnx.checker.check_model(m)
    onnx.save(m, out_path)
    return verify(model, out_path)


def verify(model, onnx_path, n=64):
    """Compare ONNX Runtime with PyTorch on random and real-shaped inputs."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = (np.random.default_rng(0).normal(size=(n, CLIP_SAMPLES)) * 0.1).astype(np.float32)
    onnx_frames, onnx_win = sess.run(None, {"audio": x})
    with torch.no_grad():
        ref = torch.sigmoid(model(torch.from_numpy(x))).numpy()
    return {"max_abs_diff_frames": float(np.abs(onnx_frames - ref).max()),
            "max_abs_diff_window": float(np.abs(onnx_win - ref.max(1)).max()),
            "output_shape": list(onnx_frames.shape)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/sincnet_lstm_real_elephant_data.pt")
    ap.add_argument("--out", default="models/sincnet_lstm_real_elephant_data.onnx")
    ap.add_argument("--tag-checkpoint", action="store_true", help="also write the data info into the .pt file")
    args = ap.parse_args(argv)
    if args.tag_checkpoint:
        tag_checkpoint(args.model)
    print(json.dumps(export(args.model, args.out), indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
