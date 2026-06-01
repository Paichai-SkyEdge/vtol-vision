#!/usr/bin/env python3
"""
Inference latency benchmark — run this ON the Jetson Orin Nano Super.

Measures per-backend latency (PT / ONNX / TRT FP16) and writes a JSON file
that generate_paper_figures.py consumes to build the latency figure.

Usage (on Jetson):
  python3 jetson_latency_bench.py --engine ~/vtol-vision/weights/best.engine
  python3 jetson_latency_bench.py --engine ~/vtol-vision/weights/best.engine \
      --pt ~/vtol-vision/weights/best.pt \
      --onnx ~/vtol-vision/weights/best.onnx

Then copy the JSON back to the dev machine:
  scp jetson:~/latency.json paper/figures/latency.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

WARMUP = 20
RUNS   = 200
IMGSZ  = 640


def bench_ultralytics(model_path: str, label: str) -> dict:
    from ultralytics import YOLO
    import numpy as np

    model = YOLO(str(model_path))
    dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)

    # warm-up
    for _ in range(WARMUP):
        model.predict(dummy, verbose=False)

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=False, imgsz=IMGSZ)
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    result = {
        "mean":   float(np.mean(arr)),
        "std":    float(np.std(arr)),
        "p50":    float(np.percentile(arr, 50)),
        "p95":    float(np.percentile(arr, 95)),
        "p99":    float(np.percentile(arr, 99)),
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
        "runs":   RUNS,
    }
    print(f"  {label}: {result['mean']:.1f} ± {result['std']:.1f} ms "
          f"(p95={result['p95']:.1f} ms)")
    return result


def bench_tensorrt_direct(engine_path: str) -> dict:
    """Direct TensorRT inference via pycuda (more accurate timing than Ultralytics)."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
    except ImportError:
        print("  [!] pycuda not found, using Ultralytics TRT wrapper instead.")
        return bench_ultralytics(engine_path, "TensorRT FP16 (.engine)")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    runtime = trt.Runtime(TRT_LOGGER)
    engine  = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()

    # Allocate buffers
    input_shape  = (1, 3, IMGSZ, IMGSZ)
    dummy_np = np.zeros(input_shape, dtype=np.float32)

    # Find binding indices
    input_idx  = engine.get_binding_index("images")
    output_idx = 1 - input_idx

    in_size  = int(np.prod(input_shape))
    out_shape = tuple(engine.get_binding_shape(output_idx))
    out_size  = int(np.prod(out_shape))

    d_input  = cuda.mem_alloc(in_size  * 4)
    d_output = cuda.mem_alloc(out_size * 4)
    h_output = np.empty(out_shape, dtype=np.float32)
    stream   = cuda.Stream()

    bindings = [None, None]
    bindings[input_idx]  = int(d_input)
    bindings[output_idx] = int(d_output)

    cuda.memcpy_htod_async(d_input, dummy_np.ravel(), stream)

    # warm-up
    for _ in range(WARMUP):
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()

    times = []
    for _ in range(RUNS):
        cuda.memcpy_htod_async(d_input, dummy_np.ravel(), stream)
        t0 = time.perf_counter()
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

    arr = np.array(times)
    result = {
        "mean":  float(np.mean(arr)),
        "std":   float(np.std(arr)),
        "p50":   float(np.percentile(arr, 50)),
        "p95":   float(np.percentile(arr, 95)),
        "p99":   float(np.percentile(arr, 99)),
        "min":   float(np.min(arr)),
        "max":   float(np.max(arr)),
        "runs":  RUNS,
    }
    label = "TensorRT FP16 (.engine)"
    print(f"  {label}: {result['mean']:.1f} ± {result['std']:.1f} ms "
          f"(p95={result['p95']:.1f} ms)")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="path to best.engine (TRT FP16)")
    ap.add_argument("--pt",   default=None,    help="path to best.pt (PyTorch FP32)")
    ap.add_argument("--onnx", default=None,    help="path to best.onnx (ONNX Runtime)")
    ap.add_argument("--out",  default="latency.json")
    args = ap.parse_args()

    print(f"Jetson latency benchmark  |  warmup={WARMUP}  runs={RUNS}  imgsz={IMGSZ}")
    results = {}

    if args.pt and Path(args.pt).exists():
        print("▶ PyTorch FP32 (.pt)...")
        results["PyTorch FP32 (.pt)"] = bench_ultralytics(args.pt, "PyTorch FP32 (.pt)")

    if args.onnx and Path(args.onnx).exists():
        print("▶ ONNX Runtime FP32...")
        results["ONNX Runtime FP32"] = bench_ultralytics(args.onnx, "ONNX Runtime FP32")

    print("▶ TensorRT FP16 (.engine)...")
    results["TensorRT FP16 (.engine)"] = bench_tensorrt_direct(args.engine)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved → {args.out}")
    print("Copy to dev machine:  scp <jetson_host>:~/latency.json paper/figures/latency.json")
    print("Then re-run:  python3 tools/generate_paper_figures.py --only latency "
          "--latency paper/figures/latency.json")


if __name__ == "__main__":
    main()
