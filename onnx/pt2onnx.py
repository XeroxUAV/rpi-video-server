"""Convert YOLO .pt weights (incl. YOLO26) to ONNX for onnxruntime.

Uses only already-installed packages (ultralytics + torch + onnx).
This script never installs anything.

Example:
    python onnx/pt2onnx.py --pt weights/yolo26n.pt --output weights/yolo26n.onnx
    python onnx/pt2onnx.py --pt yolo26n.pt --output yolo26n.onnx --imgsz 480,640 --opset 12
"""

import argparse
import os
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Convert YOLO .pt to ONNX (onnxruntime-ready)."
    )
    # .pt input (positional + flagged for flexibility)
    p.add_argument("pt_input", nargs="?", default=None, help=".pt file path (positional)")
    p.add_argument("--pt", "--weights", "--model", dest="pt", default=None,
                   help=".pt file path (e.g. yolo26n.pt)")
    # export / output path
    p.add_argument("--output", "--export", "--onnx", "--save", dest="output", default=None,
                   help="Export .onnx file path (default: same name as .pt)")
    # common export knobs
    p.add_argument("--imgsz", type=str, default="480,640",
                   help="Export size as height,width (matches ultralytics). Default: 480,640 (= 640x480 WxH camera frame)")
    p.add_argument("--opset", type=int, default=12, help="ONNX opset version (default: 12)")
    p.add_argument("--half", action="store_true", help="Export FP16 (GPU only, default: FP32)")
    p.add_argument("--dynamic", action="store_true", help="Dynamic axes (batch/height/width)")
    p.add_argument("--simplify", action="store_true", help="Simplify ONNX via onnxslim")
    p.add_argument("--device", default="cpu", help="Export device, e.g. cpu or 0 (default: cpu)")
    p.add_argument("--no-verify", action="store_true", help="Skip onnxruntime verification")
    return p.parse_args(argv)


def parse_imgsz(s):
    """Accept '640' -> 640, or '480,640' / '480x640' -> [480, 640] ([h, w], ultralytics order)."""
    s = str(s).strip().lower().replace(",", "x")
    parts = s.split("x") if "x" in s else [s]
    try:
        nums = [int(x) for x in parts if x]
    except ValueError:
        print(f"error: invalid --imgsz '{s}', use 640 or 480,640 ([h,w])", file=sys.stderr)
        sys.exit(2)
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return nums  # [h, w]
    print(f"error: invalid --imgsz '{s}', use 640 or 480,640 ([h,w])", file=sys.stderr)
    sys.exit(2)


def resolve_paths(args):
    pt_path = args.pt or args.pt_input
    if not pt_path:
        print("error: provide .pt path: positional arg or --pt <file.pt>", file=sys.stderr)
        sys.exit(2)
    if not os.path.isfile(pt_path):
        print(f"error: .pt file not found: {pt_path}", file=sys.stderr)
        sys.exit(1)
    output = args.output
    if not output:
        base, _ = os.path.splitext(pt_path)
        output = base + ".onnx"
    elif os.path.isdir(output):
        output = os.path.join(output, os.path.splitext(os.path.basename(pt_path))[0] + ".onnx")
    elif not output.lower().endswith(".onnx"):
        output += ".onnx"
    return pt_path, output


def main(argv=None):
    args = parse_args(argv)
    pt_path, output = resolve_paths(args)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("error: 'ultralytics' is not installed in this environment.", file=sys.stderr)
        print("Install it yourself first, e.g.: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)

    print(f"[pt2onnx] Loading: {pt_path}")
    model = YOLO(pt_path)

    print(f"[pt2onnx] Exporting to: {output}")
    exported = model.export(
        format="onnx",
        imgsz=parse_imgsz(args.imgsz),
        opset=args.opset,
        half=args.half,
        dynamic=args.dynamic,
        simplify=args.simplify,
        device=args.device,
    )
    # ultralytics returns path str; it derives name from model stem, so move/rename if needed
    exported = str(exported)
    if os.path.abspath(exported) != os.path.abspath(output):
        import shutil
        shutil.move(exported, output)
        print(f"[pt2onnx] Renamed {exported} -> {output}")
    else:
        print(f"[pt2onnx] Saved: {output}")

    if not args.no_verify:
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(output, providers=["CPUExecutionProvider"])
            print("[pt2onnx] onnxruntime OK. Inputs:")
            for i in sess.get_inputs():
                print(f"  - {i.name}: {i.shape} ({i.type})")
            print("[pt2onnx] Outputs:")
            for o in sess.get_outputs():
                print(f"  - {o.name}: {o.shape} ({o.type})")
        except ImportError:
            print("[pt2onnx] warning: onnxruntime not installed, skipping verify.")
        except Exception as e:
            print(f"[pt2onnx] warning: onnxruntime check failed: {e}", file=sys.stderr)

    print("[pt2onnx] Done.")
    return output


if __name__ == "__main__":
    main()
