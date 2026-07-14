"""Short-lived batch preprocessor for official Wan2.2 Animate inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _probe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,nb_read_frames",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    rate = stream.get("avg_frame_rate", "0/1").split("/")
    fps = round(float(rate[0]) / max(float(rate[1]), 1.0))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "frame_count": int(
            next(
                (value for value in (stream.get("nb_read_frames"), stream.get("nb_frames"))
                 if value not in (None, "N/A")),
                0,
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--wan-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("animate", "replace"), required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", choices=("480p", "720p"), default="720p")
    parser.add_argument("--subject-selection", choices=("largest", "center"), default="largest")
    parser.add_argument("--retarget-pose", action="store_true")
    parser.add_argument("--use-flux", action="store_true")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--kernel", type=int, default=7)
    parser.add_argument("--w-len", type=int, default=1)
    parser.add_argument("--h-len", type=int, default=1)
    args = parser.parse_args()

    import onnxruntime
    if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
        raise RuntimeError(
            "Wan Animate requires ONNX CUDAExecutionProvider; install onnxruntime-gpu "
            "and remove the CPU-only onnxruntime package"
        )

    preprocess_root = args.wan_dir / "wan" / "modules" / "animate" / "preprocess"
    if not preprocess_root.is_dir():
        raise FileNotFoundError(f"Wan Animate preprocessing code not found: {preprocess_root}")
    sys.path.insert(0, str(preprocess_root))
    from process_pipepline import ProcessPipeline

    checkpoint_dir = args.model_dir / "process_checkpoint"
    sam_path = checkpoint_dir / "sam2" / "sam2_hiera_large.pt" if args.mode == "replace" else None
    flux_path = checkpoint_dir / "FLUX.1-Kontext-dev" if args.use_flux else None
    pipeline = ProcessPipeline(
        det_checkpoint_path=str(checkpoint_dir / "det" / "yolov10m.onnx"),
        pose2d_checkpoint_path=str(checkpoint_dir / "pose2d" / "vitpose_h_wholebody.onnx"),
        sam_checkpoint_path=str(sam_path) if sam_path else None,
        flux_kontext_path=str(flux_path) if flux_path else None,
    )
    pipeline.pose2d.detector.select_type = "center" if args.subject_selection == "center" else "max"
    resolution = [832, 480] if args.resolution == "480p" else [1280, 720]

    payload = json.loads(args.batch.read_text(encoding="utf-8"))
    for item in payload["items"]:
        output = Path(item["output_path"])
        output.mkdir(parents=True, exist_ok=True)
        pipeline(
            video_path=item["video_path"],
            refer_image_path=item["reference_path"],
            output_path=str(output),
            resolution_area=resolution,
            fps=args.fps,
            iterations=args.iterations,
            k=args.kernel,
            w_len=args.w_len,
            h_len=args.h_len,
            retarget_flag=args.retarget_pose,
            use_flux=args.use_flux,
            replace_flag=args.mode == "replace",
        )
        info = _probe(output / "src_pose.mp4")
        face_info = _probe(output / "src_face.mp4")
        if info["frame_count"] <= 0 or face_info["frame_count"] != info["frame_count"]:
            raise RuntimeError(f"Pose/face preprocessing frame mismatch for {item['shot_id']}")
        if info["width"] % 16 or info["height"] % 16:
            raise RuntimeError(f"Animate conditioning dimensions must be divisible by 16: {info}")
        if args.mode == "replace":
            for name in ("src_bg.mp4", "src_mask.mp4"):
                extra = _probe(output / name)
                if extra["frame_count"] != info["frame_count"]:
                    raise RuntimeError(f"{name} frame mismatch for {item['shot_id']}")
        prepared = {
            "shot_id": item["shot_id"],
            "prepared_dir": str(output.resolve()),
            "driver_uri": item["driver_uri"],
            "start_sec": item["start_sec"],
            "end_sec": item["end_sec"],
            **info,
            "cache_hit": False,
        }
        (output / "manifest.json").write_text(
            json.dumps({"cache_key": item["cache_key"], "prepared": prepared}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
