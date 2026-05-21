import argparse
from pathlib import Path

import cv2


def split_video_to_frames(video_path: str, output_dir: str | None = None, ext: str = "png") -> None:
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_dir is None:
        output_dir = video_path.with_suffix("").as_posix() + "_frames"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_idx = 0
    saved_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_name = output_dir / f"frame_{frame_idx:06d}.{ext}"
        cv2.imwrite(str(frame_name), frame)
        frame_idx += 1
        saved_count += 1

    cap.release()
    print(f"Saved {saved_count} frames to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split an .mp4 video into image frames.")
    parser.add_argument(
        "--video_path",
        default="/home/ma-user/workspace/yunfeng/DiffSynth-Studio-main/eccv/lora_gaussion/flow_match/luxury_steak_fire_grill_closeup_1080P.mp4",
        help="Path to the input .mp4 video.")
    parser.add_argument(
        "--output_dir",
        default="/home/ma-user/workspace/yunfeng/DiffSynth-Studio-main/eccv/a",
        help="Directory to save frames. Default: <video_name>_frames next to the video.",
    )
    parser.add_argument(
        "--ext",
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Output image format. Default: png.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_video_to_frames(args.video_path, args.output_dir, args.ext)
