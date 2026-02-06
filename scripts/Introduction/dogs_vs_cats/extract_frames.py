from pathlib import Path
import cv2

def extract_frames_from_video(
    video_path: Path,
    out_dir: Path,
    target_fps: float = 5.0,
    resize_hw: tuple[int, int] = (128, 128),  # (H, W)
    max_frames: int = 80
) -> int:
    """
    Extrahiert Frames aus einem Video.
    - target_fps: effektive FPS (wie viele Frames pro Sekunde wir speichern wollen)
    - resize_hw: Zielauflösung (H, W)
    - max_frames: Option A: maximal so viele Frames pro Video speichern
    Gibt zurück: Anzahl gespeicherter Frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Kann Video nicht öffnen: {video_path}")
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        # Falls FPS nicht lesbar ist, nehme ein konservatives Intervall
        src_fps = 30.0

    # Intervall: z.B. 30fps / 5fps = 6 -> jedes 6. Frame speichern
    frame_interval = max(1, int(round(src_fps / target_fps)))

    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_interval == 0:
            # Resize: cv2 erwartet (W, H)
            w = resize_hw[1]
            h = resize_hw[0]
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

            out_path = out_dir / f"frame_{saved:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

            if saved >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return saved


def main():
    project_root = Path(__file__).resolve().parents[2]  # .../tsm
    videos_root = project_root / "data" / "videos"
    frames_root = project_root / "data" / "frames"

    target_fps = 5.0
    resize_hw = (128, 128)
    max_frames = 80

    if not videos_root.exists():
        raise FileNotFoundError(f"Ordner nicht gefunden: {videos_root}")

    class_dirs = [p for p in videos_root.iterdir() if p.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"Keine Klassenordner in {videos_root} gefunden.")

    print(f"Quelle: {videos_root}")
    print(f"Ziel  : {frames_root}")
    print(f"Settings: target_fps={target_fps}, resize={resize_hw}, max_frames={max_frames}\n")

    total_videos = 0
    total_frames = 0

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        video_files = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}])

        print(f"Klasse '{class_name}': {len(video_files)} Videos")
        for vp in video_files:
            video_stem = vp.stem
            out_dir = frames_root / class_name / video_stem

            saved = extract_frames_from_video(
                video_path=vp,
                out_dir=out_dir,
                target_fps=target_fps,
                resize_hw=resize_hw,
                max_frames=max_frames
            )

            print(f"  - {vp.name}: {saved} Frames -> {out_dir}")
            total_videos += 1
            total_frames += saved

        print()

    print(f"Fertig. Videos: {total_videos}, gespeicherte Frames: {total_frames}")


if __name__ == "__main__":
    main()
