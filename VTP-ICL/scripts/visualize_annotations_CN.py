from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 train 标注可视化并保存。")
    parser.add_argument("--train-json", type=str, default="data/train_example.json")
    parser.add_argument("--image-root", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="outputs/vis_train")
    parser.add_argument("--manifest-out", type=str, default="outputs/vis_manifest.json")
    parser.add_argument("--placeholder-size", type=int, default=1024)
    return parser.parse_args()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_gpt_value(item: dict[str, Any]) -> str:
    for turn in item.get("conversations", []):
        if turn.get("from") == "gpt":
            return turn.get("value", "").strip()
    return ""


def parse_annotation(gpt_value: str) -> tuple[list[int], str]:
    try:
        parsed = json.loads(gpt_value)
        first = parsed[0]
        bbox = [int(x) for x in first["bbox_2d"]]
        label = str(first["label"])
        return bbox, label
    except Exception:
        numbers = re.findall(r"\d+", gpt_value)
        if len(numbers) >= 4:
            bbox = [int(numbers[0]), int(numbers[1]), int(numbers[2]), int(numbers[3])]
        else:
            bbox = [128, 128, 512, 512]
        m = re.search(r"label\"?\s*:\s*\"([^\"]+)\"", gpt_value)
        label = m.group(1) if m else "class name"
        return bbox, label


def open_or_create_image(image_path: Path, placeholder_size: int) -> Image.Image:
    if image_path.exists():
        return Image.open(image_path).convert("RGB")
    return Image.new("RGB", (placeholder_size, placeholder_size), (30, 30, 30))


def main() -> None:
    args = parse_args()
    train_data = load_json(args.train_json)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for idx, item in enumerate(train_data):
        rel_image = item["image"]
        gpt_value = get_gpt_value(item)
        bbox, label = parse_annotation(gpt_value)

        src_path = image_root / rel_image
        image = open_or_create_image(src_path, args.placeholder_size)
        draw = ImageDraw.Draw(image)
        draw.rectangle(bbox, outline=(255, 0, 0), width=4)
        draw.text((bbox[0] + 4, max(0, bbox[1] - 18)), label, fill=(255, 0, 0))

        vis_name = f"{idx:04d}_{Path(rel_image).stem}_vis.png"
        vis_path = output_dir / vis_name
        image.save(vis_path)

        manifest.append(
            {
                "source_image": rel_image,
                "visualized_image": str(vis_path).replace("\\", "/"),
                "annotation_text": gpt_value,
                "label": label,
                "bbox_2d": bbox,
            }
        )

    dump_json(args.manifest_out, manifest)
    print(f"可视化完成: {len(manifest)} 张")
    print(f"清单文件: {Path(args.manifest_out).resolve()}")


if __name__ == "__main__":
    main()
