from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import dashscope
from dashscope import MultiModalConversation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据可视化样本生成类别描述，并构建 ICL 学习上下文。")
    parser.add_argument("--manifest-json", type=str, default="outputs/vis_manifest.json")
    parser.add_argument("--output-json", type=str, default="outputs/icl_context.json")
    parser.add_argument("--model", type=str, default="qwen3-vl-8b-thinking")
    parser.add_argument("--base-url", type=str, default="https://dashscope.aliyuncs.com/api/v1")
    return parser.parse_args()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def group_by_label(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        label = str(r["label"])
        grouped.setdefault(label, []).append(r)
    return grouped


def call_description_model(api_key: str, model: str, label: str, sample_a: dict, sample_b: dict) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"text": "请根据两张布匹瑕疵可视化图片，总结该类瑕疵的区别化视觉描述。只输出一段中文描述。"},
                {"text": f"类别名：{label}"},
                {"image": sample_a["visualized_image"]},
                {"image": sample_b["visualized_image"]},
            ],
        }
    ]
    response = MultiModalConversation.call(
        api_key=api_key,
        model=model,
        messages=messages,
        stream=False,
        enable_thinking=False,
    )
    content_list = response.output.choices[0].message.get("content", [])
    if content_list and isinstance(content_list, list):
        return content_list[0].get("text", "").strip()
    return ""


def main() -> None:
    args = parse_args()
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("缺少环境变量 DASHSCOPE_API_KEY")
    dashscope.base_http_api_url = args.base_url

    records = load_json(args.manifest_json)
    grouped = group_by_label(records)

    contexts = []
    for label, items in grouped.items():
        if len(items) < 2:
            continue
        sample_a, sample_b = items[0], items[1]
        desc = call_description_model(api_key, args.model, label, sample_a, sample_b)
        contexts.append(
            {
                "label": label,
                "learning_context": [
                    {"image": sample_a["visualized_image"]},
                    {"text": sample_a["annotation_text"]},
                    {"image": sample_b["visualized_image"]},
                    {"text": sample_b["annotation_text"]},
                    {"text": desc},
                ],
                "generated_description": desc,
            }
        )

    dump_json(args.output_json, {"contexts": contexts})
    print(f"ICL 上下文构建完成: {len(contexts)} 类")
    print(f"输出文件: {Path(args.output_json).resolve()}")


if __name__ == "__main__":
    main()
