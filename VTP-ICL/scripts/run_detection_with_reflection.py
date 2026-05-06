from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import dashscope
from dashscope import MultiModalConversation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load ICL context and run detection + reflection review.")
    parser.add_argument("--context-json", type=str, default="outputs/icl_context.json")
    parser.add_argument("--test-json", type=str, default="data/test_example.json")
    parser.add_argument("--output-json", type=str, default="outputs/detection_results.json")
    parser.add_argument("--model-instruct", type=str, default="qwen3-vl-8b-instruct")
    parser.add_argument("--model-thinking", type=str, default="qwen3-vl-8b-thinking")
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


def stream_text(api_key: str, model: str, messages: list[dict], stage: str) -> str:
    print(f"\n[{stage}] {model}")
    text_all = ""
    response = MultiModalConversation.call(
        api_key=api_key,
        model=model,
        messages=messages,
        stream=True,
        thinking_budget=2048,
    )
    for chunk in response:
        msg = chunk.output.choices[0].message
        content = msg.get("content", [])
        if content and isinstance(content, list):
            text = content[0].get("text", "")
            if text:
                print(text, end="", flush=True)
                text_all += text
    print()
    return text_all.strip()


def build_detection_messages(context_obj: dict[str, Any], query_image: str) -> list[dict]:
    content = [
        {
            "text": (
                "You will receive learned context for multiple defect classes. "
                "Each class follows: image1 + text1(bbox and label) + "
                "image2 + text2(bbox and label) + text3(class description). "
                "After learning, detect the most salient defect in the new image. "
                "Output JSON only: "
                "[{\"bbox_2d\": [x1,y1,x2,y2], \"label\": \"class name\"}]"
            )
        }
    ]
    for c in context_obj.get("contexts", []):
        content.append({"text": f"Class: {c['label']}"})
        content.extend(c["learning_context"])
    content.append(
        {
            "text": (
                "Now perform detection on the next image. "
                "Return one most salient defect only. "
                "Strict JSON output, no extra explanation."
            )
        }
    )
    content.append({"image": query_image})
    return [{"role": "user", "content": content}]


def reflection_prompt() -> str:
    return (
        "Review your previous detection result and revise only if there is a clear error:\n"
        "1) Is bbox aligned with defect boundaries?\n"
        "2) Is bbox size obviously too large or too small?\n"
        "3) Are coordinates clearly shifted?\n"
        "If the current result is reasonable, return it unchanged. "
        "Output format must be: "
        "[{\"bbox_2d\": [x1,y1,x2,y2], \"label\": \"class name\"}]"
    )


def main() -> None:
    args = parse_args()
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing environment variable: DASHSCOPE_API_KEY")
    dashscope.base_http_api_url = args.base_url

    context_obj = load_json(args.context_json)
    test_data = load_json(args.test_json)

    all_results = []
    for item in test_data:
        image_path = item["image"]
        print(f"\nProcessing: {image_path}")
        messages = build_detection_messages(context_obj, image_path)

        init_pred = stream_text(api_key, args.model_instruct, messages, "Stage-1 direct detection")
        messages.append({"role": "assistant", "content": [{"text": init_pred}]})
        messages.append({"role": "user", "content": [{"text": reflection_prompt()}]})
        final_pred = stream_text(api_key, args.model_thinking, messages, "Stage-2 reflection review")

        all_results.append(
            {
                "image": image_path,
                "initial_prediction": init_pred,
                "final_prediction": final_pred,
            }
        )
        time.sleep(0.3)

    dump_json(args.output_json, {"results": all_results})
    print(f"\nDetection completed. Result file: {Path(args.output_json).resolve()}")


if __name__ == "__main__":
    main()
