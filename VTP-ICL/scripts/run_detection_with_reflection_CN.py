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
    parser = argparse.ArgumentParser(description="读取 ICL 上下文并执行检测 + 反思复核。")
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
                "下面给出若干类别的学习上下文。每类结构是："
                "图片1+文本1(坐标和类型)+图片2+文本2(坐标和类型)+文本3(类别描述)。"
                "请学习后对新图进行检测，仅输出 JSON："
                "[{\"bbox_2d\": [x1,y1,x2,y2], \"label\": \"类别名\"}]。"
            )
        }
    ]
    for c in context_obj.get("contexts", []):
        content.append({"text": f"类别：{c['label']}"})
        content.extend(c["learning_context"])
    content.append(
        {
            "text": (
                "现在请对下面这张图执行检测：找最显著一个瑕疵。"
                "严格输出 JSON，禁止额外解释。"
            )
        }
    )
    content.append({"image": query_image})
    return [{"role": "user", "content": content}]


def reflection_prompt() -> str:
    return (
        "请对上一步检测结果做复核，只在明显错误时修改：\n"
        "1) bbox 是否紧贴缺陷边界；\n"
        "2) bbox 是否明显过大或过小；\n"
        "3) 坐标是否明显偏离。\n"
        "若结果合理则原样返回。输出格式必须是："
        "[{\"bbox_2d\": [x1,y1,x2,y2], \"label\": \"类别名\"}]"
    )


def main() -> None:
    args = parse_args()
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("缺少环境变量 DASHSCOPE_API_KEY")
    dashscope.base_http_api_url = args.base_url

    context_obj = load_json(args.context_json)
    test_data = load_json(args.test_json)

    all_results = []
    for item in test_data:
        image_path = item["image"]
        print(f"\n处理: {image_path}")
        messages = build_detection_messages(context_obj, image_path)

        init_pred = stream_text(api_key, args.model_instruct, messages, "第一阶段-无思考检测")
        messages.append({"role": "assistant", "content": [{"text": init_pred}]})
        messages.append({"role": "user", "content": [{"text": reflection_prompt()}]})
        final_pred = stream_text(api_key, args.model_thinking, messages, "第二阶段-思考反思复核")

        all_results.append(
            {
                "image": image_path,
                "initial_prediction": init_pred,
                "final_prediction": final_pred,
            }
        )
        time.sleep(0.3)

    dump_json(args.output_json, {"results": all_results})
    print(f"\n检测完成，结果文件: {Path(args.output_json).resolve()}")


if __name__ == "__main__":
    main()
