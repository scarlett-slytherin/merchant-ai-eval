#!/usr/bin/env python3
"""Run one fresh, text-only API request per question and prompt version."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
VERSIONS = ("V0", "V1")
SPLITS = ("pilot", "dev", "holdout")
CASE_FIELDS = ("id", "platform", "question", "evidence", "source_title", "source_url", "split")
MAX_OUTPUT_TOKENS = 4096
REQUEST_TIMEOUT_SECONDS = 120
PROVIDERS = {
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1/responses"),
    "qwen": ("DASHSCOPE_API_KEY", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
    "doubao": ("ARK_API_KEY", "https://ark.cn-beijing.volces.com/api/v3/chat/completions"),
}


def parse_args(argv):
    parser = argparse.ArgumentParser(description="运行商家咨询 Prompt 评测，每题每版本发送一次请求。")
    parser.add_argument("--provider", required=True, choices=PROVIDERS)
    parser.add_argument("--model", required=True, help="账号已开通的模型 ID 或方舟推理接入点 ID")
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="pilot")
    parser.add_argument("--version", choices=(*VERSIONS, "all"), default="all")
    parser.add_argument("--limit", type=int, help="最多选取前 N 题，按数据集顺序")
    parser.add_argument("--dry-run", action="store_true", help="只显示请求数量和题目编号，不访问网络")
    parser.add_argument("--output", type=Path, help="新的 JSONL 文件路径，默认在当前目录按时间命名")
    args = parser.parse_args(argv)
    if not args.model.strip():
        parser.error("--model 需要非空的模型 ID")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 需要为正整数")
    return args


def load_inputs():
    cases = json.loads((ROOT / "dataset.json").read_text(encoding="utf-8-sig"))
    prompts = json.loads((ROOT / "prompts.json").read_text(encoding="utf-8-sig"))
    if not isinstance(cases, list) or not isinstance(prompts, dict):
        raise ValueError("dataset.json 需为数组，prompts.json 需为对象。")
    if any(not isinstance(prompts.get(version), str) or not prompts[version].strip() for version in VERSIONS):
        raise ValueError("prompts.json 需要非空的 V0 和 V1。")

    seen_ids = set()
    for number, case in enumerate(cases, start=1):
        if not isinstance(case, dict) or any(
            not isinstance(case.get(field), str) or not case[field].strip() for field in CASE_FIELDS
        ):
            raise ValueError(f"dataset.json 第 {number} 题缺少非空的必要字段。")
        if case["split"] not in SPLITS or case["id"] in seen_ids:
            raise ValueError(f"dataset.json 第 {number} 题的分组无效或题号重复。")
        seen_ids.add(case["id"])
    return cases, prompts


def user_input(case):
    return (
        f"平台：{case['platform']}\n"
        f"商家问题：{case['question']}\n\n"
        f"参考资料：{case['evidence']}\n"
        f"来源：{case['source_title']}\n"
        f"来源网址：{case['source_url']}"
    )


def request_payload(provider, model, prompt, question):
    if provider == "openai":
        return {
            "model": model,
            "instructions": prompt,
            "input": question,
            "store": False,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        }
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        "max_completion_tokens": MAX_OUTPUT_TOKENS,
    }


def extract_reply(provider, response):
    if provider == "openai":
        messages = [item for item in response["output"] if item["type"] == "message"]
        segments = [
            part.get("text", part.get("refusal", ""))
            for message in messages
            for part in message["content"]
            if part["type"] in {"output_text", "refusal"}
        ]
        reason = (response.get("incomplete_details") or {}).get("reason") or response.get("status")
        return "\n".join(segments), reason

    choice = response["choices"][0]
    message = choice["message"]
    return message.get("content") or message.get("refusal") or "", choice.get("finish_reason")


def take_sample(provider, model, key, case, version, prompt):
    payload = request_payload(provider, model, prompt, user_input(case))
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = Request(PROVIDERS[provider][1], data=data, headers={"Content-Type": "application/json"})
    request.add_unredirected_header("Authorization", f"Bearer {key}")
    sample = {
        "provider": provider,
        "requested_model": model,
        "returned_model": None,
        "case_id": case["id"],
        "split": case["split"],
        "version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "request_sha256": hashlib.sha256(data).hexdigest(),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "reply": "",
        "finish_reason": None,
    }
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as received:
            response = json.load(received)
        reply, finish_reason = extract_reply(provider, response)
        if not isinstance(reply, str):
            raise ValueError("invalid reply")
        sample.update(
            returned_model=response.get("model"),
            reply=reply.replace(key, "[redacted]"),
            finish_reason=finish_reason,
            status="ok" if reply.strip() and finish_reason in {"completed", "stop"} else "review",
        )
    except HTTPError as error:
        sample.update(status="error", error=f"HTTP {error.code}")
        error.close()
    except (URLError, TimeoutError, OSError):
        sample.update(status="error", error="network_or_timeout")
    except (AttributeError, TypeError, ValueError, KeyError, IndexError):
        sample.update(status="error", error="invalid_response")
    sample["latency_ms"] = round((time.perf_counter() - started) * 1000)
    return sample


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
        sys.stderr.reconfigure(errors="backslashreplace")
    args = parse_args(argv)
    try:
        cases, prompts = load_inputs()
    except (OSError, ValueError) as error:
        print(f"输入文件错误：{error}", file=sys.stderr)
        return 1

    selected = [case for case in cases if args.split in ("all", case["split"])][: args.limit]
    versions = VERSIONS if args.version == "all" else (args.version,)
    if not selected:
        print("所选分组没有题目。", file=sys.stderr)
        return 1

    request_count = len(selected) * len(versions)
    if args.dry_run:
        print(json.dumps({
            "provider": args.provider,
            "model": args.model,
            "split": args.split,
            "versions": versions,
            "request_count": request_count,
            "max_output_tokens_per_request": MAX_OUTPUT_TOKENS,
            "case_ids": [case["id"] for case in selected],
        }, ensure_ascii=False, indent=2))
        return 0

    key_name = PROVIDERS[args.provider][0]
    key = os.environ.get(key_name, "").strip()
    if not key:
        print(f"缺少环境变量 {key_name}。", file=sys.stderr)
        return 1

    run_time = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path(f"results-{args.provider}-{args.split}-{run_time}.jsonl")
    try:
        with output.open("x", encoding="utf-8", newline="\n") as log:
            print(f"计划发送 {request_count} 次请求。结果文件 {output}", flush=True)
            review_count = 0
            for case in selected:
                for version in versions:
                    sample = take_sample(args.provider, args.model, key, case, version, prompts[version])
                    log.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    log.flush()
                    if sample["status"] == "error":
                        print(f"{case['id']} {version} 请求停止（{sample['error']}）。已收到的记录保存在 {output}。", file=sys.stderr)
                        return 1
                    review_count += sample["status"] == "review"
    except (OSError, KeyboardInterrupt) as error:
        print(f"运行中止。结果文件 {output}。{error}", file=sys.stderr)
        return 1

    print(f"已保存 {request_count} 条记录，其中 {review_count} 条需检查输出是否完整。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
