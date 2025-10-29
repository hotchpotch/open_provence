"""Identify MLDR questions to ignore via OpenAI LLM judgement."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import litellm
import yaml
from datasets import Dataset
from tqdm import tqdm

LOGGER = logging.getLogger("mldr_ignore")

IGNORES_PATH_DEFAULT = Path("output/eval_mldr/ignore/ignored_questions.yaml")


def load_dataset_records(dataset_path: Path) -> list[dict[str, Any]]:
    ds = Dataset.load_from_disk(str(dataset_path))
    return [cast(dict[str, Any], row) for row in ds]


def group_by_query(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["query_id"]].append(record)
    return grouped


def ensure_openai_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise OSError(
            "OPENAI_API_KEY is not set. Please export the key before running this script."
        )
    return api_key


def build_prompt(question: str, positives: list[str]) -> str:
    positives_text = "\n".join(f"{idx + 1}. {text}" for idx, text in enumerate(positives))
    if not positives_text:
        positives_text = "(none)"

    examples = """
Example 1:
Question: "京都の有名な寺院は？"
Positive passages:
1. "京都には清水寺や金閣寺などの歴史的な寺院があり..."
Assistant: {"ignore": 0, "reason": "question is well-formed and positives contain the expected information"}

Example 2:
Question: "Who wrote 'Frankenstein'?"
Positive passages:
1. "This paragraph discusses the population of Berlin."
Assistant: {"ignore": 0, "reason": "question is well-formed even if the passage is unrelated"}

Example 3:
Question: "What is the best strategy to solve this [EMPTY]?"
Positive passages:
(none)
Assistant: {"ignore": 1, "reason": "question is malformed/empty"}

Example 4:
Question: "Which city hosted the 2020 Olympics?"
Positive passages:
1. "The 2020 Summer Olympics were held in Tokyo, Japan."
2. "Tokyo welcomed athletes from around the world."
Assistant: {"ignore": 0, "reason": "positive passages clearly contain the answer"}

Example 5:
Question: "給食を配るときに気をつけることは？"
Positive passages:
1. "給食当番は感染防止のため手袋とマスクを着用する。"
2. "給食前には手洗いを徹底し..."
Assistant: {"ignore": 0, "reason": "question is coherent and positives contain relevant guidance"}

Example 6:
Question: "[BLANK QUESTION ???]"
Positive passages:
1. "This passage is about an unrelated topic."
Assistant: {"ignore": 1, "reason": "question text is incoherent or missing essential words"}
""".strip()

    return (
        "You are an evaluator that flags only malformed MLDR questions. "
        "Return JSON with fields 'ignore' (0 or 1) and 'reason'. "
        "Set ignore=1 only when the question itself is malformed, incoherent, empty, or clearly unusable. "
        "If the question is well-formed—even when the positives seem irrelevant—set ignore=0. "
        "Respond in English even if the question is in another language.\n\n"
        f"{examples}\n\n"
        f"Question: {question}\n"
        f"Positive passages:\n{positives_text}\n"
        "Assistant:"
    )


def call_llm(
    *,
    model: str,
    reasoning_effort: str | None,
    request_timeout: float,
    question: str,
    positives: list[str],
) -> dict[str, Any]:
    prompt = build_prompt(question, positives)
    temperature = 0.0
    if "gpt-5" in model.lower():
        temperature = 1.0

    extra_kwargs: dict[str, Any] = {
        "response_format": {"type": "json_object"},
        "temperature": temperature,
        "metadata": {"purpose": "ignore_qid"},
    }
    response = cast(
        dict[str, Any],
        litellm.completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return strict JSON with fields 'ignore' (0 or 1) and 'reason'. "
                        "Set ignore=1 only if the question is malformed, incoherent, or empty. "
                        "Do not set ignore=1 merely because the passages may lack an answer."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            timeout=request_timeout,
            **extra_kwargs,
        ),
    )

    choices = response.get("choices")
    if not choices:
        raise RuntimeError(f"Unexpected response payload: {response}")

    content = choices[0]["message"]["content"]
    if not content:
        raise RuntimeError(f"Empty content from LLM: {response}")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse model output as JSON: {content}") from exc

    if not isinstance(parsed, dict) or "ignore" not in parsed or "reason" not in parsed:
        raise ValueError(f"Model output missing expected fields: {parsed}")
    return parsed


@dataclass(slots=True)
class QueryExample:
    qid: str
    question: str
    positives: list[str]


def load_ignore_yaml(path: Path) -> dict[str, list[dict[str, str]]]:
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        data = {}
    for lang in ("en", "jp"):
        data.setdefault(lang, [])
    return data


def update_ignore_yaml(
    path: Path,
    existing: dict[str, list[dict[str, str]]],
    lang: str,
    entries: list[dict[str, str]],
) -> None:
    current = existing.get(lang, [])
    keep = {item["qid"]: item for item in current if "qid" in item}

    for entry in entries:
        keep[entry["qid"]] = entry

    existing[lang] = list(keep.values())
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(existing, fh, allow_unicode=True, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ignore list for MLDR questions using OpenAI."
    )
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Path to MLDR dataset directory."
    )
    parser.add_argument(
        "--lang", choices=["en", "jp"], required=True, help="Language key to update."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-nano",
        help="LLM model name routed via LiteLLM (default: gpt-5-nano).",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="low",
        help="Reasoning effort passed to the LLM (default: low).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=IGNORES_PATH_DEFAULT,
        help="Path to ignore YAML file (default: output/eval_mldr/ignore/ignored_questions.yaml).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Number of simultaneous LLM requests (default: 20).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Maximum number of retry attempts per example (default: 2).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Delay in seconds between retry attempts (default: 2.0).",
    )
    parser.add_argument(
        "--max-positive-chars",
        type=int,
        default=60000,
        help="Cap each positive passage to this many characters before sending to the LLM (default: 60000).",
    )
    args = parser.parse_args()

    ensure_openai_api_key()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s:%(message)s",
    )

    records = load_dataset_records(args.dataset)
    grouped = group_by_query(records)
    positive_text_key = "original_text"
    fallback_text_key = "pruned_text"

    flagged_entries: list[dict[str, str]] = []
    examples: list[QueryExample] = []
    max_chars = max(0, args.max_positive_chars)
    for qid, group in grouped.items():
        question = str(group[0]["query"])
        positive_passages: list[str] = []
        too_long = False
        longest_length = 0
        for row in group:
            if row.get("label") != 1:
                continue
            title = row.get("title")
            raw_text = row.get(positive_text_key) or row.get(fallback_text_key, "")
            text = str(raw_text or "")
            if title and str(title).strip():
                combined = f"{str(title).strip()}\n{text}".strip()
            else:
                combined = text.strip()

            text_length = len(combined)
            longest_length = max(longest_length, text_length)
            if max_chars > 0 and text_length > max_chars:
                too_long = True
                combined = combined[:max_chars]
            positive_passages.append(combined)

        if too_long:
            reason = (
                "Positive passage length exceeds limit "
                f"({longest_length} characters > {max_chars}). Automatically flagged."
            )
            flagged_entries.append(
                {
                    "qid": str(qid),
                    "question": question,
                    "reason": reason,
                }
            )
            LOGGER.debug(
                "Auto-flagging qid=%s due to positive length %d > %d characters.",
                qid,
                longest_length,
                max_chars,
            )
            continue

        examples.append(QueryExample(qid=str(qid), question=question, positives=positive_passages))

    async def evaluate_examples() -> None:
        semaphore = asyncio.Semaphore(max(1, args.concurrency))
        progress = tqdm(examples, desc="Evaluating questions", unit="qid")

        async def run_single(example: QueryExample) -> None:
            if not example.positives:
                LOGGER.debug(
                    "Skipping qid=%s because no positive passages were found.", example.qid
                )
                progress.update()
                return

            attempts = 0
            while attempts <= args.retries:
                attempts += 1
                try:
                    start_time = perf_counter()
                    async with semaphore:
                        judgement = await asyncio.to_thread(
                            call_llm,
                            model=args.model,
                            reasoning_effort=args.reasoning_effort,
                            request_timeout=args.request_timeout,
                            question=example.question,
                            positives=example.positives,
                        )
                    elapsed = perf_counter() - start_time
                    LOGGER.debug(
                        "qid=%s attempt=%d ignore=%s latency=%.2fs",
                        example.qid,
                        attempts,
                        judgement.get("ignore"),
                        elapsed,
                    )
                    if int(judgement.get("ignore", 0)) == 1:
                        reason = str(judgement.get("reason", "")).strip()
                        reason_lower = reason.lower()
                        about_question = any(
                            keyword in reason_lower
                            for keyword in [
                                "question",
                                "prompt",
                                "query",
                                "malformed",
                                "incoherent",
                                "blank",
                                "empty",
                            ]
                        )
                        mentions_missing_answer = any(
                            phrase in reason_lower
                            for phrase in [
                                "does not contain",
                                "no answer",
                                "answer missing",
                                "passage",
                            ]
                        )
                        if about_question and not (
                            mentions_missing_answer
                            and not any(
                                k in reason_lower
                                for k in ["malformed", "incoherent", "broken", "empty", "invalid"]
                            )
                        ):
                            flagged_entries.append(
                                {
                                    "qid": example.qid,
                                    "question": example.question,
                                    "reason": reason,
                                }
                            )
                        else:
                            LOGGER.debug(
                                "Discarding ignore=1 for qid=%s because reason does not point to a malformed question: %s",
                                example.qid,
                                reason,
                            )
                    break
                except Exception as exc:  # pragma: no cover - network errors
                    LOGGER.warning(
                        "Evaluation failed for qid=%s (attempt %d/%d): %s",
                        example.qid,
                        attempts,
                        args.retries + 1,
                        exc,
                    )
                    if attempts > args.retries:
                        LOGGER.error(
                            "Giving up on qid=%s after %d attempts.", example.qid, attempts
                        )
                        break
                    if args.retry_delay > 0:
                        await asyncio.sleep(args.retry_delay)
            progress.update()

        await asyncio.gather(*(run_single(example) for example in examples))
        progress.close()

    if examples:
        asyncio.run(evaluate_examples())

    if not flagged_entries:
        print("No questions were flagged. YAML was left unchanged.")
        return

    ignore_path = args.output
    ignore_path.parent.mkdir(parents=True, exist_ok=True)
    data = load_ignore_yaml(ignore_path)
    update_ignore_yaml(ignore_path, data, args.lang, flagged_entries)

    print(f"Updated {ignore_path} with {len(flagged_entries)} entries for language '{args.lang}'.")


if __name__ == "__main__":
    main()
