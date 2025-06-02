import json
import os
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional

from pydantic import BaseModel
from agents import TraceProcessor, Span, Trace, add_trace_processor
from agents.processors import OTLPHTTPTraceSpanProcessor

# --- Directory Setup ---
TRACES_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "traces"))
RAW_SDK_SPANS_DIR = os.path.join(TRACES_ROOT_DIR, "raw_sdk_spans")
EVAL_SETS_DIR = os.path.join(TRACES_ROOT_DIR, "eval_sets")

os.makedirs(RAW_SDK_SPANS_DIR, exist_ok=True)
os.makedirs(EVAL_SETS_DIR, exist_ok=True)


class LLMCallRecord(BaseModel):
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    workflow_name: str
    module_name: str
    agent_name: Optional[str] = None
    timestamp: str
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    full_input_prompt: str
    input_tool_results_json: Optional[str] = None
    llm_output_text: Optional[str] = None
    output_tool_calls_json: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    expected_output: str = ""


COST_PER_MODEL_PER_TOKEN = {
    "gpt-4o": {"prompt": 0.005 / 1000, "completion": 0.015 / 1000, "default_avg": 0.010 / 1000},
    "gpt-4": {"prompt": 0.03 / 1000, "completion": 0.06 / 1000, "default_avg": 0.045 / 1000},
    "gpt-3.5-turbo": {"prompt": 0.0005 / 1000, "completion": 0.0015 / 1000, "default_avg": 0.001 / 1000},
}


def _calculate_cost(
    model_name: Optional[str],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    total_tokens: Optional[int],
) -> Optional[float]:
    if not model_name:
        return None

    normalized_model_name = model_name.lower()
    for model_key, costs in COST_PER_MODEL_PER_TOKEN.items():
        if model_key in normalized_model_name:
            if prompt_tokens is not None and completion_tokens is not None:
                return (prompt_tokens * costs["prompt"]) + (completion_tokens * costs["completion"])
            elif total_tokens is not None:
                return total_tokens * costs["default_avg"]
    return None


class AgentoTraceProcessor(TraceProcessor):
    def __init__(self, module_name: str, run_id: str):
        self.module_name = module_name
        self.run_id = run_id
        self.llm_calls_buffer: List[LLMCallRecord] = []
        self.raw_spans_buffer: List[Dict[str, Any]] = []

        self.raw_sdk_jsonl_file_path: Optional[str] = None
        self.eval_csv_file_path: Optional[str] = None
        self.current_workflow_name: Optional[str] = None

    def process_span(self, span: Span):
        self.raw_spans_buffer.append(span.model_dump(exclude_none=True))

        if not self.current_workflow_name and span.trace:
            self.current_workflow_name = span.trace.workflow_name

        is_llm_span = False
        operation_name = span.attributes.get("operation.name", "").lower()
        event_type = span.attributes.get("openai.event_type", "").lower()
        span_name = span.name.lower()
        if (
            "llm" in operation_name
            or "llm_run" in event_type
            or "llm.generation" in span_name
            or "generation" in span_name
        ):
            if "messages" in span.attributes or "input" in span.attributes or "prompt" in span.attributes:
                is_llm_span = True
        if span.span_data and span.span_data.__class__.__name__ == "GenerationSpanData":
            is_llm_span = True

        if is_llm_span:
            attributes = span.attributes
            full_input_parts = []
            input_tool_results_list = []
            input_messages = attributes.get("input", attributes.get("messages"))
            if isinstance(input_messages, list):
                for msg in input_messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content_str_parts = []
                        for item in content:
                            if isinstance(item, dict) and "type" in item:
                                if item["type"] == "text":
                                    content_str_parts.append(item.get("text", ""))
                                elif item["type"] == "image_url":
                                    content_str_parts.append(f"[Image: {item.get('image_url', {}).get('url', '')[:50]}...]")
                                else:
                                    content_str_parts.append(f"[{item['type']}]")
                            else:
                                content_str_parts.append(str(item))
                        content = "\n".join(content_str_parts)

                    full_input_parts.append(f"{role}: {content}")
                    if role == "system":
                        system_prompt_content = content
                    if role == "tool":
                        tool_call_id = msg.get("tool_call_id", "unknown_tool_call")
                        input_tool_results_list.append({"tool_call_id": tool_call_id, "output": content})
            elif isinstance(input_messages, str):
                full_input_parts.append(input_messages)
            full_input_prompt_str = "\n".join(full_input_parts)
            input_tool_results_json_str = json.dumps(input_tool_results_list) if input_tool_results_list else None

            llm_output_text_content = None
            output_tool_calls_list = []
            raw_output = attributes.get("output")
            if isinstance(raw_output, dict):
                choices = raw_output.get("choices", [])
                if choices and isinstance(choices, list) and choices[0].get("message"):
                    message = choices[0]["message"]
                    llm_output_text_content = message.get("content")
                    if message.get("tool_calls"):
                        output_tool_calls_list = message["tool_calls"]
            elif isinstance(raw_output, str):
                llm_output_text_content = raw_output

            output_tool_calls_json_str = json.dumps(output_tool_calls_list) if output_tool_calls_list else None

            usage = attributes.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")

            latency = None
            if span.start_time and span.end_time:
                latency = (span.end_time - span.start_time).total_seconds() * 1000

            model_name = attributes.get("model", attributes.get("openai.llm.model"))
            cost = _calculate_cost(model_name, prompt_tokens, completion_tokens, total_tokens)

            record = LLMCallRecord(
                trace_id=span.trace_id,
                span_id=span.span_id,
                parent_span_id=span.parent_id,
                workflow_name=self.current_workflow_name or f"{self.module_name}_Workflow_{self.run_id}",
                module_name=self.module_name,
                agent_name=span.attributes.get("agent.name", span.attributes.get("agent_name")),
                timestamp=span.end_time.isoformat() if span.end_time else datetime.now().isoformat(),
                model=model_name,
                system_prompt=system_prompt_content,
                full_input_prompt=full_input_prompt_str,
                input_tool_results_json=input_tool_results_json_str,
                llm_output_text=llm_output_text_content,
                output_tool_calls_json=output_tool_calls_json_str,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency,
                cost_usd=cost,
                expected_output="",
            )
            self.llm_calls_buffer.append(record)

    def process_trace(self, trace: Trace):
        if trace and trace.workflow_name:
            self.current_workflow_name = trace.workflow_name

    def finalize_and_write_files(self):
        self.raw_sdk_jsonl_file_path = os.path.join(
            RAW_SDK_SPANS_DIR, f"raw_sdk_spans_{self.module_name}_{self.run_id}.jsonl"
        )
        with open(self.raw_sdk_jsonl_file_path, "w", encoding="utf-8") as f:
            for span_dict in self.raw_spans_buffer:
                f.write(json.dumps(span_dict) + "\n")
        print(f"Saved raw SDK spans to: {self.raw_sdk_jsonl_file_path}")

        if self.llm_calls_buffer:
            self.eval_csv_file_path = os.path.join(
                EVAL_SETS_DIR, f"eval_data_{self.module_name}_{self.run_id}.csv"
            )
            field_names = LLMCallRecord.model_fields.keys()
            with open(self.eval_csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=field_names)
                writer.writeheader()
                for record in self.llm_calls_buffer:
                    writer.writerow(record.model_dump(exclude_none=True))
            print(f"Saved eval data to: {self.eval_csv_file_path}")
        self.raw_spans_buffer = []
        self.llm_calls_buffer = []

    def get_generated_file_paths(self) -> Dict[str, Optional[str]]:
        return {
            "raw_sdk_spans_jsonl": self.raw_sdk_jsonl_file_path,
            "eval_data_csv": self.eval_csv_file_path,
        }


def init_tracing(module_name: str, run_id: str) -> AgentoTraceProcessor:
    agento_file_processor = AgentoTraceProcessor(module_name=module_name, run_id=run_id)
    add_trace_processor(agento_file_processor)
    print(f"Registered AgentoTraceProcessor for {module_name}, run {run_id}")

    otel_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if otel_endpoint:
        try:
            otel_sdk_processor = OTLPHTTPTraceSpanProcessor(endpoint=otel_endpoint)
            add_trace_processor(otel_sdk_processor)
            print(f"Registered OTLPHTTPTraceSpanProcessor to endpoint: {otel_endpoint}")
        except Exception as e:
            print(f"Failed to initialize OTLPHTTPTraceSpanProcessor: {e}")
    else:
        print("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT not set. Skipping OTLPHTTPTraceSpanProcessor.")

    return agento_file_processor
