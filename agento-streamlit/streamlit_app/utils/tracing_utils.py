# revised

import json
import os
import csv
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from pydantic import BaseModel
from agents import TracingProcessor, Span, Trace, add_trace_processor

# Attempt to import OpenTelemetry components for optional OTLP export
try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    OTEL_AVAILABLE = True
except Exception:
    OTEL_AVAILABLE = False

# --- Directory Setup ---
TRACES_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "traces"))
RAW_SDK_SPANS_DIR = os.path.join(TRACES_ROOT_DIR, "raw_sdk_spans")
EVAL_SETS_DIR = os.path.join(TRACES_ROOT_DIR, "eval_sets")

os.makedirs(RAW_SDK_SPANS_DIR, exist_ok=True)
os.makedirs(EVAL_SETS_DIR, exist_ok=True)
print(f"[tracing_utils] TRACES_ROOT_DIR initialized to: {TRACES_ROOT_DIR}") # Debug print


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


COST_PER_MODEL_TOKEN = {
    "gpt-4o": {"prompt": 0.005 / 1000, "completion": 0.015 / 1000},
    "gpt-4o-mini": {"prompt": 0.00015 / 1000, "completion": 0.0006 / 1000},
    "gpt-4-turbo": {"prompt": 0.01 / 1000, "completion": 0.03 / 1000},
    "gpt-4": {"prompt": 0.03 / 1000, "completion": 0.06 / 1000},
    "gpt-3.5-turbo": {"prompt": 0.0005 / 1000, "completion": 0.0015 / 1000},
}


def _calculate_cost(
    model_name: Optional[str],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
) -> Optional[float]:
    if not model_name or (prompt_tokens is None and completion_tokens is None):
        return None

    normalized_model_name = model_name.lower()

    for model_key_substring, costs in COST_PER_MODEL_TOKEN.items():
        if model_key_substring in normalized_model_name:
            calculated_cost = 0.0
            if prompt_tokens is not None:
                calculated_cost += prompt_tokens * costs["prompt"]
            if completion_tokens is not None:
                calculated_cost += completion_tokens * costs["completion"]
            return calculated_cost if calculated_cost > 0.0 else None
    return None


class AgentoTraceProcessor(TracingProcessor):
    def __init__(self, module_name: str, run_id: str):
        self.module_name = module_name
        self.run_id = run_id
        self.llm_calls_buffer: List[LLMCallRecord] = []
        self.raw_spans_buffer: List[Dict[str, Any]] = []

        self.raw_sdk_jsonl_file_path: Optional[str] = None
        self.eval_csv_file_path: Optional[str] = None
        self.current_workflow_name: Optional[str] = None

    def on_trace_start(self, trace: Trace):
        """Called when a trace is started."""
        if trace and trace.workflow_name:
            self.current_workflow_name = trace.workflow_name

    def on_trace_end(self, trace: Trace):
        """Called when a trace is finished."""
        # We handle everything in on_span_end and finalize_and_write_files
        pass

    def on_span_start(self, span: Span):
        """Called when a span is started."""
        # We don't need to do anything on span start
        pass

    def on_span_end(self, span: Span):
        """Called when a span is finished. Should not block or raise exceptions."""
        try:
            # Buffer all raw spans
            self.raw_spans_buffer.append(span.model_dump(exclude_none=True))

            # If a workflow name isn't set yet from trace, try to get it from span's trace context
            if not self.current_workflow_name and hasattr(span, "trace") and span.trace:
                self.current_workflow_name = span.trace.workflow_name

            # Check for LLM Generation Span
            is_llm_span = False
            attributes = span.attributes if hasattr(span, "attributes") else {}

            # Multiple ways to detect LLM spans
            operation_name = attributes.get("operation.name", "").lower()
            event_type = attributes.get("openai.event_type", "").lower()
            span_name = span.name.lower() if hasattr(span, "name") else ""

            if (
                "llm" in operation_name
                or "llm_run" in event_type
                or "llm.generation" in span_name
                or "generation" in span_name
            ):
                if "messages" in attributes or "input" in attributes or "prompt" in attributes:
                    is_llm_span = True

            if hasattr(span, "span_data") and span.span_data and span.span_data.__class__.__name__ == "GenerationSpanData":
                is_llm_span = True

            if is_llm_span:
                self._process_llm_span(span, attributes)

        except Exception as e:
            # Don't raise exceptions in on_span_end
            print(f"Error processing span in on_span_end: {e}")

    def _process_llm_span(self, span: Span, attributes: Dict[str, Any]):
        """Extract LLM call information from a span."""
        try:
            system_prompt_content = None
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
                                    content_str_parts.append(
                                        f"[Image: {item.get('image_url', {}).get('url', '')[:50]}...]"
                                    )
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
                if choices and isinstance(choices, list) and len(choices) > 0:
                    choice = choices[0]
                    if isinstance(choice, dict) and "message" in choice:
                        message = choice["message"]
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
            if (
                hasattr(span, "start_time")
                and hasattr(span, "end_time")
                and span.start_time
                and span.end_time
            ):
                latency = (span.end_time - span.start_time).total_seconds() * 1000

            model_name = attributes.get("model", attributes.get("openai.llm.model"))
            cost = _calculate_cost(model_name, prompt_tokens, completion_tokens)

            trace_id = span.trace_id if hasattr(span, "trace_id") else None
            span_id = span.span_id if hasattr(span, "span_id") else None
            parent_id = span.parent_id if hasattr(span, "parent_id") else None

            timestamp = datetime.now(timezone.utc).isoformat()
            if hasattr(span, "end_time") and span.end_time:
                timestamp = span.end_time.isoformat()

            record = LLMCallRecord(
                trace_id=trace_id or "unknown",
                span_id=span_id or "unknown",
                parent_span_id=parent_id,
                workflow_name=self.current_workflow_name or f"{self.module_name}_Workflow_{self.run_id}",
                module_name=self.module_name,
                agent_name=attributes.get("agent.name", attributes.get("agent_name")),
                timestamp=timestamp,
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

        except Exception as e:
            print(f"Error processing LLM span: {e}")

    def shutdown(self, timeout: Optional[float] = None):
        """Called when the application stops."""
        # We handle file writing in finalize_and_write_files
        pass

    def force_flush(self, timeout: Optional[float] = None) -> bool:
        """Forces an immediate flush of all queued spans/traces."""
        # Since we write files manually via finalize_and_write_files, we don't need to do anything here
        return True

    def finalize_and_write_files(self):
        """Writes all buffered data to their respective files for the current module run."""
        try:
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

        except Exception as e:
            print(f"Error in finalize_and_write_files: {e}")

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
    if otel_endpoint and OTEL_AVAILABLE:
        try:
            provider = TracerProvider()
            exporter = OTLPSpanExporter(endpoint=otel_endpoint)
            span_processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(span_processor)
            print(f"OTLP endpoint configured: {otel_endpoint}")
            print(
                "Note: Direct OTLP export requires bridging Agento spans to OpenTelemetry traces."
            )
        except Exception as e:
            print(f"Failed to initialize OTLP components: {e}")
    elif otel_endpoint:
        print("OpenTelemetry libraries not available. Cannot enable OTLP export.")
    else:
        print("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT not set. Skipping OTLP export.")

    return agento_file_processor
