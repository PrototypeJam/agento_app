# Agento Streamlit App

This directory contains the Streamlit web interface for the Agento multi-agent planning system. Filenames have been simplified to avoid emoji or special characters.

Run the application with:

```bash
cd streamlit_app
streamlit run app.py
```

## Environment Variables

Create a `.env` file in the `agento-streamlit` directory or set these variables in your environment:

- `OPENAI_API_KEY`: **Required.** Your OpenAI API key.
- `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`: **Optional.** If you have an OpenTelemetry collector running and want to export traces to it via OTLP/HTTP, set this variable to your collector's endpoint (e.g., `http://localhost:4318/v1/traces`). If not set, trace export over OTLP will be skipped.
