# Security

This release is a local development and inference building block. By default the API binds to `127.0.0.1`. The raw vLLM endpoint should also stay on loopback or a private network.

Before exposing it, put a TLS gateway with authentication, body limits, timeouts, and rate limits in front of the API. Set the optional server-side API key described in the runtime guide. A browser bundle is public code: minification improves delivery size but does not protect secrets. Never embed a deployment credential or a shared privileged key in JavaScript.

Images are supplied as bounded base64 data URLs; the API does not fetch arbitrary image URLs. Prompts and image content are untrusted model input, not authorization instructions. Treat decisions as model predictions, not access-control checks.

The basic API has bounded admission and explicit errors. It does not include a durable queue, replay storage, or an exactly-once billing layer. Callers must handle timeouts and retryable failures. Do not claim a request was persisted unless your own deployment implements persistence.

Keep `.env`, model caches, generated reports, logs, and credentials out of version control. The repository's `.gitignore` is a convenience, not a secret scanner.

For a suspected vulnerability, use GitHub's private vulnerability reporting when available. Do not open a public issue containing a working credential or sensitive reproduction. Share a minimal, redacted example first.
