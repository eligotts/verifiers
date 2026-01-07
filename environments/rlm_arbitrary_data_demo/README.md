# RLM Arbitrary Data Demo

This toy environment exercises the custom data serialization path in `RLMEnv`.

The environment supports multiple `context_dtype` options so you can test different
serializers from a single entrypoint:

- `text` → string (default serializer)
- `json` → dict/list (default serializer)
- `tuple` → custom serializer/deserializer in this environment
- `polars` → custom serializer/deserializer in this environment

Run:

```bash
vf-eval rlm-arbitrary-data-demo -n 1
```

To pick a dtype explicitly:

```bash
vf-eval rlm-arbitrary-data-demo -n 1 --env-arg context_dtype=polars
```
