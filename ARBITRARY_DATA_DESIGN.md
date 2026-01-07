# Arbitrary Data Uploads for RLMEnv — Design Document

## Scope and goals
Enable **users (data providers)** to pass arbitrary data types into RLMEnv while keeping the **designer** workload low and preserving strict safety boundaries. The **RLM** (agent) runs only in the sandbox and can execute arbitrary sandbox code to explore/deserialize data.

Goals:
- Support text by default with zero designer effort.
- Allow designers to override serialization/deserialization for non‑text types.
- Provide a clean path to add first‑party serializers over time (dataframes, images, arrays, nested containers, etc.).
- Keep metadata minimal, generic, and auto‑extracted.
- Fail early and clearly when a type is unsupported.

Non‑goals (for now):
- External URLs/object storage as data sources.
- Complex type‑specific metadata extraction inside RLMEnv.
- Sandbox hardening beyond current trust model (FIFO hijacking etc.).

---

## Terminology
- **User**: data provider (uploads the data only).
- **Designer**: environment author (defines serializers/deserializers, config).
- **RLM**: agent using the RLM scaffold inside the sandbox.

---

## Key requirements

### 1) Single data‑provision path (file‑only)
Designers should always provide data in one way. The system always writes payloads to files and passes a file path into the sandbox. Designers can override serialization/deserialization but should not need to manage transport details.

### 2) Strict trust boundary
- Do **not** execute user (data‑provider) code on the host.
- Designer code may run on the host if it is reviewed and trusted.
- The RLM runs in the sandbox only.

### 3) No external URLs (for now)
- Data must flow via host → sandbox uploads, not external URLs.
- External URLs add auth/SSRF complexity and are deferred.

### 4) Minimal generic metadata
- Always include `type` (`str(type(data))`).
- Include `size` when `__len__` is available.
- If uploaded to a file, include `path`, `file_size`, and `hash`.
- Include `format` or `dtype` only if it is known and not too token‑heavy.
- No dtype‑specific metadata extraction in RLMEnv; the RLM can inspect data directly.

### 5) Early failure with clear errors
- If data is unsupported, fail before sandbox work.
- Error messages should be explicit about the unsupported type and how to fix it (specify dtype or provide a custom serializer).

---

## Proposed architecture

### A) Serializer/deserializer registry (first‑party + designer‑provided)
Maintain a registry in a utilities module:
- Each entry provides:
  - `dtype` (string identifier)
  - `can_handle(obj)` (optional; for auto‑detection)
  - `serialize(obj) -> (payload_bytes, metadata, file_name, format, encoding)`
  - `deserialize(payload, metadata) -> obj` (sandbox side)

The registry holds:
- Built‑in serializers (text + JSON containers by default; more added over time).
- Designer‑provided serializers registered explicitly.

### B) Optional `dtype` field
Designers may set `dtype` explicitly:
- If `dtype` is present:
  - Validate it against the registry.
  - If not supported → hard fail with actionable error.
- If `dtype` is absent:
  - Run a **conservative** auto‑detector.
  - If detected → use that serializer and record the resolved dtype.
  - If not detected → hard fail (no guessing).

### C) Auto‑detection rules (conservative)
- Use strict precedence (e.g., dataframe/image/array before generic containers).
- Avoid expensive deep inspection; sample or check top‑level only.
- If uncertain → return “unknown” and fail.

### D) Transport (host → sandbox)
- Use a single path from the designer’s perspective.
- Under the hood: **always write payloads to files** and pass the file path into the sandbox.
- Do not attempt uploads larger than sandbox disk size.

### E) Metadata policy
Minimal fields (always if available):
- `type`: `str(type(data))`
- `size`: `len(data)` if defined
- `path`: file path in sandbox
- `file_size`: bytes on disk
- `hash`: checksum of payload
- `dtype`: resolved dtype string (if known)

The RLM can inspect data directly rather than relying on richer metadata.

---

## Error handling
- Fail early when:
  - `dtype` is provided but unsupported.
  - auto‑detection fails.
  - payload exceeds sandbox storage or upload limits.
- Error messages must name the unsupported type and the expected next step:
  - “Provide a custom serializer” or “Specify dtype=...”.

---

## Extension strategy
- Start with text default (current behavior).
- Add first‑party serializers incrementally (dataframes, images, arrays, nested containers).
- Provide a simple registration API for designers to add custom serializers.

---

## Open questions
- How to expose the registry to designers (config file, decorator, or registry module)?
- Whether to allow explicit `format` separate from `dtype` (e.g., `dtype=dataframe`, `format=parquet`).
- Where to run the deserializer (worker init vs first REPL call).

---

## Summary
This design keeps the **designer** experience simple (single file‑based path, optional overrides), protects the host from untrusted **user** code, and allows strict, explicit handling of complex data types. The registry + optional dtype model avoids silent errors while supporting gradual expansion of first‑party serializers.
