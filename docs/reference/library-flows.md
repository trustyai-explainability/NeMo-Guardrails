# NeMo Guardrails Library Flows

    
This document lists all available flows in the NeMo Guardrails library.

## Understanding the tables
### Library
The `Library` column indicates which library within the NeMo Guardrails repository provides this flow.
To see the source code for a flow, navigate to the specified directory inside `nemoguardrails/library`,
For example, the `self_check` library is located at `nemoguardrails/library/self_check`.

### Requires a Configured LLM
Flows marked with ✓ in this column use `llm_call()` to invoke an LLM from your `config.models`. These flows:

* Require an LLM to be configured in `config.yml` under the `models` section
* Will make LLM API calls (e.g., to OpenAI, Azure OpenAI, or local LLM servers)
* May incur costs depending on your LLM provider
* Performance depends on LLM latency and quality
* Examples: Self-check rails, hallucination detection, content safety via LLM

Flows marked with ✗ do not require an LLM configuration.

### Requires External Server Calls
Flows marked with ✓ in this column make network calls to external services or APIs *other than the configured LLMs*. These flows:

* Require network connectivity to external services beyond your LLM provider
* May need additional configuration (API keys, service endpoints, credentials)
* Have external service dependencies that must be available
* Examples: GLiNER server calls, PolicyAI API, Pangea services, AutoAlign API, CrowdStrike AIDR

Flows marked with ✗ do not make external server calls (though they may still use LLMs if indicated in the previous column).

### Self-Contained Flows
Flows that are marked with ✗ in *both* columns are fully self-contained. They:

* Work entirely offline (no network required)
* Do not require LLM configuration
* Have minimal latency and no per-request costs
* Examples: Regex-based checks, local pattern matching, sensitive data detection

### Example Configs
The `Example Configs` column in the table provide locations of example configurations that use the specified flow.
To view the example, navigate to the specified directory within the `example/configs` directory of the NeMo Guardrails repository.


## Input Rails

These flows can be configured in `rails.input.flows` in your config.yml.

| Flow Name | Library (`nemoguardrails/library/...`) | Requires a Configured LLM | Requires External Server Calls | Description | Example Configs |
|-----------|----------------------------------------|---------------------------|--------------------------------|-------------|-----------------|
| `ai defense inspect prompt` | [`nemoguardrails/library/ai_defense`](../../nemoguardrails/library/ai_defense) | ✗ | ✔ | Check if the prompt is safe according to AI Defense. | [`examples/configs/ai_defense`](../../examples/configs/ai_defense) |
| `autoalign check input` | [`nemoguardrails/library/autoalign`](../../nemoguardrails/library/autoalign) | ✗ | ✗ |  | [`examples/configs/autoalign/autoalign_config`](../../examples/configs/autoalign/autoalign_config) |
| `content safety check input` | [`nemoguardrails/library/content_safety`](../../nemoguardrails/library/content_safety) | ✔ | ✗ |  | [`examples/configs/nemoguards`](../../examples/configs/nemoguards)<br/>[`examples/configs/content_safety`](../../examples/configs/content_safety)<br/>[`examples/configs/nemoguards_cache`](../../examples/configs/nemoguards_cache)<br/>[`examples/configs/content_safety_multilingual`](../../examples/configs/content_safety_multilingual)<br/>[`examples/configs/content_safety_local`](../../examples/configs/content_safety_local)<br/>[`examples/configs/content_safety_api_keys`](../../examples/configs/content_safety_api_keys)<br/>[`examples/configs/gs_content_safety/config`](../../examples/configs/gs_content_safety/config)<br/>[`examples/configs/content_safety_vision`](../../examples/configs/content_safety_vision)<br/>[`examples/configs/content_safety_reasoning`](../../examples/configs/content_safety_reasoning) |
| `crowdstrike aidr guard input` | [`nemoguardrails/library/crowdstrike_aidr`](../../nemoguardrails/library/crowdstrike_aidr) | ✗ | ✔ |  | [`examples/configs/crowdstrike_aidr`](../../examples/configs/crowdstrike_aidr) |
| `gliner detect pii on input` | [`nemoguardrails/library/gliner`](../../nemoguardrails/library/gliner) | ✗ | ✔ | Check if the user input has PII using GLiNER. | [`examples/configs/gliner/pii_detection`](../../examples/configs/gliner/pii_detection) |
| `gliner mask pii on input` | [`nemoguardrails/library/gliner`](../../nemoguardrails/library/gliner) | ✗ | ✔ | Mask any detected PII in the user input using GLiNER. | [`examples/configs/gliner/pii_masking`](../../examples/configs/gliner/pii_masking) |
| `guardrailsai check input` | [`nemoguardrails/library/guardrails_ai`](../../nemoguardrails/library/guardrails_ai) | ✗ | ✗ | Check input text using relevant Guardrails AI validators. | [`examples/configs/guardrails_ai`](../../examples/configs/guardrails_ai)<br/>[`examples/configs/guardrails_ai`](../../examples/configs/guardrails_ai) |
| `llama guard check input` | [`nemoguardrails/library/llama_guard`](../../nemoguardrails/library/llama_guard) | ✔ | ✗ |  | [`examples/configs/llama_guard`](../../examples/configs/llama_guard) |
| `pangea ai guard input` | [`nemoguardrails/library/pangea`](../../nemoguardrails/library/pangea) | ✗ | ✔ |  | [`examples/configs/pangea`](../../examples/configs/pangea) |
| `policyai moderation on input` | [`nemoguardrails/library/policyai`](../../nemoguardrails/library/policyai) | ✗ | ✔ | Guardrail based on PolicyAI assessment. | N/A |
| `regex check input` | [`nemoguardrails/library/regex`](../../nemoguardrails/library/regex) | ✗ | ✗ | Check if the user input matches any forbidden regex patterns. | N/A |
| `self check input` | [`nemoguardrails/library/self_check/input_check`](../../nemoguardrails/library/self_check/input_check) | ✔ | ✗ |  | [`examples/configs/llm/vertexai`](../../examples/configs/llm/vertexai) |
| `detect sensitive data on input` | [`nemoguardrails/library/sensitive_data_detection`](../../nemoguardrails/library/sensitive_data_detection) | ✗ | ✗ | Check if the user input has any sensitive data. | N/A |
| `mask sensitive data on input` | [`nemoguardrails/library/sensitive_data_detection`](../../nemoguardrails/library/sensitive_data_detection) | ✗ | ✗ | Mask any sensitive data found in the user input. | N/A |
| `topic safety check input` | [`nemoguardrails/library/topic_safety`](../../nemoguardrails/library/topic_safety) | ✔ | ✗ |  | [`examples/configs/nemoguards`](../../examples/configs/nemoguards)<br/>[`examples/configs/nemoguards_cache`](../../examples/configs/nemoguards_cache)<br/>[`examples/configs/topic_safety`](../../examples/configs/topic_safety) |
| `trend ai guard input` | [`nemoguardrails/library/trend_micro`](../../nemoguardrails/library/trend_micro) | ✗ | ✔ |  | [`examples/configs/trend_micro`](../../examples/configs/trend_micro) |

## Output Rails

These flows can be configured in `rails.output.flows` in your config.yml.

| Flow Name | Library (`nemoguardrails/library/...`) | Requires a Configured LLM | Requires External Server Calls | Description | Example Configs |
|-----------|----------------------------------------|---------------------------|--------------------------------|-------------|-----------------|
| `ai defense inspect response` | [`nemoguardrails/library/ai_defense`](../../nemoguardrails/library/ai_defense) | ✗ | ✔ | Check if the response is safe according to AI Defense. | [`examples/configs/ai_defense`](../../examples/configs/ai_defense) |
| `autoalign check output` | [`nemoguardrails/library/autoalign`](../../nemoguardrails/library/autoalign) | ✗ | ✗ |  | [`examples/configs/autoalign/autoalign_config`](../../examples/configs/autoalign/autoalign_config) |
| `autoalign factcheck output` | [`nemoguardrails/library/autoalign`](../../nemoguardrails/library/autoalign) | ✗ | ✗ |  | [`examples/configs/autoalign/autoalign_factcheck_config`](../../examples/configs/autoalign/autoalign_factcheck_config) |
| `autoalign groundedness output` | [`nemoguardrails/library/autoalign`](../../nemoguardrails/library/autoalign) | ✗ | ✗ |  | [`examples/configs/autoalign/autoalign_groundness_config`](../../examples/configs/autoalign/autoalign_groundness_config) |
| `content safety check output` | [`nemoguardrails/library/content_safety`](../../nemoguardrails/library/content_safety) | ✔ | ✗ |  | [`examples/configs/nemoguards`](../../examples/configs/nemoguards)<br/>[`examples/configs/content_safety`](../../examples/configs/content_safety)<br/>[`examples/configs/nemoguards_cache`](../../examples/configs/nemoguards_cache)<br/>[`examples/configs/content_safety_multilingual`](../../examples/configs/content_safety_multilingual)<br/>[`examples/configs/content_safety_local`](../../examples/configs/content_safety_local)<br/>[`examples/configs/content_safety_api_keys`](../../examples/configs/content_safety_api_keys)<br/>[`examples/configs/gs_content_safety/config`](../../examples/configs/gs_content_safety/config)<br/>[`examples/configs/content_safety_reasoning`](../../examples/configs/content_safety_reasoning) |
| `crowdstrike aidr guard output` | [`nemoguardrails/library/crowdstrike_aidr`](../../nemoguardrails/library/crowdstrike_aidr) | ✗ | ✔ |  | [`examples/configs/crowdstrike_aidr`](../../examples/configs/crowdstrike_aidr) |
| `alignscore check facts` | [`nemoguardrails/library/factchecking/align_score`](../../nemoguardrails/library/factchecking/align_score) | ✗ | ✗ |  | [`examples/configs/rag/fact_checking`](../../examples/configs/rag/fact_checking) |
| `gliner detect pii on output` | [`nemoguardrails/library/gliner`](../../nemoguardrails/library/gliner) | ✗ | ✔ | Check if the bot output has PII using GLiNER. | [`examples/configs/gliner/pii_detection`](../../examples/configs/gliner/pii_detection) |
| `gliner mask pii on output` | [`nemoguardrails/library/gliner`](../../nemoguardrails/library/gliner) | ✗ | ✔ | Mask any detected PII in the bot output using GLiNER. | [`examples/configs/gliner/pii_masking`](../../examples/configs/gliner/pii_masking) |
| `guardrailsai check output` | [`nemoguardrails/library/guardrails_ai`](../../nemoguardrails/library/guardrails_ai) | ✗ | ✗ | Check output text using relevant Guardrails AI validators. | [`examples/configs/guardrails_ai`](../../examples/configs/guardrails_ai) |
| `hallucination warning` | [`nemoguardrails/library/hallucination`](../../nemoguardrails/library/hallucination) | ✔ | ✗ | Warning rail for hallucination. | N/A |
| `self check hallucination` | [`nemoguardrails/library/hallucination`](../../nemoguardrails/library/hallucination) | ✔ | ✗ | Output rail for checking hallucinations. | [`examples/configs/rag/custom_rag_output_rails`](../../examples/configs/rag/custom_rag_output_rails) |
| `injection detection` | [`nemoguardrails/library/injection_detection`](../../nemoguardrails/library/injection_detection) | ✗ | ✗ |  | N/A |
| `llama guard check output` | [`nemoguardrails/library/llama_guard`](../../nemoguardrails/library/llama_guard) | ✔ | ✗ |  | [`examples/configs/llama_guard`](../../examples/configs/llama_guard) |
| `pangea ai guard output` | [`nemoguardrails/library/pangea`](../../nemoguardrails/library/pangea) | ✗ | ✔ |  | [`examples/configs/pangea`](../../examples/configs/pangea) |
| `policyai moderation on output` | [`nemoguardrails/library/policyai`](../../nemoguardrails/library/policyai) | ✗ | ✔ | Guardrail based on PolicyAI assessment. | N/A |
| `regex check output` | [`nemoguardrails/library/regex`](../../nemoguardrails/library/regex) | ✗ | ✗ | Check if the bot output matches any forbidden regex patterns. | N/A |
| `self check facts` | [`nemoguardrails/library/self_check/facts`](../../nemoguardrails/library/self_check/facts) | ✔ | ✗ |  | [`examples/configs/rag/custom_rag_output_rails`](../../examples/configs/rag/custom_rag_output_rails)<br/>[`examples/configs/llm/hf_pipeline_llama2`](../../examples/configs/llm/hf_pipeline_llama2) |
| `self check output` | [`nemoguardrails/library/self_check/output_check`](../../nemoguardrails/library/self_check/output_check) | ✔ | ✗ |  | [`examples/configs/self_check_thinking`](../../examples/configs/self_check_thinking)<br/>[`examples/configs/llm/vertexai`](../../examples/configs/llm/vertexai) |
| `detect sensitive data on output` | [`nemoguardrails/library/sensitive_data_detection`](../../nemoguardrails/library/sensitive_data_detection) | ✗ | ✗ | Check if the bot output has any sensitive data. | N/A |
| `mask sensitive data on output` | [`nemoguardrails/library/sensitive_data_detection`](../../nemoguardrails/library/sensitive_data_detection) | ✗ | ✗ | Mask any sensitive data found in the bot output. | N/A |
| `trend ai guard output` | [`nemoguardrails/library/trend_micro`](../../nemoguardrails/library/trend_micro) | ✗ | ✔ |  | [`examples/configs/trend_micro`](../../examples/configs/trend_micro) |

## Retrieval Rails

These flows can be configured in `rails.retrieval.flows` in your config.yml.

| Flow Name | Library (`nemoguardrails/library/...`) | Requires a Configured LLM | Requires External Server Calls | Description | Example Configs |
|-----------|----------------------------------------|---------------------------|--------------------------------|-------------|-----------------|
| `gliner detect pii on retrieval` | [`nemoguardrails/library/gliner`](../../nemoguardrails/library/gliner) | ✗ | ✔ | Check if the relevant chunks from the knowledge base have any PII using GLiNER. | N/A |
| `gliner mask pii on retrieval` | [`nemoguardrails/library/gliner`](../../nemoguardrails/library/gliner) | ✗ | ✔ | Mask any detected PII in the relevant chunks from the knowledge base using GLiNER. | N/A |
| `regex check retrieval` | [`nemoguardrails/library/regex`](../../nemoguardrails/library/regex) | ✗ | ✗ |  | N/A |
| `detect sensitive data on retrieval` | [`nemoguardrails/library/sensitive_data_detection`](../../nemoguardrails/library/sensitive_data_detection) | ✗ | ✗ | Check if the relevant chunks from the knowledge base have any sensitive data. | N/A |
| `mask sensitive data on retrieval` | [`nemoguardrails/library/sensitive_data_detection`](../../nemoguardrails/library/sensitive_data_detection) | ✗ | ✗ | Mask any sensitive data found in the relevant chunks from the knowledge base. | N/A |

## Statistics

* Total flows: 43
  * Self-contained (no external deps or LLM): 17
  * Requires external dependencies: 16
  * Uses LLM from `config.models`: 10
* Input rails: 16
* Output rails: 22
* Retrieval rails: 5
* Dialog rails: 0
