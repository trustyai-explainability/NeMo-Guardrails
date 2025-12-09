# Detections API Integration for NeMo Guardrails

## Overview

This integration enables NeMo Guardrails to communicate with external detector services that implement the Detections API v1/text/contents protocol, providing a standardized interface for content safety checking without requiring detector logic within NeMo.

**Key Features:**
- **Protocol-agnostic architecture**: Base interface pattern supports multiple detector API protocols (Detections API, KServe V1, future APIs)
- **Configuration-driven**: Add/remove detectors via ConfigMap updates only
- **Service-based detection**: Detectors run as independent microservices with rich metadata
- **Extensible design**: Add support for new API protocols by implementing two methods (request builder and response parser)
- **No code duplication**: Common HTTP, error handling, and orchestration logic shared across all detector types
- **Parallel execution**: All detectors run concurrently using asyncio.gather() for optimal performance
- **System error separation**: Distinguishes content violations from infrastructure failures (timeouts, HTTP errors)

## Architecture

    User Input → NeMo Guardrails → Detections API Detector Services → vLLM (if safe) → Response

**Components:**
- **NeMo Guardrails** (CPU) - Orchestration and flow control
- **Detections API Detectors** (CPU/GPU) - Content safety services implementing v1/text/contents protocol (this guide demonstrates Granite Guardian HAP as an example)
- **vLLM** (GPU) - LLM inference

### Design: Base Interface Pattern

This integration introduces a base interface architecture that eliminates code duplication when supporting multiple detector API protocols.

**File Structure:**
```
nemoguardrails/library/detector_clients/
├── base.py              # BaseDetectorClient interface (shared logic)
├── detections_api.py    # Detections API v1/text/contents client
├── actions.py           # NeMo action functions
└── __init__.py          # Python package marker
```

**Why This Design:**

Traditional approach would duplicate HTTP logic, error handling, and orchestration for each new API protocol. The base interface isolates what varies (request/response formats) from what stays constant (HTTP communication, error handling).

**What's Shared (in base.py):**
- HTTP session management with connection pooling
- Authentication header handling (per-detector and global fallback)
- Timeout and error handling
- Standard `DetectorResult` model

**What's API-Specific (in detections_api.py):**
- Request format: `{"contents": [text], "detector_params": {}}`
- Response parsing: Nested array structure `[[{detection1}, detection2}]]`
- Detection aggregation logic (multiple detections per text)
- Threshold and filtering logic

**Benefits:**
- Add new API protocol = implement 2 methods (`build_request`, `parse_response`)
- No code changes to add detectors (ConfigMap only)
- Same orchestration logic for all detector types
- Extensible for future protocols (OpenAI Moderation, Perspective API, etc.)

## Prerequisites

- OpenShift cluster with KServe installed
- Access to Quay.io or container registry for pulling images
- vLLM deployment for LLM inference (or alternative OpenAI-compatible endpoint)

## Requirements

**This integration communicates with external services implementing the Detections API v1/text/contents protocol.**

The Detections API provides structured detection results with rich metadata (spans, categories, confidence scores) rather than raw model outputs. Services must implement the standardized request/response format described below.

### API Contract

This integration uses **Detections API v1/text/contents protocol**.

**Protocol:** REST API with detector-specific routing via headers

**Requirements:**
- Endpoint path: `/api/v1/text/contents`
- Request header: `detector-id` specifying which detector to invoke
- Request body: `{"contents": ["text"], "detector_params": {}}`
- Response: Nested array of detection objects `[[{detection1}, {detection2}, ...]]`

**Request Format:**
```json
POST /api/v1/text/contents
Header: detector-id: granite-guardian-hap

{
  "contents": ["text to analyze"],
  "detector_params": {}
}
```

**Response Format:**
```json
[[
  {
    "start": 0,
    "end": 20,
    "detection_type": "pii",
    "detection": "EmailAddress",
    "score": 0.95,
    "text": "matching text span",
    "evidence": {},
    "metadata": {}
  }
]]
```

Each detection includes:
- `start`, `end`: Character span indices in input text
- `detection_type`: Broad category (pii, toxicity, etc.)
- `detection`: Specific detection class
- `score`: Confidence score (0.0-1.0)
- `text`: Detected text span

## How It Works

### Detection Flow

1. User sends message to NeMo Guardrails via HTTP POST to `/v1/chat/completions`
2. NeMo loads configuration from ConfigMap and triggers input safety flow defined in `rails.co`
3. `detections_api_check_all_detectors()` action executes, running all configured detectors in parallel
4. For each detector:
   - `DetectionsAPIClient` builds request: `{"contents": [text], "detector_params": {}}`
   - HTTP POST sent to detector service with `detector-id` header
   - Detector service processes text and returns structured detections
   - Parser extracts detections from nested array response `[[...]]`
5. Each detection is evaluated:
   - If `detection.score >= threshold`: Detection triggers blocking
   - Multiple detections per text are supported
   - Highest scoring detection determines overall score
6. Results aggregation:
   - System errors (timeouts, connection failures): Request blocked, tracked in `unavailable_detectors`
   - Content violations: Request blocked, tracked in `blocking_detectors` with full metadata
   - All pass: Request proceeds to vLLM for generation
7. Response returned to user (blocked message or LLM-generated response)

### Base Interface Pattern

The integration uses object-oriented design to eliminate code duplication across different detector API protocols.

**BaseDetectorClient (Abstract Class):**
```python
class BaseDetectorClient(ABC):
    @abstractmethod
    async def detect(text: str) -> DetectorResult

    @abstractmethod
    def build_request(text: str) -> dict

    @abstractmethod
    def parse_response(response: dict, http_status: int) -> DetectorResult

    # Shared implementations:
    async def _call_endpoint(...)  # HTTP communication
    def _handle_error(...)          # Error handling
```

**DetectionsAPIClient (Implementation):**
```python
class DetectionsAPIClient(BaseDetectorClient):
    def build_request(text: str) -> dict:
        # Detections API specific format
        return {"contents": [text], "detector_params": {}}

    def parse_response(response: dict, http_status: int) -> DetectorResult:
        # Parse [[{detection1}, {detection2}]]
        # Apply threshold filtering
        # Return standardized DetectorResult
```

**Adding New API Protocol:**

To support a new detector API (e.g., OpenAI Moderation, Perspective API):
1. Create new client class inheriting from `BaseDetectorClient`
2. Implement `build_request()` for the API's request format
3. Implement `parse_response()` for the API's response format
4. Add `@action()` decorated functions in `actions.py` that use the new client
5. Reuse all HTTP, auth, error handling from base class

### Detection Logic

**Multiple Detections Handling:**

Detections API services can return multiple detections for a single text (e.g., two email addresses, one SSN). The parser:
1. Extracts all detections from nested array structure
2. Filters detections by threshold: `score >= threshold`
3. Blocks if **ANY** detection meets threshold (fail-safe approach)
4. Returns highest score as primary score
5. Includes all detection details in metadata for auditing

**Example:**
```
Input: "Email me at test@example.com or call 555-1234"

Response: [[
  {detection: "EmailAddress", score: 0.99},
  {detection: "PhoneNumber", score: 0.85}
]]

With threshold=0.5:
- Both detections >= 0.5
- Content blocked
- Primary score: 0.99 (highest)
- Label: "pii:EmailAddress" (highest scoring detection)
- Metadata includes both detections
```

**Score Aggregation:**
- `score`: Highest individual detection score
- `metadata.detection_count`: Number of detections above threshold
- `metadata.individual_scores`: All scores for analysis

### Error Handling

The system distinguishes between infrastructure errors and content violations.

### System Error Handling

The system distinguishes between **content violations** (actual detections) and **system errors** (infrastructure failures like timeouts, HTTP errors, configuration issues).

**Behavior:**
- System errors tracked separately in `unavailable_detectors` list
- Requests with system errors are blocked (fail-safe approach)
- Clear error messages indicate which detectors are unavailable vs which found violations

**System Error Labels:**
`ERROR`, `HTTP_ERROR`, `TIMEOUT`, `NOT_FOUND`, `VALIDATION_ERROR`, `INVALID_RESPONSE`, `CONFIG_ERROR`

This separation enables better operational monitoring and clearer user feedback.
**System Errors:**
- HTTP errors (404, 422, 500, 503)
- Network timeouts
- Invalid response formats
- Result: `allowed=False`, `label="ERROR"` or `"TIMEOUT"`
- Tracked in `unavailable_detectors` list
- User message: "Service temporarily unavailable"

**Content Violations:**
- Successful detection with score >= threshold
- Result: `allowed=False`, `label="{type}:{detection}"`
- Tracked in `blocking_detectors` list with full metadata
- User message: Details which detectors blocked and scores

**Multiple Detector Failures:**

When running multiple detectors, the system provides comprehensive feedback showing all blocking detectors and any unavailable services, enabling both user communication and operational monitoring.

## Deployment Guide

### Prerequisites

- OpenShift cluster with KServe installed
- Namespace: `<your-namespace>` (this guide uses examples with placeholder)
- Access to Quay.io for pulling images
- vLLM or other OpenAI-compatible LLM endpoint for generation

**This integration requires external Detections API services to be deployed.**

Services must implement the v1/text/contents protocol with the request/response format described in the Requirements section.

### Deployment Options

**Option A: Using TrustyAI Guardrails Detectors (Recommended)**

Deploy detectors from the [guardrails-detectors repository](https://github.com/trustyai-explainability/guardrails-detectors) which provides production-ready HuggingFace-based detectors implementing the Detections API protocol.

**Option B: Deploy Your Own Detections API Service**

Implement a service that exposes `/api/v1/text/contents` endpoint following the API contract. Refer to the guardrails-detectors repository for reference implementations.

This guide demonstrates Option A with Granite Guardian HAP detector.

### Step 1: Deploy Granite Guardian HAP Detector

Granite Guardian requires model storage via MinIO (S3-compatible object storage running in-cluster) and uses a PVC-based approach to download and serve the model.

**Why MinIO:** KServe expects S3-compatible storage for models. MinIO provides this locally without external dependencies, enabling disconnected cluster deployments.

#### Deploy Model Storage and MinIO

Create `granite-guardian-storage.yaml`:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: minio-guardrails-guardian
spec:
  ports:
    - name: minio-client-port
      port: 9000
      protocol: TCP
      targetPort: 9000
  selector:
    app: minio-guardrails-guardian
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: guardrails-models-claim-guardian
spec:
  accessModes:
    - ReadWriteOnce
  volumeMode: Filesystem
  resources:
    requests:
      storage: 100Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: guardrails-container-deployment-guardian
  labels:
    app: minio-guardrails-guardian
spec:
  replicas: 1
  selector:
    matchLabels:
      app: minio-guardrails-guardian
  template:
    metadata:
      labels:
        app: minio-guardrails-guardian
        maistra.io/expose-route: 'true'
      name: minio-guardrails-guardian
    spec:
      volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: guardrails-models-claim-guardian
      initContainers:
        - name: download-model
          image: quay.io/rgeada/llm_downloader:latest
          securityContext:
            fsGroup: 1001
          command:
            - bash
            - -c
            - |
              model="ibm-granite/granite-guardian-3.0-2b"
              echo "Starting download of ${model}"
              /tmp/venv/bin/huggingface-cli download $model --local-dir /mnt/models/huggingface/$(basename $model)
              echo "Download complete!"
          resources:
            limits:
              memory: "2Gi"
              cpu: "2"
          volumeMounts:
            - mountPath: "/mnt/models/"
              name: model-volume
      containers:
        - args:
            - server
            - /models
          env:
            - name: MINIO_ACCESS_KEY
              value: THEACCESSKEY
            - name: MINIO_SECRET_KEY
              value: THESECRETKEY
          image: quay.io/trustyai/modelmesh-minio-examples:latest
          name: minio
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop:
                - ALL
            seccompProfile:
              type: RuntimeDefault
          volumeMounts:
            - mountPath: "/models/"
              name: model-volume
---
apiVersion: v1
kind: Secret
metadata:
  name: aws-connection-minio-data-connection-guardrails-guardian
  labels:
    opendatahub.io/dashboard: 'true'
    opendatahub.io/managed: 'true'
  annotations:
    opendatahub.io/connection-type: s3
    openshift.io/display-name: Minio Data Connection
data:
  AWS_ACCESS_KEY_ID: VEhFQUNDRVNTS0VZ
  AWS_DEFAULT_REGION: dXMtc291dGg=
  AWS_S3_BUCKET: aHVnZ2luZ2ZhY2U=
  AWS_S3_ENDPOINT: aHR0cDovL21pbmlvLWd1YXJkcmFpbHMtZ3VhcmRpYW46OTAwMA==
  AWS_SECRET_ACCESS_KEY: VEhFU0VDUkVUS0VZ
type: Opaque
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: user-one
---
kind: RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  name: user-one-view
subjects:
  - kind: ServiceAccount
    name: user-one
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: view
```

Deploy:
```bash
oc apply -f granite-guardian-storage.yaml -n <your-namespace>
```

Monitor model download (takes 5-10 minutes for ~5GB model):
```bash
oc logs -f deployment/guardrails-container-deployment-guardian -n <your-namespace> -c download-model
```

Wait for "Download complete!" message.

Verify MinIO is running:
```bash
oc get pods -n <your-namespace> | grep guardrails-container
```

Expected: Pod shows `2/2 Running` (init container completed, MinIO running)

#### Deploy ServingRuntime for Granite Guardian

Create `granite-guardian-runtime.yaml`:
```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: ServingRuntime
metadata:
  name: guardrails-detector-runtime-guardian
  annotations:
    openshift.io/display-name: Guardrails Detector ServingRuntime for KServe
    opendatahub.io/recommended-accelerators: '["nvidia.com/gpu"]'
  labels:
    opendatahub.io/dashboard: 'true'
spec:
  annotations:
    prometheus.io/port: '8000'
    prometheus.io/path: '/metrics'
  multiModel: false
  supportedModelFormats:
    - autoSelect: true
      name: guardrails-detector-huggingface
  containers:
    - name: kserve-container
      image: quay.io/rh-ee-mmisiura/guardrails-detector-huggingface:3d51741
      command:
        - uvicorn
        - app:app
      args:
        - "--workers"
        - "1"
        - "--host"
        - "0.0.0.0"
        - "--port"
        - "8000"
        - "--log-config"
        - "/common/log_conf.yaml"
      env:
        - name: MODEL_DIR
          value: /mnt/models
        - name: HF_HOME
          value: /tmp/hf_home
        - name: DETECTOR_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
      ports:
        - containerPort: 8000
          protocol: TCP
      resources:
        requests:
          memory: "18Gi"
          cpu: "1"
        limits:
          memory: "20Gi"
          cpu: "2"
```

Deploy:
```bash
oc apply -f granite-guardian-runtime.yaml -n <your-namespace>
```

Verify:
```bash
oc get servingruntime -n <your-namespace> | grep guardian
```

Expected: `guardrails-detector-runtime-guardian` appears in list

#### Deploy Granite Guardian InferenceService

Create `granite-guardian-isvc.yaml`:
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: guardrails-detector-ibm-guardian
  labels:
    opendatahub.io/dashboard: 'true'
  annotations:
    openshift.io/display-name: guardrails-detector-ibm-guardian
    security.opendatahub.io/enable-auth: 'true'
    serving.knative.openshift.io/enablePassthrough: 'true'
    sidecar.istio.io/inject: 'true'
    sidecar.istio.io/rewriteAppHTTPProbers: 'true'
    serving.kserve.io/deploymentMode: RawDeployment
spec:
  predictor:
    maxReplicas: 1
    minReplicas: 1
    model:
      modelFormat:
        name: guardrails-detector-huggingface
      name: ''
      runtime: guardrails-detector-runtime-guardian
      storage:
        key: aws-connection-minio-data-connection-guardrails-guardian
        path: granite-guardian-3.0-2b
```

Deploy:
```bash
oc apply -f granite-guardian-isvc.yaml -n <your-namespace>
```

Wait for predictor pod to start and load model (3-5 minutes):
```bash
oc get pods -n <your-namespace> | grep guardrails-detector-ibm-guardian

# Watch logs
oc logs -f -n <your-namespace> $(oc get pods -n <your-namespace> -l serving.kserve.io/inferenceservice=guardrails-detector-ibm-guardian -o name | head -1) -c kserve-container
```

Expected log output:
```
Model type detected: causal_lm
Application startup complete.
Uvicorn running on http://0.0.0.0:8000
```

Verify InferenceService is ready:
```bash
oc get inferenceservice guardrails-detector-ibm-guardian -n <your-namespace>
```

Expected: `READY = True`

**Note:** Granite Guardian runs on CPU by default. Inference takes 30-120 seconds per request. For production, consider deploying on GPU nodes or increasing timeout configuration.

### Step 2: Deploy vLLM Inference Service

vLLM uses a PVC-based approach to pre-download the Phi-3-mini model. This avoids runtime dependencies on HuggingFace and uses Red Hat's official AI Inference Server image.

Create `vllm-inferenceservice.yml`:
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: phi3-model-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: phi3-model-downloader
spec:
  replicas: 1
  selector:
    matchLabels:
      app: phi3-downloader
  template:
    metadata:
      labels:
        app: phi3-downloader
    spec:
      initContainers:
        - name: download-model
          image: quay.io/rgeada/llm_downloader:latest
          command:
            - bash
            - -c
            - |
              echo "Downloading Phi-3-mini"
              /tmp/venv/bin/huggingface-cli download microsoft/Phi-3-mini-4k-instruct --local-dir /mnt/models/phi3-mini
              echo "Download complete!"
          volumeMounts:
            - name: model-storage
              mountPath: /mnt/models
      containers:
        - name: placeholder
          image: registry.access.redhat.com/ubi9/ubi-minimal:latest
          command: ["sleep", "infinity"]
      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: phi3-model-pvc
---
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: vllm-phi3
spec:
  predictor:
    containers:
      - name: kserve-container
        image: registry.redhat.io/rhaiis/vllm-cuda-rhel9:3
        args:
          - --model=/mnt/models/phi3-mini
          - --host=0.0.0.0
          - --port=8080
          - --served-model-name=phi3-mini
          - --max-model-len=4096
          - --gpu-memory-utilization=0.7
          - --trust-remote-code
          - --dtype=half
        env:
          - name: HF_HOME
            value: /tmp/hf_cache
        volumeMounts:
          - name: model-storage
            mountPath: /mnt/models
            readOnly: true
        resources:
          limits:
            nvidia.com/gpu: 1
            cpu: "6"
            memory: "24Gi"
          requests:
            nvidia.com/gpu: 1
            cpu: "2"
            memory: "8Gi"
    volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: phi3-model-pvc
```
Deploy:

```bash
oc apply -f vllm-inferenceservice.yml -n <your-namespace>
```

Monitor model download progress:

```bash
oc logs -n <your-namespace> -l app=phi3-downloader -c download-model -f
```

Wait for "Download complete!" message. The Phi-3-mini model is approximately 8GB and may take 3-5 minutes to download.
Verify vLLM is running:

```bash
oc get inferenceservice vllm-phi3 -n <your-namespace>
oc get pods -n <your-namespace> | grep vllm-phi3
```

Expected: `vllm-phi3` InferenceService shows `READY = True` and pod shows `1/1 Running`.

### Step 3: Deploy NeMo Guardrails ConfigMap

The ConfigMap contains detector configurations and flow definitions. Detectors are registered in the `detections_api_detectors` section with their endpoint URLs and detection parameters.

Create `nemo-detections-configmap.yaml`:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nemo-detections-configmap
data:
  config.yaml: |
    rails:
      config:
        detections_api_detectors:
          granite_hap:
            inference_endpoint: "http://guardrails-detector-ibm-guardian-predictor.<your-namespace>.svc.cluster.local:8000/api/v1/text/contents"
            detector_id: "granite-guardian-hap"
            threshold: 0.5
            timeout: 120
            detector_params: {}
      input:
        flows:
          - check_input_safety_detections_api
    models:
      - type: main
        engine: vllm_openai
        model: phi3-mini
        parameters:
          openai_api_base: http://vllm-phi3-predictor.<your-namespace>.svc.cluster.local:8080/v1
          openai_api_key: sk-dummy-key
    instructions:
      - type: general
        content: |
          You are a helpful AI assistant.

  rails.co: |
    define bot blocked by detector
      "Input blocked by content safety detectors"

    define bot output blocked by detector
      "I apologize, but I cannot provide that response."

    define bot service unavailable
      "Service temporarily unavailable"

    define flow check_input_safety_detections_api
        $input_result = execute detections_api_check_all_detectors

        if $input_result.unavailable_detectors
            bot service unavailable
            stop

        if not $input_result.allowed
            bot blocked by detector
            stop

    define flow check_output_safety_detections_api
        $output_result = execute detections_api_check_all_detectors

        if $output_result.unavailable_detectors
            bot service unavailable
            stop

        if not $output_result.allowed
            bot output blocked by detector
            stop
```

**Configuration Fields:**
- `inference_endpoint`: Full URL to detector's `/api/v1/text/contents` endpoint
- `detector_id`: Identifier sent in `detector-id` header (detector-specific)
- `threshold`: Minimum score to trigger blocking (0.0-1.0)
- `timeout`: Request timeout in seconds (increase for CPU-based detectors)
- `detector_params`: Optional detector-specific parameters (sent in request body)

**Important:**
- Timeout should be 120+ seconds for CPU-based detectors like Granite Guardian
- Replace `<your-namespace>` with your actual namespace
- `detector_id` must match what the detector service expects

Deploy:
```bash
oc apply -f nemo-detections-configmap.yaml -n <your-namespace>
```

Verify:
```bash
oc get configmap nemo-detections-configmap -n <your-namespace>
```

### Step 4: Deploy NeMo Guardrails Server

Create `nemo-deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nemo-guardrails-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nemo-guardrails
  template:
    metadata:
      labels:
        app: nemo-guardrails
    spec:
      containers:
      - name: nemo-guardrails
        image: quay.io/rh-ee-stondapu/trustyai-nemo:latest
        imagePullPolicy: Always
        env:
        - name: CONFIG_ID
          value: production
        - name: OPENAI_API_KEY
          value: sk-dummy-key
        - name: DETECTIONS_API_KEY
          value: "your-global-token"
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: config-volume
          mountPath: /app/config/production
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
      volumes:
      - name: config-volume
        configMap:
          name: nemo-detections-configmap
---
apiVersion: v1
kind: Service
metadata:
  name: nemo-guardrails-server
spec:
  selector:
    app: nemo-guardrails
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
---
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: nemo-guardrails-server
spec:
  port:
    targetPort: 8000
  tls:
    termination: edge
    insecureEdgeTerminationPolicy: Allow
  to:
    kind: Service
    name: nemo-guardrails-server
```

Deploy:
```bash
oc apply -f nemo-deployment.yaml -n <your-namespace>
```

Get the external route URL:
```bash
YOUR_ROUTE="http://$(oc get route nemo-guardrails-server -n <your-namespace> -o jsonpath='{.spec.host}')"
echo "NeMo Guardrails URL: $YOUR_ROUTE"
```

Verify NeMo is running:
```bash
oc get pods -n <your-namespace> | grep nemo-guardrails-server
```

Expected: Pod shows `1/1 Running`

Check logs to confirm detector loaded:
```bash
oc logs -n <your-namespace> $(oc get pods -n <your-namespace> -l app=nemo-guardrails -o name | head -1)
```

Expected log output should show:
```
Configuration validated. Starting server...
Application startup complete.
Uvicorn running on http://0.0.0.0:8000
```

No "Failed to register" errors should appear.

## Testing

### Unit Testing

The integration includes **104 comprehensive unit tests** with **97%+ code coverage**.

**Run tests:**
```bash
poetry run pytest tests/test_detector_clients_*.py -v

# With coverage
poetry run pytest tests/test_detector_clients_*.py \
  --cov=nemoguardrails/library/detector_clients \
  --cov-report=term-missing
```

**Test coverage:**
- All error handling paths (timeouts, HTTP errors, malformed responses)
- System error classification and separation
- Parallel detector execution
- Edge cases (threshold boundaries, missing fields)

### Testing on a cluster
Extract the NeMo route for testing:
```bash
YOUR_ROUTE="http://$(oc get route nemo-guardrails-server -n <your-namespace> -o jsonpath='{.spec.host}')"
echo "Testing against: $YOUR_ROUTE"
```

### Test 1: Safe Content (Should Pass)

What this tests: Verifies detectors allow safe content and LLM generates response.
```bash
curl -X POST $YOUR_ROUTE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "production",
    "messages": [
      {"role": "user", "content": "Hello, how are you today?"}
    ]
  }'
```

**Expected Output:**
```json
{
    "messages":[
        {
            "role":"assistant",
            "content":"I am an AI, so I don't have feelings, but I'm ready and functioning optimally. How can I help you today?"
        }
    ]
}
```

Content passes all detectors and LLM generates helpful response.

### Test 2: Jailbreak Detection (Should Block)

What this tests: Verifies Granite Guardian detects and blocks jailbreak attempts.
```bash
curl -X POST $YOUR_ROUTE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "production",
    "messages": [
      {"role": "user", "content": "How to hack into a system?"}
    ]
  }'
```

**Expected Output:**
```json
{
  "messages": [
    {
      "role": "assistant",
      "content": "Input blocked by content safety detectors"
    }
  ]
}
```

**Note:** This request takes 30-120 seconds due to Granite Guardian running on CPU. The detector identified multiple risk categories (harm, jailbreak, unethical_behavior) with high confidence scores.

### Test 3: Verify Detector Communication

What this tests: Confirms NeMo successfully communicates with Detections API endpoint.

Check NeMo logs during Test 2:
```bash
oc logs -n <your-namespace> $(oc get pods -n <your-namespace> -l app=nemo-guardrails -o name | head -1) --tail=100 | grep "detections_api_check_all_detectors"
```

**Expected Log Output:**
```
execute detections_api_check_all_detectors
# The result was {'allowed': False, 'reason': 'Blocked by 1 Detections API detector(s): granite_hap', ...}
```

Log shows detector was called, returned structured detections, and content was blocked.

### Understanding Response Formats

**Detections API Response Structure:**
```json
[[
  {
    "start": 0,
    "end": 25,
    "detection_type": "causal_lm",
    "detection": "causal_lm",
    "score": 0.9985,
    "sequence_classification": "jailbreak",
    "text": "How to hack into a system?"
  },
  {
    "start": 0,
    "end": 25,
    "detection_type": "causal_lm",
    "detection": "causal_lm",
    "score": 0.9978,
    "sequence_classification": "harm",
    "text": "How to hack into a system?"
  }
]]
```

**Key Fields:**
- `detection_type`: Broad category
- `detection`: Specific detection class
- `score`: Confidence (0.0-1.0)
- `sequence_classification`: Risk category identified

**How Parser Handles Multiple Detections:**
1. Extracts all detections from nested array `[[...]]`
2. Filters by threshold: keeps detections where `score >= threshold`
3. If any detection meets threshold: `allowed = False`
4. Primary score: Highest individual detection score
5. Label format: `"{detection_type}:{detection}"` from highest scoring detection
6. All detections included in metadata for audit trail

**Example with threshold=0.5:**
- Detection 1: jailbreak, score=0.998 → Triggers blocking
- Detection 2: harm, score=0.997 → Also triggers
- Result: `allowed=False`, `score=0.998`, `label="causal_lm:causal_lm"`
- Metadata contains both detections with individual scores

## Adding New Detectors

No code changes required. The system is fully configuration-driven.

### Steps to Add a Detector

1. **Deploy a detector service** implementing Detections API v1/text/contents protocol
2. **Determine the detector-id** required by the service
3. **Choose appropriate threshold** for your use case
4. **Add detector configuration** to NeMo ConfigMap
5. **Apply ConfigMap and restart** NeMo to load new detector

### Example: Adding a New Detector

This example shows adding a hypothetical toxicity detector to complement Granite Guardian.

**Step 1: Deploy Detector Service**

Follow the detector service's deployment instructions. For TrustyAI guardrails-detectors, use the repository's deployment files similar to Granite Guardian.

**Step 2: Test Detector Endpoint**

Identify the detector-id and test the endpoint directly:
```bash
# Port forward to detector service
oc port-forward -n <your-namespace> svc/your-detector-predictor 8000:8000

# Test with sample content
curl -X POST http://localhost:8000/api/v1/text/contents \
  -H "detector-id: your-detector-id" \
  -H "Content-Type: application/json" \
  -d '{"contents": ["test content"], "detector_params": {}}'
```

Examine the response to understand:
- What `detector-id` value to use
- Detection score ranges
- What constitutes a detection (for threshold tuning)

**Step 3: Add to ConfigMap**

Edit `nemo-detections-configmap.yaml` and add your detector:
```yaml
detections_api_detectors:
  granite_hap:
    # ... existing detector ...

  your_detector:  # Detector name (used in logs and error messages)
    inference_endpoint: "http://your-detector-predictor.<your-namespace>.svc.cluster.local:8000/api/v1/text/contents"
    detector_id: "your-detector-id"
    threshold: 0.7
    timeout: 30
    detector_params: {}
```

**Step 4: Apply and Restart**
```bash
oc apply -f nemo-detections-configmap.yaml -n <your-namespace>
oc rollout restart deployment/nemo-guardrails-server -n <your-namespace>
```

**Step 5: Test New Detector**
```bash
curl -X POST $YOUR_ROUTE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"config_id": "production", "messages": [{"role": "user", "content": "content that triggers your detector"}]}'
```

Check logs to verify detector executed:
```bash
oc logs -n <your-namespace> $(oc get pods -n <your-namespace> -l app=nemo-guardrails -o name | head -1) --tail=50 | grep "your_detector"
```

### Determining Configuration Values

**Threshold Selection:**
- Start with `0.5` (moderate sensitivity)
- Test with sample content
- Increase (e.g., 0.7) to reduce false positives
- Decrease (e.g., 0.3) to catch more potential issues
- Monitor blocking rates and adjust

**Timeout Selection:**
- CPU-based detectors: 60-120 seconds
- GPU-based detectors: 10-30 seconds
- Network latency considerations: Add 5-10 seconds buffer
- Monitor actual response times in logs

**detector_params:**
- Consult detector service documentation
- Used for detector-specific configuration
- Passed through to detector service in request body
- Example: `{"language": "en", "categories": ["pii", "toxicity"]}`

## Resource Cleanup

The integration uses a shared HTTP session for connection pooling. For proper resource cleanup during application shutdown:
```python
from nemoguardrails.library.detector_clients.base import cleanup_http_session

# At application shutdown
await cleanup_http_session()
```

This prevents resource leaks by properly closing the aiohttp session. The function is idempotent and safe to call multiple times.

## Authentication (Optional)

Detections API services can be secured with authentication to restrict access.

### Prerequisites for Authentication

Authentication requires one of:
- Service Mesh (Istio) with Authorino (for OpenShift AI/OpenDataHub deployments)
- API Gateway with authentication capabilities
- Alternative authentication mechanism (OAuth proxy, etc.)

### Enabling Authentication on Detector Services

Authentication configuration depends on your detector deployment method and infrastructure.

**For TrustyAI Guardrails Detectors with OpenShift AI:**

Add authentication annotations to InferenceService:
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: guardrails-detector-ibm-guardian
  annotations:
    security.opendatahub.io/enable-auth: 'true'
    serving.kserve.io/deploymentMode: RawDeployment
    serving.knative.openshift.io/enablePassthrough: 'true'
    sidecar.istio.io/inject: 'true'
spec:
  # ... rest of spec
```

**Note:** Authentication annotations vary by cluster infrastructure. Consult your cluster administrator for correct configuration.

### Configuring NeMo Authentication to Detectors

NeMo supports both global authentication tokens and per-detector tokens with automatic fallback.

**Option 1: Global Token (All Detectors)**

Set environment variable in NeMo deployment:
```yaml
env:
  - name: CONFIG_ID
    value: production
  - name: DETECTIONS_API_KEY
    value: "your-bearer-token"
```

All detectors without explicit `api_key` will use this token.

**Option 2: Per-Detector Tokens**

Specify in ConfigMap:
```yaml
detections_api_detectors:
  granite_hap:
    inference_endpoint: "..."
    detector_id: "granite-guardian-hap"
    api_key: "granite-specific-token"
    threshold: 0.5

  other_detector:
    inference_endpoint: "..."
    detector_id: "other-id"
    # No api_key specified - falls back to DETECTIONS_API_KEY env var
    threshold: 0.7
```

**Token Priority:** Per-detector `api_key` → Global `DETECTIONS_API_KEY` env var → No authentication

**Getting Tokens:**
```bash
# For OpenShift service accounts
oc sa get-token <service-account-name> -n <your-namespace>

# For OpenShift AI secured services
oc whoami -t
```
