# KServe Detector Integration for NeMo Guardrails
## Overview

This integration enables NeMo Guardrails to work with any KServe-hosted HuggingFace detection model through pure configuration, without code changes or container rebuilds.

**Key Features:**
- **Configuration-driven**: Add/remove detectors via ConfigMap updates only
- **Format-agnostic**: Handles probability distributions, integer arrays, named labels, and entity dicts
- **Flexible detection logic**: Configurable `safe_labels` approach works with any model semantics
- **Parallel execution**: All detectors run simultaneously for low latency

## Architecture
    User Input → NeMo Guardrails → [Detectors in Parallel] → vLLM (if safe) → Response

**Components:**
- **NeMo Guardrails** (CPU) - Orchestration and flow control
- **KServe Detectors** (CPU) - Toxicity, jailbreak, PII, HAP detection
- **vLLM** (GPU) - LLM inference with Phi-3-mini

## Prerequisites

- OpenShift cluster with KServe installed
- GPU node pool (for vLLM)
- Access to Quay.io or ability to mirror images
## Changes Made

### Files Added

**`nemoguardrails/library/kserve_detector/actions.py`**

Core detector integration actions:

- `_parse_safe_labels_env()` - Parse SAFE_LABELS environment variable with fallback to [0]
- `parse_kserve_response()` - Generic parser that handles any detector response format (probability distributions, integer arrays, named labels, entity dicts, booleans)
- `parse_kserve_response_detailed()` - Wraps parse result with metadata (detector name, risk type, reason)
- `_call_kserve_endpoint()` - HTTP client for KServe inference endpoints with timeout and auth support
- `_run_detector()` - Execute single detector with error handling and safe_labels merging
- `kserve_check_all_detectors()` - Run all configured detectors in parallel and aggregate results
- `generate_block_message()` - Generate user-friendly blocking messages with detector details
- `kserve_check_detector()` - Run specific detector by type from registry
- `kserve_check_input()` - Check user input with specified detector
- `kserve_check_output()` - Check bot output with specified detector

### Files Modified

**`nemoguardrails/rails/llm/config.py`**

Added configuration classes:

- `KServeDetectorConfig` - Configuration for single KServe detector
  - `inference_endpoint` (str) - KServe API endpoint URL
  - `model_name` (Optional[str]) - HuggingFace model identifier
  - `threshold` (float) - Probability threshold for detection (default: 0.5)
  - `timeout` (int) - HTTP request timeout in seconds (default: 30)
  - `risk_type` (Optional[str]) - Risk classification type (defaults to detector key name)
  - `safe_labels` (List[Union[int, str]]) - Class indices or label names considered safe (default: [0])

Modified `RailsConfigData` class:
- Added `kserve_detectors` (Dict[str, KServeDetectorConfig]) - Dynamic registry of KServe detectors, keys are detector names

**Key changes from initial version:**
- Removed `detector_type` field (now uses dictionary key)
- Removed `invert_logic` field (replaced by safe_labels approach)
- Added `safe_labels` field for flexible detection logic
- Retained `risk_type` field as optional for critical functionality:
  - Distinguishes system errors (`risk_type: "system_error"`) from content violations (e.g., `"hate_speech"`, `"privacy_violation"`)
  - Enables semantic separation between technical detector names and business risk classifications
  - Allows multiple detectors to map to the same risk category for reporting and analytics
  - Provides flexibility to swap detector implementations without changing risk taxonomy
  - Defaults to detector key name if not specified
- Removed `kserve_detector` single detector field (backward compatibility no longer needed)

## How It Works

### Detection Flow

1. User sends message to NeMo Guardrails via HTTP POST request
2. NeMo loads configuration from ConfigMap and triggers `check_input_safety` flow defined in `rails.co`
3. All configured detectors execute in parallel via `kserve_check_all_detectors()` action
4. Each detector:
   - Receives the user message via HTTP POST to its KServe endpoint
   - Processes with its model (toxicity, jailbreak, PII, HAP, etc.)
   - Returns prediction in its native format
5. Generic parser processes each response:
   - Automatically detects response format (probability distributions, integer arrays, named labels, entity dicts)
   - Extracts predicted class and confidence score
   - Compares predicted class against configured `safe_labels`
   - Returns safety decision with metadata (allowed/blocked, score, risk_type)
6. Results aggregation:
   - If ANY detector unavailable: Request blocked with system error message
   - If ANY detector blocks content: Request blocked with detailed message showing blocking detector(s)
   - If ALL detectors approve: Request proceeds to vLLM for generation
7. Response generation (if allowed) by vLLM and returned to user

### Safe Labels Logic

The `safe_labels` approach provides flexible detection logic that works with any model's labeling convention, replacing hardcoded assumptions about which classes represent safe content.

**Detection process:**
1. Detector returns predicted class (integer ID, string label, or probability distribution)
2. Parser identifies the class with highest confidence
3. Check: Is predicted class in `safe_labels`?
   - YES: Content is safe for this detector
   - NO: Check if confidence >= threshold
     - YES: Flag as unsafe, block
     - NO: Low confidence, treat as safe
4. For token classification: Calculate ratio of flagged tokens and compare against threshold

### Error Handling

The system distinguishes between infrastructure errors and content violations to provide appropriate feedback and enable proper monitoring.

**System Errors:**

Infrastructure issues such as network timeouts, connection failures, or parse errors are handled separately:
- Marked with `risk_type: "system_error"`
- Score set to 0.0 (indicates not a detection score)
- Tracked in `unavailable_detectors` list
- User receives service unavailability message
- Request is blocked (fail-safe behavior) but clearly communicates infrastructure issue rather than content violation

**Content Violations:**

Actual detections by models:
- `risk_type`: Detector's configured risk type (e.g., hate_speech, privacy_violation, prompt_injection)
- Score: Model's confidence score (0.0-1.0)
- Tracked in `blocking_detectors` list
- User receives detailed blocking message with detector name, risk type, and confidence score

**Multiple Detectors:**

When multiple detectors flag content simultaneously, all blocking detectors are reported in the response message, enabling full visibility into which safety checks triggered.

This separation ensures users receive appropriate feedback (service issue vs content issue) and operators can distinguish between content problems and infrastructure failures in logs and monitoring systems.

## Deployment Guide

### Prerequisites

- OpenShift cluster with KServe installed
- Namespace: `kserve-hfdetector` (or your preferred namespace)
- GPU node pool with g4dn.2xlarge or similar instances (for vLLM)
- Access to Quay.io or container registry for pulling images

### Step 1: Deploy HuggingFace ServingRuntime

Create `huggingface-runtime.yaml`:
```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: ServingRuntime
metadata:
  name: kserve-huggingfaceruntimev1
  namespace: kserve-hfdetector
spec:
  supportedModelFormats:
    - name: huggingface
      version: "1"
      autoSelect: true
  containers:
    - name: kserve-container
      image: quay.io/rh-ee-stondapu/huggingfaceserver:v0.14.0
      args:
        - --model_name={{.Name}}
        - --model_id=$(MODEL_NAME)
      env:
        - name: HF_TASK
          value: "$(HF_TASK)"
        - name: MODEL_NAME
          value: "$(MODEL_NAME)"
        - name: TRANSFORMERS_CACHE
          value: "/tmp/transformers_cache"
        - name: HF_HUB_CACHE
          value: "/tmp/hf_c
      resources:
        requests:
          cpu: "1"
          memory: "2Gi"
        limits:
          cpu: "2"
          memory: "4Gi"
      ports:
        - containerPort: 8080
          protocol: TCP
  protocolVersions:
    - v1
    - v2
```

### Step 2: Deploy Detection Models

Deploy each detector InferenceService. All detectors use the HuggingFace ServingRuntime created in Step 1.

#### Toxicity Detector

**File:** `toxicity-detector.yml`
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: toxicity-detector
  namespace: kserve-hfdetector
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 2
    model:
      modelFormat:
        name: huggingface
      image: kserve/huggingfaceserver:v0.13.0
      env:
        - name: MODEL_NAME
          value: "martin-ha/toxic-comment-model"
        - name: HF_TASK
          value: "text-classification"
    nodeSelector:
      node.kubernetes.io/instance-type: m5.2xlarge
    resources:
      requests:
        cpu: "500m"
        memory: "2Gi"
      limits:
        cpu: "1"
        memory: "4Gi"
```
#### Jailbreak Detector

**File:** `jailbreak-detector.yml`
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: jailbreak-detector
  namespace: kserve-hfdetector
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 1
    model:
      modelFormat:
        name: huggingface
      image: quay.io/rh-ee-stondapu/huggingfaceserver:v0.14.0
      env:
        - name: MODEL_NAME
          value: "jackhhao/jailbreak-classifier"
        - name: HF_TASK
          value: "text-classification"
    nodeSelector:
      node.kubernetes.io/instance-type: m5.2xlarge
    resources:
      requests:
        cpu: "500m"
        memory: "2Gi"
      limits:
        cpu: "1"
        memory: "4Gi"
```
#### PII Detector

**File:** `pii-detector.yml`
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: pii-detector
  namespace: kserve-hfdetector
spec:
  predictor:
    model:
      modelFormat:
        name: huggingface
      image: quay.io/rh-ee-stondapu/huggingfaceserver:v0.14.0
      args:
        - --model_name=pii-detector
        - --model_id=iiiorg/piiranha-v1-detect-personal-information
        - --task=token_classification
        - --backend=huggingface
        - --dtype=float32
      resources:
        requests:
          cpu: "2"
          memory: "4Gi"
        limits:
          cpu: "4"
          memory: "8Gi"
```
**File:** `hap-detector.yml`
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: hap-detector
  namespace: kserve-hfdetector
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 2
    model:
      modelFormat:
        name: huggingface
      image: quay.io/rh-ee-stondapu/huggingfaceserver:v0.14.0
      args:
        - --model_name=hap-detector
        - --model_id=ibm-granite/granite-guardian-hap-38m
        - --task=sequence_classification
        - --backend=huggingface
        - --dtype=float32
      resources:
        requests:
          cpu: "1"
          memory: "2Gi"
        limits:
          cpu: "2"
          memory: "4Gi"
```
Deploy all detectors:
```bash
oc apply -f toxicity-detector.yml -n kserve-hfdetector
oc apply -f jailbreak-detector.yml -n kserve-hfdetector
oc apply -f pii-detector.yml -n kserve-hfdetector
oc apply -f hap-detector.yml -n kserve-hfdetector
```
Verify all detectors are ready:
```bash 
oc get inferenceservice -n kserve-hfdetector
```
Expected output showing all with READY = True:
NAME                 READY
toxicity-detector    True
jailbreak-detector   True
pii-detector         True
hap-detector         True

This may take 2-5 minutes as models download from HuggingFace.

### Step 3: Deploy vLLM Inference Service

vLLM uses a PVC-based approach to pre-download the Phi-3-mini model. This avoids runtime dependencies on HuggingFace and uses Red Hat's official AI Inference Server image.

Create `vllm-inferenceservice.yml`:
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: phi3-model-pvc
  namespace: kserve-hfdetector
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
  namespace: kserve-hfdetector
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
  namespace: kserve-hfdetector
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
    nodeSelector:
      node.kubernetes.io/instance-type: g4dn.2xlarge
```
Deploy:

```bash
oc apply -f vllm-inferenceservice.yml -n kserve-hfdetector
```

Monitor model download progress:

```bash
oc logs -n kserve-hfdetector -l app=phi3-downloader -c download-model -f
```

Wait for "Download complete!" message. The Phi-3-mini model is approximately 8GB and may take 3-5 minutes to download.
Verify vLLM is running:

```bash
oc get inferenceservice vllm-phi3 -n kserve-hfdetector
oc get pods -n kserve-hfdetector | grep vllm-phi3
```

Expected: `vllm-phi3` InferenceService shows `READY = True` and pod shows `1/1 Running`.

### Step 4: Deploy NeMo Guardrails ConfigMap

The ConfigMap contains the detector registry configuration and flow definitions.

Create `nemo-configmap.yml`:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nemo-production-config
  namespace: kserve-hfdetector
data:
  config.yaml: |
    rails:
      config:
        kserve_detectors:
          toxicity:
            inference_endpoint: "http://toxicity-detector-predictor.kserve-hfdetector.svc.cluster.local:8080/v1/models/toxicity-detector:predict"
            model_name: "ibm-granite/granite-guardian-hap-38m"
            threshold: 0.4
            timeout: 30
            safe_labels: [0]
            risk_type: "hate_speech"
          jailbreak:
            inference_endpoint: "http://jailbreak-detector-predictor.kserve-hfdetector.svc.cluster.local:8080/v1/models/jailbreak-detector:predict"
            model_name: "jackhhao/jailbreak-classifier"
            threshold: 0.5
            timeout: 30
            safe_labels: [0]
            risk_type: "prompt_injection"
          pii:
            inference_endpoint: "http://pii-detector-predictor.kserve-hfdetector.svc.cluster.local:8080/v1/models/pii-detector:predict"
            model_name: "iiiorg/piiranha-v1-detect-personal-information"
            threshold: 0.15
            timeout: 30
            safe_labels: [17]
            risk_type: "privacy_violation"
          hap:
            inference_endpoint: "http://hap-detector-predictor.kserve-hfdetector.svc.cluster.local:8080/v1/models/hap-detector:predict"
            model_name: "ibm-granite/granite-guardian-hap-38m"
            threshold: 0.5
            timeout: 30
            safe_labels: [0]
            risk_type: "hate_abuse_profanity"
      input:
        flows:
          - check_input_safety
    models:
      - type: main
        engine: vllm_openai
        model: phi3-mini
        parameters:
          openai_api_base: http://vllm-phi3-predictor.kserve-hfdetector.svc.cluster.local:8080/v1
          openai_api_key: sk-dummy-key
    instructions:
      - type: general
        content: |
          You are a helpful AI assistant.
  rails.co: |
   define flow check_input_safety
      $input_result = execute kserve_check_all_detectors
      
      if $input_result.unavailable_detectors
          $msg = execute generate_block_message
          bot refuse with message $msg
          stop
      
      if not $input_result.allowed
          $msg = execute generate_block_message
          bot refuse with message $msg
          stop

    define bot refuse with message $msg
        $msg
```
Important: 
Ensure each detector in kserve_detectors has the safe_labels field configured appropriately:

Toxicity/Jailbreak/HAP: safe_labels: [0] (class 0 = safe)

PII: safe_labels: [17] (class 17 = background/no PII)

Adjust based on your detector model's output classes

Deploy:

```bash
oc apply -f nemo-configmap.yml -n kserve-hfdetector
```

Verify:

```bash
oc get configmap nemo-production-config -n kserve-hfdetector
```
### Step 5: Deploy NeMo Guardrails Server

Create `nemo-deployment.yml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nemo-guardrails-server
  namespace: kserve-hfdetector
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
      nodeSelector:
        node.kubernetes.io/instance-type: m5.2xlarge
      containers:
      - name: nemo-guardrails
        image: quay.io/rh-ee-stondapu/trustyai-nemo:latest
        imagePullPolicy: Always
        env:
        - name: CONFIG_ID
          value: production
        - name: OPENAI_API_KEY
          value: sk-dummy-key-for-vllm
        - name: SAFE_LABELS
          value: "[0]"
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
          name: nemo-production-config
---
apiVersion: v1
kind: Service
metadata:
  name: nemo-guardrails-server
  namespace: kserve-hfdetector
spec:
  selector:
    app: nemo-guardrails
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
```
Deploy:
```bash
oc apply -f nemo-deployment.yml -n kserve-hfdetector
```
Expose service externally:
```bash 
oc expose service nemo-guardrails-server -n kserve-hfdetector
```
Get the external route URL:
```bash 
YOUR_ROUTE="http://$(oc get route nemo-guardrails-server -n kserve-hfdetector -o jsonpath='{.spec.host}')"

echo "NeMo Guardrails URL: $YOUR_ROUTE"
```
Verify all components are running:
```bash 
oc get pods -n kserve-hfdetector
```
Expected pods (all with status Running):

    nemo-guardrails-server-* (1/1)
    toxicity-detector-predictor-* (1/1)
    jailbreak-detector-predictor-* (1/1)
    pii-detector-predictor-* (1/1)
    hap-detector-predictor-* (1/1)
    vllm-phi3-predictor-* (1/1)
    phi3-model-downloader-* (1/1)


## Testing

Use the route URL from Step 5:
```bash
# If you haven't set it yet:
YOUR_ROUTE="http://$(oc get route nemo-guardrails-server -n kserve-hfdetector -o jsonpath='{.spec.host}')"
```

Test 1: Safe Content

```bash
curl -X POST $YOUR_ROUTE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"config_id": "production", "messages": [{"role": "user", "content": "What is 2+2?"}]}'
```
Expected: LLM responds with the answer.

Test 2: Toxic Content Detection

```bash
curl -X POST $YOUR_ROUTE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"config_id": "production", "messages": [{"role": "user", "content": "I hate you and want to kill you"}]}'
```
Expected: Blocked with message showing blocking detectors.

Test 3: PII Detection

```bash
curl -X POST $YOUR_ROUTE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"config_id": "production", "messages": [{"role": "user", "content": "My SSN is 123-45-6789"}]}'
```
Expected: Blocked by PII detector with confidence score.

Test 4: Multiple Detectors
```bash
curl -X POST $YOUR_ROUTE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"config_id": "production", "messages": [{"role": "user", "content": "You idiot, my SSN is 123-45-6789"}]}'
```
Expected: Blocked by multiple detectors (toxicity, pii, hap).

## Adding New Detectors

No code changes required to add new detectors. The system is fully configuration-driven.

### Steps to Add a Detector

1. **Deploy your detector as a KServe InferenceService** using the HuggingFace ServingRuntime
2. **Determine the safe_labels** for your model by testing its output format
3. **Add detector configuration** to the NeMo ConfigMap under `kserve_detectors`
4. **Restart NeMo Guardrails** to load the new configuration

### Example: Adding a New Detector

**Step 1:** Deploy your detector InferenceService (similar to toxicity-detector.yml)

**Step 2:** Test the detector to identify safe classes:
```bash
oc exec -n kserve-hfdetector <nemo-pod-name> -- curl -X POST \
  http://your-detector-predictor.kserve-hfdetector.svc.cluster.local:8080/v1/models/your-detector:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": ["test content"]}'
```

Examine the output to determine which class IDs or labels represent safe content.

Step 3: Add to ConfigMap under `kserve_detectors`:
```yaml
kserve_detectors:
  toxicity:
    # existing detector configs...
  your_new_detector:
    inference_endpoint: "http://your-detector-predictor.kserve-hfdetector.svc.cluster.local:8080/v1/models/your-detector:predict"
    model_name: "your/huggingface-model-id"
    threshold: 0.5
    timeout: 30
    safe_labels: [0]  # Adjust based on your model's output
    risk_type: "your_risk_category"
```

Step 4: Apply updated ConfigMap and restart:

```bash
oc apply -f nemo-configmap.yml -n kserve-hfdetector
oc rollout restart deployment/nemo-guardrails-server -n kserve-hfdetector
```

Step 5: Test the new detector:

```bash
curl -X POST $YOUR_ROUTE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"config_id": "production", "messages": [{"role": "user", "content": "test input for your detector"}]}'
```

Determining Safe Labels
For binary classifiers: Test with known safe and unsafe content to see which class (0 or 1) represents safe.
For multi-class: Examine model documentation or test outputs to identify background/safe class indices.
For token classification: Identify which class represents background/no-detection (often 0 or the highest class number).