# KServe Detector Integration for NeMo Guardrails

## Overview

This integration enables NeMo Guardrails to work with any KServe-hosted detection model through pure configuration. It supports multiple detector response formats (binary classification, sequence classification, token classification) and allows adding or removing detectors via ConfigMap updates without code changes or container rebuilds. The implementation has been production-validated with toxicity detection, jailbreak detection, and PII detection running in parallel on OpenShift.

## Changes Made

### Files Added

**`nemoguardrails/library/kserve_detector/actions.py`**
- Generic KServe detector integration actions
- `kserve_check_all_detectors()` - Runs all configured detectors in parallel
- `kserve_check_detector()` - Runs specific detector by type
- `kserve_check_input()` - Generic input validation
- `kserve_check_output()` - Generic output validation
- `parse_kserve_response()` - Handles any detector response format

### Files Modified

**`nemoguardrails/rails/llm/config.py`**
- Added `KServeDetectorConfig` class with fields:
  - `inference_endpoint`: KServe API endpoint URL
  - `model_name`: HuggingFace model identifier
  - `threshold`: Detection threshold (0.0-1.0)
  - `timeout`: HTTP timeout in seconds
  - `detector_type`: Detector identifier
  - `risk_type`: Risk classification type
  - `invert_logic`: Score inversion for reversed semantics
- Added `kserve_detectors` field to `RailsConfigData` class for dynamic detector registry
- Added `kserve_detector` field for backward compatibility

## How It Works

The integration uses a dynamic detector registry that automatically discovers all configured detectors from the ConfigMap at runtime. When a user input is received:

1. NeMo Guardrails extracts the user message
2. All detectors in `kserve_detectors` are called in parallel via async HTTP requests
3. Each detector returns a response in its native format
4. The generic parser automatically handles the response format
5. If any detector flags the content as unsafe (score >= threshold), the input is blocked
6. If all detectors approve, the request proceeds to the LLM for response generation

**Supported Response Formats:**
- Sequence classification (binary): `{"predictions": [0]}` or `{"predictions": [1]}` - Used by toxicity and jailbreak detectors
- Token classification (integer arrays): `{"predictions": [[[17,17,10,10,17]]]}` - Used by PII detector, where non-background labels indicate detected entities

The parser automatically identifies background labels (typically 0 or the highest value) and counts non-background tokens as detections.


## Configuration

### NeMo Guardrails ConfigMap

**File:** `nemo-configmap.yml`
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
            threshold: 0.5
            timeout: 30
            detector_type: "toxicity"
            risk_type: "hate_speech"
          jailbreak:
            inference_endpoint: "http://jailbreak-detector-predictor.kserve-hfdetector.svc.cluster.local:8080/v1/models/jailbreak-detector:predict"
            model_name: "jackhhao/jailbreak-classifier"
            threshold: 0.5
            timeout: 30
            detector_type: "jailbreak"
            risk_type: "prompt_injection"
          pii:
            inference_endpoint: "http://pii-detector-predictor.kserve-hfdetector.svc.cluster.local:8080/v1/models/pii-detector:predict"
            model_name: "iiiorg/piiranha-v1-detect-personal-information"
            threshold: 0.5
            timeout: 30
            detector_type: "pii"
            risk_type: "privacy_violation"
      input:
        flows:
          - check_input_safety
    models:
      - type: main
        engine: vllm_openai
        model: phi3-mini
        parameters:
          openai_api_base: http://vllm-server.kserve-hfdetector.svc.cluster.local:8000/v1
          openai_api_key: sk-dummy-key
    instructions:
      - type: general
        content: |
          You are a helpful AI assistant. Respond naturally and helpfully to user questions.
  rails.co: |
    define flow check_input_safety
        $input_result = execute kserve_check_all_detectors
        
        if not $input_result.allowed
            bot refuse input
            stop
    
    define bot refuse input $input_result
      "Input blocked. Detector: {$input_result.blocking_detectors[0].detector}, Risk: {$input_result.blocking_detectors[0].risk_type}, Score: {$input_result.blocking_detectors[0].score:.3f}"

```
### Detector Deployments
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
      image: kserve/huggingfaceserver:v0.13.0
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
### NeMo Server Deployment

**File:** `nemo-deployment.yml`
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
        env:
        - name: CONFIG_ID
          value: production
        - name: OPENAI_API_KEY
          value: sk-dummy-key-for-vllm
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
```
### vLLM Deployment (LLM Inference)

**File:** `vllm-phi3-gpu.yaml`
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
  namespace: kserve-hfdetector
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-server
  template:
    metadata:
      labels:
        app: vllm-server
    spec:
      nodeSelector:
        node.kubernetes.io/instance-type: g4dn.2xlarge
      containers:
      - name: vllm
        image: vllm/vllm-openai:v0.4.2
        args:
        - --model=microsoft/Phi-3-mini-4k-instruct
        - --host=0.0.0.0
        - --port=8000
        - --served-model-name=phi3-mini
        - --max-model-len=4096
        - --gpu-memory-utilization=0.7
        - --trust-remote-code
        - --dtype=half
        ports:
        - containerPort: 8000
        env:
        - name: HF_HOME
          value: /tmp/hf_cache
        - name: NUMBA_CACHE_DIR
          value: /tmp/numba_cache
        volumeMounts:
        - name: cache-volume
          mountPath: /tmp
        resources:
          requests:
            nvidia.com/gpu: 1
            cpu: "2"
            memory: "8Gi"
          limits:
            nvidia.com/gpu: 1
            cpu: "6"
            memory: "24Gi"
      volumes:
      - name: cache-volume
        emptyDir:
          sizeLimit: 20Gi
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-server
  namespace: kserve-hfdetector
spec:
  selector:
    app: vllm-server
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
```
## Deployment Steps

### Step 1: Deploy Detection Models

Deploy the three KServe detectors:

```bash
oc apply -f toxicity-detector.yaml
oc apply -f jailbreak-detector.yaml
oc apply -f pii-detector.yaml
```

Wait for all detectors to be ready:

```bash
oc get inferenceservice -n kserve-hfdetector
```

All three should show READY = True before proceeding.

### Step 2: Deploy vLLM Server

```bash
oc apply -f vllm-phi3-gpu.yaml
```

Verify vLLM pod is running:

```bash
oc get pods -n kserve-hfdetector -l app=vllm-server
```

### Step 3: Deploy NeMo Guardrails ConfigMap

```bash
oc apply -f nemo-configmap.yaml
```

### Step 4: Deploy NeMo Guardrails Server

```bash
oc apply -f nemo-deployment.yaml
```

Expose the service externally:

```bash
oc expose service nemo-guardrails-server -n kserve-hfdetector
```

Get the external route:

```bash
oc get route nemo-guardrails-server -n kserve-hfdetector
```

### Step 5: Verify All Components

Check all pods are running:

```bash
oc get pods -n kserve-hfdetector
```

Expected output should show all pods in Running state:
- nemo-guardrails-server
- toxicity-detector-predictor
- jailbreak-detector-predictor
- pii-detector-predictor
- vllm-server

## Testing

Replace YOUR_ROUTE with your NeMo Guardrails route URL.

### Test 1: Safe Content (Should Pass)

```bash
curl -X POST http://YOUR_ROUTE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"config_id": "production", "messages": [{"role": "user", "content": "What is 2+2?"}]}'
```

Expected: Normal LLM response with answer

### Test 2: Toxicity Detection (Should Block)

```bash
curl -X POST http://YOUR_ROUTE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"config_id": "production", "messages": [{"role": "user", "content": "damn you"}]}'
```

Expected: Blocked with detector details

### Test 3: PII Detection (Should Block)

```bash
curl -X POST http://YOUR_ROUTE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"config_id": "production", "messages": [{"role": "user", "content": "My SSN is 123-45-6789"}]}'
```

Expected: Blocked with detector details

## Adding New Detectors

No code changes required. Simply:

1. Deploy your KServe InferenceService
2. Add detector configuration to the ConfigMap under kserve_detectors
3. Restart NeMo Guardrails

Example - adding a new detector:

```yaml
kserve_detectors:
  toxicity:
    # existing detector config
  your_new_detector:
    inference_endpoint: "http://your-detector-predictor.namespace.svc.cluster.local:8080/v1/models/your-detector:predict"
    model_name: "your/huggingface-model"
    threshold: 0.5
    timeout: 30
    detector_type: "your_detector"
    risk_type: "your_risk_type"
```

Then restart:

```bash
oc rollout restart deployment/nemo-guardrails-server -n kserve-hfdetector
```