# Integrating geolm into services

A trained model is a single `*.geolm` file. Deployment means copying that file to
the target machine and choosing one of three integration paths. Russian version:
[INTEGRATION.ru.md](INTEGRATION.ru.md).

The contract to understand first: the model takes a string and returns a string. Any
data (numbers, form fields, JSON) is serialized to a string in one agreed format
before being passed in, exactly the way you agree on a format for any API. One model
is trained for one task; serving several input kinds is covered at the end.

## Path 1. HTTP service (any language, any stack)

```bash
geolm serve --model task.geolm --port 8080
```

Contract: `POST /predict` with a JSON body, two forms.

```json
request:  {"input": "7 марта 1961"}
response: {"output": "1961-03-07"}

request:  {"inputs": ["1+1=", "40+50="]}
response: {"outputs": ["2", "90"]}
```

Clients:

```bash
curl -X POST localhost:8080/predict -d '{"input": "7 марта 1961"}'
```

```python
import requests
r = requests.post("http://localhost:8080/predict", json={"input": "7 марта 1961"})
print(r.json()["output"])
```

```javascript
const r = await fetch("http://localhost:8080/predict", {
  method: "POST",
  body: JSON.stringify({ input: "7 марта 1961" }),
});
const { output } = await r.json();
```

```go
resp, _ := http.Post("http://localhost:8080/predict", "application/json",
    strings.NewReader(`{"input": "7 марта 1961"}`))
```

Minimal Dockerfile:

```dockerfile
FROM python:3.12-slim
COPY geolm /opt/geolm
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install /opt/geolm
COPY task.geolm /opt/task.geolm
EXPOSE 8080
CMD ["geolm", "serve", "--model", "/opt/task.geolm", "--port", "8080"]
```

## Path 2. Direct import (Python services)

No network, no separate process; the model lives inside your application:

```python
from geolm import GeoLM

model = GeoLM.load("task.geolm")          # once at service start

def handle(request_text: str) -> str:      # inside a request handler
    return model.predict(request_text)

answers = model.predict_batch(["1+1=", "40+50=", "7+8="])   # faster for streams
```

Fits FastAPI and Flask apps, queue workers, and bots.

## Path 3. Pipelines and scripts (stdin, line by line)

```bash
cat inputs.txt | geolm predict --model task.geolm > outputs.txt
```

Each input line becomes one answer line. Convenient for batch jobs and cron.

## Different kinds of input data

Rule: the model understands strings, so different data kinds are handled at the
string-format level. Three typical cases.

Case 1, structured fields: serialize them with separators and train on that format:

```json
{"input": "city=Moscow; street=Tverskaya; house=1", "output": "119019"}
```

Case 2, several input types with one model: prefix the task name in the training
data, and one model learns to route by prefix (multi-task training):

```json
{"input": "date: 7 марта 1961",  "output": "1961-03-07"}
{"input": "sum: 12+34",          "output": "46"}
{"input": "phone: 8 916 1234567", "output": "+79161234567"}
```

Case 3, many complex formats: keep several narrow models (one file per task) and
choose the model in code by request type. Files are small; a specialist per task is
a normal setup.

## Rules for reliable operation

Validate input before the model. The model will answer something for any input,
including garbage, so format validation (length, allowed characters) belongs in
front of it. Characters never seen in training are unknown to the model.

Validate output after the model. If the answer must be a date or a number, check it
with a regular expression and handle rare mismatches.

Warm the model up at service start with one dummy request so the first real request
is not slower than the rest.

Measure quality on held-out data before every rollout of a new model version
(`geolm eval --model new.geolm --data holdout.jsonl`) and do not ship if accuracy
dropped relative to the current version.

Scaling: one process serves thousands of queries per second on CPU; for more, run
several processes behind a load balancer. The model keeps no state between requests,
so replicas are independent.
