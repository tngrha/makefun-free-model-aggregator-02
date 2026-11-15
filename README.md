# Live Free Model Aggregator

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API, in JSON and in streaming, and answers every prompt with a real
completion from a live free OpenRouter model dialled through the grader's
gateway.

## Run

```sh
python3 main.py
```

The server listens on `0.0.0.0` at a free port it picks, registers that port
and its catalog with the env at `$MAKEFUN_ENV_URL`, and exits 0 when
`GET $MAKEFUN_ENV_URL/work` says the workload is done. `python3 main.py
--list` prints the catalog document and exits 0.

## Routing

For each request, the free models that meet its requirements (tools, vision,
JSON mode, context) are ranked by the brief's quality rating, best first,
then by how little they have been used so far. A 429 or 5xx is retried once
on the same model, then the request fails over to the next candidate; a 401
means the entrant's OpenRouter key is missing or refused, which no other
model can fix, so that prompt is given up at once. Recovered refusals are
reported in the `x-makefun-upstream-failures` header. A model that refuses
twice in a row is skipped while any other candidate exists. At most 48 model
calls are made in a trial.

## Wires

Both client wires are served; every free model is dialled on the gateway's
Chat Completions route (`POST $OPENAI_BASE_URL/chat/completions`) with the
trial's dummy key. An Anthropic-shaped request is translated into a Chat
Completions body first, and the gateway's answer is translated back. The
`model` field on every answer and every streamed chunk names the id that was
really dialled (`openrouter/<upstream id>:free`).

Everything is a deterministic function of the brief, the requests and the
gateway's answers: no wall clock, no unseeded randomness, so a replay
leaves the same state export.

## Layout

- `main.py` — the whole server (Python 3.13, standard library only)
- `LICENSE` — MIT
