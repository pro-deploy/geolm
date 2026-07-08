"""Command line: geolm train / predict / eval / serve."""
import argparse
import json
import sys

from .data import read_jsonl
from .pipeline import GeoLM


def cmd_train(args):
    pairs = read_jsonl(args.data)
    print(f"[geolm] examples: {len(pairs)}")
    model = GeoLM.new(pairs, dim=args.dim, hidden=args.hidden, layers=args.layers,
                      max_new=args.max_new, device=args.device)
    n_par = sum(p.numel() for p in model.model.parameters())
    print(f"[geolm] parameters: {n_par/1e6:.2f}M, vocabulary: {len(model.vocab)}")
    best = model.fit(pairs, steps=args.steps, batch_size=args.batch, lr=args.lr)
    model.save(args.out)
    print(f"[geolm] done: best exact match {best:.1f}%, model saved to {args.out}")


def cmd_predict(args):
    model = GeoLM.load(args.model, device=args.device)
    if args.input is not None:
        print(model.predict(args.input, temperature=args.temperature))
        return
    for line in sys.stdin:                      # batch mode: one input per stdin line
        line = line.rstrip("\n")
        if line:
            print(model.predict(line, temperature=args.temperature))


def cmd_eval(args):
    model = GeoLM.load(args.model, device=args.device)
    pairs = read_jsonl(args.data)
    acc = model.evaluate(pairs)
    print(f"[geolm] exact matches: {acc:.1f}% on {len(pairs)} examples")


def cmd_serve(args):
    """HTTP service on the standard library: POST /predict {"input": "..."}."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    model = GeoLM.load(args.model, device=args.device)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/predict":
                self.send_error(404)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if "inputs" in body:
                    result = {"outputs": model.predict_batch([str(t) for t in body["inputs"]])}
                else:
                    result = {"output": model.predict(str(body["input"]))}
                payload = json.dumps(result, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as exc:  # noqa: BLE001 -- the service must not die on a bad request
                self.send_error(400, str(exc))

        def log_message(self, fmt, *a):
            pass

    print(f"[geolm] serving on http://0.0.0.0:{args.port}  (POST /predict)")
    HTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


def main():
    p = argparse.ArgumentParser(prog="geolm", description="Narrow geometric models: train for a task, ship to inference")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train a model on JSONL with input/output fields")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--dim", type=int, default=128)
    t.add_argument("--hidden", type=int, default=512)
    t.add_argument("--layers", type=int, default=2)
    t.add_argument("--max-new", type=int, default=64)
    t.add_argument("--device", default="auto")
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="get an answer (argument or stdin, one input per line)")
    pr.add_argument("--model", required=True)
    pr.add_argument("--input")
    pr.add_argument("--temperature", type=float, default=0.0)
    pr.add_argument("--device", default="auto")
    pr.set_defaults(func=cmd_predict)

    ev = sub.add_parser("eval", help="exact-match accuracy on JSONL with input/output fields")
    ev.add_argument("--model", required=True)
    ev.add_argument("--data", required=True)
    ev.add_argument("--device", default="auto")
    ev.set_defaults(func=cmd_eval)

    sv = sub.add_parser("serve", help="start an HTTP inference service")
    sv.add_argument("--model", required=True)
    sv.add_argument("--port", type=int, default=8080)
    sv.add_argument("--device", default="auto")
    sv.set_defaults(func=cmd_serve)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
