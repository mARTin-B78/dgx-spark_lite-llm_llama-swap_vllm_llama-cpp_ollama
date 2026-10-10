#!/usr/bin/env python3
"""Keep LiteLLM's OpenRouter free-model aliases/fallbacks alive (stdlib only).

Source of truth: LiteLLM_Config row param_name='router_settings' in litellm-postgres
(keys model_group_alias + fallbacks only; no secrets are read or written).

    refresh-openrouter-aliases.py             # dry-run: print diff + suggestions
    refresh-openrouter-aliases.py --apply     # backup DB, write the two keys (then restart/reload litellm)
    refresh-openrouter-aliases.py --selftest

Policy: an alias target / fallback entry is replaced ONLY if it is no longer free
(gone or paid). Better candidates are just reported (stdout + logs/openrouter-aliases.log).
Every OpenRouter chain ends with openrouter/openrouter/free (non-audio aliases only).
Cron (nightly, swaps dead models only):
    0 4 * * * /usr/bin/python3 <repo>/scripts/refresh-openrouter-aliases.py --apply >> <repo>/logs/openrouter-aliases.log 2>&1
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

MODELS_URL = "https://openrouter.ai/api/v1/models"
PREFIX = "openrouter/"                     # LiteLLM provider prefix
TAIL = "openrouter/openrouter/free"        # router verified to exist + route (see Part 1)
PREFERRED_VENDORS = ["nvidia", "qwen", "google", "poolside", "cohere", "thinkingmachines"]
# listed free but 403 for plain API calls ("only available on agentic harnesses") -> treated as dead
BLOCKED = {"thinkingmachines/inkling-small:free"}
# routed model groups without an alias that still need a fallback chain ending in TAIL
EXTRA_CHAINS = ["openrouter/google/gemma-4-31b-it:free"]
# alias-name regex -> required capability; first match wins. Fallback chains need "tools".
RULES = [
    (r"^tts", "audio_out"),
    (r"^(ocr|vision)$", "image_in"),
    (r"^long-context$", "ctx200k"),
    (r"^(coding|reasoning|default)", "tools"),
    (r".*", "text"),
]
CHECKS = {
    # ponytail: free audio-out models on OpenRouter are music generators (lyria), not TTS -> never auto-pick; warn only
    "audio_out": lambda m: False,
    "image_in": lambda m: "image" in m["architecture"]["input_modalities"] and "tools" in m["supported_parameters"],
    "ctx200k": lambda m: (m["context_length"] or 0) >= 200_000 and "tools" in m["supported_parameters"],
    "tools": lambda m: "tools" in m["supported_parameters"],
    "text": lambda m: "text" in m["architecture"]["output_modalities"],
}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(REPO, "logs", "openrouter-aliases.log")  # logs/ is git-ignored


def fetch_models():
    for i in range(3):
        try:
            with urllib.request.urlopen(MODELS_URL, timeout=20) as r:
                return json.load(r)["data"]
        except Exception as e:
            print(f"fetch attempt {i + 1} failed: {e}", file=sys.stderr)
            time.sleep(2 ** i)
    sys.exit("could not fetch OpenRouter models")


def free_models(data):
    p = lambda m, k: str((m.get("pricing") or {}).get(k))
    return {m["id"]: m for m in data
            if not m["id"].startswith("openrouter/")
            and m["id"] not in BLOCKED
            and (m["id"].endswith(":free") or (p(m, "prompt") == "0" and p(m, "completion") == "0"))}


def rule_for(alias):
    return next(c for rx, c in RULES if re.search(rx, alias))


def candidates(free, cap, exclude=()):
    ok = [m for m in free.values() if CHECKS[cap](m) and PREFIX + m["id"] not in exclude]
    vi = lambda m: PREFERRED_VENDORS.index(m["id"].split("/")[0]) if m["id"].split("/")[0] in PREFERRED_VENDORS else 99
    ok.sort(key=lambda m: (vi(m), -(m["context_length"] or 0), -m["created"]))
    return [PREFIX + m["id"] for m in ok]


def alive(target, free):
    return target in (PREFIX + i for i in free)


def plan(aliases, fallbacks, free):
    """Return (new_aliases, new_fallbacks, replaced[], suggestions[], warnings[])."""
    aliases, fallbacks = dict(aliases), [{k: list(v) for k, v in e.items()} for e in fallbacks]
    replaced, sugg, warn = [], [], []
    cap_of = {}  # openrouter target -> capability (from the aliases using it)
    for a, t in aliases.items():
        if not t.startswith(PREFIX) or t == TAIL:
            continue
        cap = rule_for(a)
        cands = candidates(free, cap)
        if not alive(t, free):
            new = next((c for c in cands if c != t), None)
            if new:
                replaced.append(f"alias {a}: {t} -> {new}")
                aliases[a] = t = new
            else:
                warn.append(f"alias {a}: {t} is gone/paid and no free model has '{cap}' - kept")
        elif cands and cands[0] != t:
            sugg.append(f"alias {a}: better candidate {cands[0]} (current {t})")
        cap_of[t] = cap
    for e in fallbacks:
        for key, chain in e.items():
            if not key.startswith(PREFIX) or key == TAIL:
                continue
            cap = cap_of.get(key, "tools")
            for i, f in enumerate(chain):
                if f.startswith(PREFIX) and f != TAIL and not alive(f, free):
                    new = next((c for c in candidates(free, cap, exclude=[key] + chain) ), None)
                    if new:
                        replaced.append(f"fallback {key}[{i}]: {f} -> {new}")
                        chain[i] = new
                    else:
                        warn.append(f"fallback {key}[{i}]: {f} dead, no candidate - kept")
    for t in EXTRA_CHAINS:
        cap_of.setdefault(t, "tools")
    for t, cap in cap_of.items():  # tail: every non-audio openrouter chain ends with the free router
        if cap == "audio_out":
            continue
        chain = next((e[t] for e in fallbacks if t in e), None)
        if chain is None:
            fallbacks.append({t: [TAIL]})
            replaced.append(f"fallback {t}: new chain [{TAIL}]")
        elif TAIL not in chain:
            chain.append(TAIL)
            replaced.append(f"fallback {t}: appended tail {TAIL}")
    for e in fallbacks:  # existing OpenRouter chains not tied to an alias
        for key, chain in e.items():
            if key.startswith(PREFIX) and key != TAIL and key not in cap_of and TAIL not in chain:
                chain.append(TAIL)
                replaced.append(f"fallback {key}: appended tail {TAIL}")
    return aliases, fallbacks, replaced, sugg, warn


def psql(sql, stdin=None):
    cmd = ["docker", "exec", "-i", "litellm-postgres", "sh", "-c",
           'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -At -v ON_ERROR_STOP=1 ' + sql]
    r = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=60)
    if r.returncode:
        sys.exit("psql failed: " + r.stderr.strip()[:300])
    return r.stdout


def read_settings():
    out = psql("""-c "select jsonb_build_object('a',param_value->'model_group_alias','f',param_value->'fallbacks') from \\"LiteLLM_Config\\" where param_name='router_settings'" """)
    d = json.loads(out)
    return d["a"] or {}, d["f"] or []


def write_settings(aliases, fallbacks):
    sql = ("update \"LiteLLM_Config\" set param_value = jsonb_set(jsonb_set(param_value,'{model_group_alias}',"
           f"$j${json.dumps(aliases)}$j$::jsonb),'{{fallbacks}}',$j${json.dumps(fallbacks)}$j$::jsonb) "
           "where param_name='router_settings';\n")
    psql("", stdin=sql)  # SQL via stdin; no secrets involved


def selftest():
    mk = lambda i, ctx=262144, inp=("text",), tools=True, out=("text",), created=1: {
        "id": i, "context_length": ctx, "created": created, "pricing": {"prompt": "0", "completion": "0"},
        "architecture": {"input_modalities": list(inp), "output_modalities": list(out)},
        "supported_parameters": ["tools"] if tools else []}
    free = free_models([
        mk("nvidia/a:free"), mk("qwen/vis:free", inp=("text", "image")), mk("google/old:free", created=1),
        mk("google/new:free", created=5), mk("x/notools:free", tools=False),
        mk("openrouter/free"), {**mk("p/paid"), "pricing": {"prompt": "0.1", "completion": "0.1"}}])
    assert "openrouter/free" not in free and "p/paid" not in free
    assert candidates(free, "tools")[:3] == ["openrouter/nvidia/a:free", "openrouter/qwen/vis:free", "openrouter/google/new:free"]
    assert candidates(free, "image_in") == ["openrouter/qwen/vis:free"]
    assert candidates(free, "audio_out") == []
    al = {"vision": "openrouter/gone/v:free", "coding": "openrouter/google/old:free", "tts": "openrouter/gone/t:free",
          "default": "openai/local"}
    fb = [{"openrouter/google/old:free": ["openrouter/gone/x:free", "openrouter/nvidia/a:free"]}]
    na, nf, rep, sug, warn = plan(al, fb, free)
    assert na["vision"] == "openrouter/qwen/vis:free"          # dead -> replaced by capable model
    assert na["coding"] == "openrouter/google/old:free"        # alive but not best -> kept
    assert any("coding" in s for s in sug)                     # ...and reported
    assert na["tts"] == "openrouter/gone/t:free" and warn      # no audio model -> kept + warned
    assert na["default"] == "openai/local"                     # non-OpenRouter untouched
    ch = nf[0]["openrouter/google/old:free"]
    assert ch[0] == "openrouter/qwen/vis:free" or ch[0].startswith("openrouter/") and ch[0] != "openrouter/gone/x:free"
    assert ch[1] == "openrouter/nvidia/a:free" and ch[-1] == TAIL
    assert not any("tts" in list(e)[0] for e in nf if TAIL in list(e.values())[0] and "gone/t" in list(e)[0])
    _, nf2, rep2, _, _ = plan(na, nf, free)
    assert nf2 == nf and not [r for r in rep2 if "->" in r]    # idempotent
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    free = free_models(fetch_models())
    aliases, fallbacks = read_settings()
    na, nf, rep, sugg, warn = plan(aliases, fallbacks, free)
    lines = [f"[{datetime.datetime.now():%F %T}] free models: {len(free)}; mode: {'apply' if args.apply else 'dry-run'}"]
    lines += ["CHANGE  " + r for r in rep] + ["SUGGEST " + s for s in sugg] + ["WARN    " + w for w in warn]
    if not (rep or sugg or warn):
        lines.append("nothing to do")
    print("\n".join(lines))
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write("\n".join(lines) + "\n")
    if args.apply and rep:
        subprocess.run([os.path.join(REPO, "scripts", "backup-litellm-db.sh")], check=True, stdout=subprocess.DEVNULL)
        write_settings(na, nf)
        print("applied (backup in LiteLLM/backup/). restart/reload litellm needed: docker restart litellm")
    elif args.apply:
        print("no changes to apply")


if __name__ == "__main__":
    main()
