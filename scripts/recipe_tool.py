#!/usr/bin/env python3
"""Export/import llama-swap model blocks as standalone recipe files.

A recipe file is a single model's config.yaml block (cmd/ttl/checkEndpoint/
proxy/endpoint/etc.) plus a bit of export metadata, saved under recipes/ so
a known-working model config can be versioned, diffed, and shared without
hand-editing the live llama-swap config.yaml.

    recipe_tool.py export <model-name>     # config.yaml -> recipes/<model>.yaml
    recipe_tool.py import <recipe-file>    # recipes/<model>.yaml -> config.yaml (dry-run by default)
    recipe_tool.py import <recipe-file> --apply   # actually write it (backs up config.yaml first)
    recipe_tool.py import-sparkrun <sparkrun-recipe.yaml> [--as NAME] [--apply]
                                            # translate an upstream sparkrun/spark-arena
                                            # recipe.yaml (https://sparkrun.dev/recipes/format/)
                                            # into a llama-swap model block
    recipe_tool.py list                    # list recipes/ contents

Round-trips comments and block-scalar (">"/"|") formatting in config.yaml via
ruamel.yaml, so importing/exporting doesn't scramble the surrounding file.
"""
import argparse
import datetime
import os
import re
import shlex
import shutil
import sys

try:
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap, CommentedSeq
except ImportError:
    sys.exit("error: this tool needs ruamel.yaml — install with: pip install ruamel.yaml")

yaml = YAML()
yaml.preserve_quotes = True
yaml.width = 4096  # don't rewrap long docker-run cmd lines
yaml.indent(mapping=2, sequence=2, offset=0)

RECIPE_VERSION = "1"


def strip_comments(obj):
    """Drop ruamel's attached comments (they bleed in from neighboring keys
    when a CommentedMap sub-block is dumped in isolation) while keeping
    scalar-string subtypes (FoldedScalarString/LiteralScalarString), which
    are plain str subclasses and carry their own block-style formatting.
    """
    if isinstance(obj, CommentedMap):
        return {k: strip_comments(v) for k, v in obj.items()}
    if isinstance(obj, CommentedSeq):
        return [strip_comments(v) for v in obj]
    return obj


def load(path):
    with open(path) as f:
        return yaml.load(f)


def dump(data, path_or_stream):
    if hasattr(path_or_stream, "write"):
        yaml.dump(data, path_or_stream)
    else:
        with open(path_or_stream, "w") as f:
            yaml.dump(data, f)


def cmd_export(args):
    cfg = load(args.config)
    models = cfg.get("models") or {}
    if args.model not in models:
        sys.exit(f"error: model '{args.model}' not found in {args.config}")

    block = strip_comments(models[args.model])
    recipe = {
        "recipe_version": RECIPE_VERSION,
        "name": args.model,
        "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": os.path.relpath(args.config),
        "model": block,
    }

    os.makedirs(args.recipes_dir, exist_ok=True)
    out_path = os.path.join(args.recipes_dir, f"{args.model}.yaml")
    if os.path.exists(out_path) and not args.force:
        sys.exit(f"error: {out_path} already exists (use --force to overwrite)")

    dump(recipe, out_path)
    print(f"exported: {out_path}")


def preview_or_apply(config_path, name, block, apply):
    """Shared diff/write logic for both `import` and `import-sparkrun`."""
    cfg = load(config_path)
    models = cfg.setdefault("models", {})
    existing_raw = models.get(name)
    existing = strip_comments(existing_raw) if existing_raw is not None else None

    if existing == block:
        print(f"no changes: '{name}' already matches")
        return

    if not apply:
        print(f"--- current models.{name} ({config_path}) ---")
        if existing is None:
            print("(not present)")
        else:
            dump({name: existing}, sys.stdout)
        print(f"\n--- new models.{name} ---")
        dump({name: block}, sys.stdout)
        print(f"\n(dry run — rerun with --apply to write this into {config_path})")
        return

    backup = f"{config_path}.bak.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(config_path, backup)

    # Mutate the existing submap in place rather than replacing it outright —
    # ruamel can attach trailing/neighboring file comments to a submap object
    # itself, so swapping in a fresh plain dict silently drops unrelated
    # comments that happened to live right after this model's block.
    if isinstance(existing_raw, CommentedMap):
        existing_raw.clear()
        for k, v in block.items():
            existing_raw[k] = v
    else:
        models[name] = block

    dump(cfg, config_path)
    print(f"applied: models.{name} updated in {config_path}")
    print(f"backup saved: {backup}")


def cmd_import(args):
    recipe = load(args.recipe)
    name = recipe.get("name")
    block = strip_comments(recipe.get("model"))
    if not name or block is None:
        sys.exit(f"error: {args.recipe} is missing required 'name' or 'model' field")
    preview_or_apply(args.config, name, block, args.apply)


SUPPORTED_SPARKRUN_RUNTIMES = ("vllm", "llama-cpp")


def build_block_from_sparkrun(recipe, name, network, hf_cache):
    """Translate a sparkrun/spark-arena recipe (https://sparkrun.dev/recipes/format/)
    into a llama-swap model block.

    sparkrun recipes describe one runtime process directly (model, container,
    defaults, a `command:` template with {placeholder} substitution). llama-swap
    instead wants a full `docker run ...` string per model, with ${PORT}/${host}
    filled in by llama-swap itself at launch time — so {port}/{host} placeholders
    are mapped to those macros (never baked to the recipe's literal defaults),
    and every other {placeholder} is substituted from `defaults`.
    """
    runtime = recipe.get("runtime")
    if runtime not in SUPPORTED_SPARKRUN_RUNTIMES:
        sys.exit(
            f"error: runtime '{runtime}' isn't one this stack knows how to run "
            f"(supported: {', '.join(SUPPORTED_SPARKRUN_RUNTIMES)}) — "
            "translate it by hand into a config.yaml cmd block instead"
        )

    container = recipe.get("container")
    model = recipe.get("model")
    command_tmpl = recipe.get("command")
    if not container or not model or not command_tmpl:
        sys.exit("error: recipe is missing required 'container', 'model', or 'command' field")

    defaults = dict(recipe.get("defaults") or {})
    env = recipe.get("env") or {}

    fmt_vars = dict(defaults)
    fmt_vars["model"] = model
    fmt_vars["port"] = "${PORT}"
    fmt_vars["host"] = "${host}"

    # Flatten the command template to one space-joined line: sparkrun recipes
    # write it with `\`-continued lines for human readability, but those
    # backslashes are meaningless (and unsafe) once embedded in a quoted
    # `bash -c '...'` string below, so collapse whitespace instead of
    # preserving the line breaks.
    flat = re.sub(r"\\\s*\n\s*", " ", command_tmpl).strip()
    flat = re.sub(r"\s+", " ", flat)
    try:
        command = flat.format(**fmt_vars)
    except KeyError as e:
        sys.exit(f"error: recipe command references undefined placeholder {{{e.args[0]}}} (not in defaults)")

    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "-", name)
    container_name = f"sparkrun-{safe_name}-${{PORT}}"

    parts = [
        f"docker run --rm --name {container_name}",
        f"--runtime nvidia --gpus all --ipc=host --network {network}",
    ]
    for k, v in env.items():
        parts.append(f"-e {k}={shlex.quote(str(v))}")
    # sparkrun recipes assume the runtime downloads the model by HF id on
    # first launch — mount the host HF cache so that download only happens
    # once instead of on every container start.
    parts.append(f"-v {hf_cache}:/root/.cache/huggingface")
    parts.append(f"--entrypoint /bin/bash {container}")
    parts.append(f"-c {shlex.quote(command)}")

    # Joined with plain spaces, NOT embedded newlines: a FoldedScalarString
    # built from a Python string containing literal "\n" forces ruamel to
    # preserve those exact newlines on dump (via blank-line escaping, per
    # YAML folding rules) so the string round-trips byte-for-byte. That
    # would land real newlines in the middle of this shell command, which
    # splits it into multiple invalid statements when llama-swap execs it.
    # A single space-joined line has no such ambiguity.
    full_cmd = " ".join(parts)

    block = {
        "ttl": 0,
        # First launch may need to download the model from HF — generous
        # timeout so that doesn't get mistaken for a hung/crashed container.
        "readyTimeout": 1200,
        "checkEndpoint": "/health",
        "cmd": full_cmd,
        "cmdStop": f"docker stop {container_name}",
    }
    return block


def cmd_import_sparkrun(args):
    recipe = load(args.recipe)
    name = args.as_name or (recipe.get("defaults") or {}).get("served_model_name") \
        or re.sub(r"^.*/", "", str(recipe.get("model") or "")) \
        or None
    if not name:
        sys.exit("error: couldn't derive a model name — pass --as <name>")

    block = build_block_from_sparkrun(recipe, name, args.network, args.hf_cache)
    preview_or_apply(args.config, name, block, args.apply)


def cmd_list(args):
    if not os.path.isdir(args.recipes_dir):
        print(f"(no recipes yet — {args.recipes_dir}/ does not exist)")
        return
    found = sorted(f for f in os.listdir(args.recipes_dir) if f.endswith(".yaml"))
    if not found:
        print(f"(no recipes in {args.recipes_dir}/)")
        return
    for fn in found:
        print(fn[:-5])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="llama-swap/config.yaml", help="path to llama-swap config.yaml")
    p.add_argument("--recipes-dir", default="recipes", help="directory to read/write recipe files")
    sub = p.add_subparsers(dest="action", required=True)

    pe = sub.add_parser("export", help="write a model's config.yaml block to a recipe file")
    pe.add_argument("model", help="model name (key under models: in config.yaml)")
    pe.add_argument("--force", action="store_true", help="overwrite an existing recipe file")
    pe.set_defaults(func=cmd_export)

    pi = sub.add_parser("import", help="apply a recipe file's block into config.yaml")
    pi.add_argument("recipe", help="path to a recipe .yaml file")
    pi.add_argument("--apply", action="store_true", help="write the change (default: dry-run diff only)")
    pi.set_defaults(func=cmd_import)

    ps = sub.add_parser(
        "import-sparkrun",
        help="translate an upstream sparkrun/spark-arena recipe.yaml into a config.yaml model block",
    )
    ps.add_argument("recipe", help="path to a sparkrun-format recipe .yaml file")
    ps.add_argument("--as", dest="as_name", default=None, help="model name to use in config.yaml (default: served_model_name or the HF repo's last path segment)")
    ps.add_argument("--network", default="container:llama-swap", help="docker --network value (default: container:llama-swap)")
    ps.add_argument("--hf-cache", default=os.path.expanduser("~/.cache/huggingface"), help="host path to mount as the container's HF cache (default: ~/.cache/huggingface)")
    ps.add_argument("--apply", action="store_true", help="write the change (default: dry-run diff only)")
    ps.set_defaults(func=cmd_import_sparkrun)

    pl = sub.add_parser("list", help="list available recipe files")
    pl.set_defaults(func=cmd_list)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
