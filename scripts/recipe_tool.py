#!/usr/bin/env python3
"""Export/import llama-swap model blocks as standalone recipe files.

A recipe file is a single model's config.yaml block (cmd/ttl/checkEndpoint/
proxy/endpoint/etc.) plus a bit of export metadata, saved under recipes/ so
a known-working model config can be versioned, diffed, and shared without
hand-editing the live llama-swap config.yaml.

    recipe_tool.py export <model-name>     # config.yaml -> recipes/<model>.yaml
    recipe_tool.py import <recipe-file>    # recipes/<model>.yaml -> config.yaml (dry-run by default)
    recipe_tool.py import <recipe-file> --apply   # actually write it (backs up config.yaml first)
    recipe_tool.py list                    # list recipes/ contents

Round-trips comments and block-scalar (">"/"|") formatting in config.yaml via
ruamel.yaml, so importing/exporting doesn't scramble the surrounding file.
"""
import argparse
import datetime
import os
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


def cmd_import(args):
    recipe = load(args.recipe)
    name = recipe.get("name")
    block = strip_comments(recipe.get("model"))
    if not name or block is None:
        sys.exit(f"error: {args.recipe} is missing required 'name' or 'model' field")

    cfg = load(args.config)
    models = cfg.setdefault("models", {})
    existing_raw = models.get(name)
    existing = strip_comments(existing_raw) if existing_raw is not None else None

    if existing == block:
        print(f"no changes: '{name}' already matches {args.recipe}")
        return

    if not args.apply:
        print(f"--- current models.{name} ({args.config}) ---")
        if existing is None:
            print("(not present)")
        else:
            dump({name: existing}, sys.stdout)
        print(f"\n--- recipe models.{name} ({args.recipe}) ---")
        dump({name: block}, sys.stdout)
        print(f"\n(dry run — rerun with --apply to write this into {args.config})")
        return

    backup = f"{args.config}.bak.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(args.config, backup)

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

    dump(cfg, args.config)
    print(f"applied: models.{name} updated in {args.config}")
    print(f"backup saved: {backup}")


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

    pl = sub.add_parser("list", help="list available recipe files")
    pl.set_defaults(func=cmd_list)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
