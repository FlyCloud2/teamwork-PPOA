#!/usr/bin/env python
"""Audit selected PPO math and default initialization against the pinned upstream file."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from run import REPO, atomic_json


def dump(node):
    return ast.dump(node, include_attributes=False)


def named(tree, kind, name):
    return next(node for node in ast.walk(tree) if isinstance(node, kind) and node.name == name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()
    upstream = args.upstream.read_text(encoding="utf-8-sig")
    upstream_digest = hashlib.sha256(upstream.rstrip().encode()).hexdigest()
    if upstream_digest != "612fb2559e0fc169cd80d305dfc0241486819ff0501a465b2a04d5d3b9c894f2":
        parser.error("input does not match the pinned CleanRL source fingerprint")
    local = (REPO / "src/ppo_continuous_action.py").read_text(encoding="utf-8")
    trees = [ast.parse(source) for source in (upstream, local)]
    checks = {}
    for name in ("make_env", "layer_init", "get_value", "get_action_and_value"):
        checks[name] = dump(named(trees[0], ast.FunctionDef, name)) == dump(named(trees[1], ast.FunctionDef, name))
    minibatch_bodies = []
    recurrences = []
    for tree in trees:
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "start")
        # The first two statements derive the batch boundaries, which W2 changes.
        minibatch_bodies.append([dump(node) for node in loop.body[2:]])
        recurrences.append([dump(node) for node in ast.walk(tree) if isinstance(node, ast.Assign)
                            and any((isinstance(target, ast.Name) and target.id == "delta")
                                    or (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "advantages")
                                    for target in node.targets)])
    checks["minibatch_loss_and_optimizer_body"] = minibatch_bodies[0] == minibatch_bodies[1]
    checks["gae_recurrence"] = recurrences[0] == recurrences[1]
    models = []
    envs = SimpleNamespace(single_observation_space=SimpleNamespace(shape=(17,)), single_action_space=SimpleNamespace(shape=(6,)))
    torch.set_num_threads(1)
    for tree in trees:
        nodes = [named(tree, ast.FunctionDef, "layer_init"), named(tree, ast.ClassDef, "Agent")]
        namespace = {"torch": torch, "nn": nn, "np": np, "Normal": Normal}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<audited agent definitions>", "exec"), namespace)
        torch.manual_seed(0)
        models.append(namespace["Agent"](envs).state_dict())
    checks["default_network_initialization"] = models[0].keys() == models[1].keys() and all(torch.equal(models[0][key], models[1][key]) for key in models[0])
    result = {
        "upstream_repository": "https://github.com/vwxyzjn/cleanrl",
        "upstream_commit": "cf20043b852deedf52dc9f90a325c6866d599005",
        "upstream_file": "cleanrl/ppo_continuous_action.py",
        "upstream_normalized_source_sha256": upstream_digest,
        "local_source_sha256": hashlib.sha256((REPO / "src/ppo_continuous_action.py").read_bytes()).hexdigest(),
        "checks": checks,
        "scope": "Selected mathematical code and default initialization; the whole file intentionally differs (CLI, logging, exact final rollout).",
    }
    atomic_json(REPO / "logs/upstream_audit.json", result)
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
