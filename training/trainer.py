"""
Chainer Agent — Production PPO Training Service

Single-host training service with:
  - shared policy families with agent/archetype conditioning
  - per-agent ONNX exports with baked bindings
  - filesystem-backed registry for families, agents, aliases, and evaluation reports
  - promotion governance for latest / candidate / champion
"""

from __future__ import annotations

import copy
import json
import math
import os
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

STATE_DIM = 24
ACTION_DIM = 6
STRATEGY_KEYS = [
    "aggression",
    "accuracy_focus",
    "crystal_priority",
    "ability_usage",
    "retreat_threshold",
]
STRATEGY_DIM = len(STRATEGY_KEYS)
AGENT_EMBED_DIM = 12
ARCHETYPE_EMBED_DIM = 8
HIDDEN_DIM = 128
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
PPO_EPOCHS = 4
MODEL_SCHEMA_VERSION = 3
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
INPUT_CLAMP = 8.0
ACTION_CLAMP = 1.0
MODELS_DIR = Path("models")
POLICIES_DIR = MODELS_DIR / "policies"
AGENTS_DIR = MODELS_DIR / "agents"
LOGS_DIR = Path("training_logs")
DEFAULT_POLICY_FAMILY = "arena-main"
DEFAULT_ARCHETYPE_ID = "tactician"
PROMOTION_MARGIN = 0.05
PROMOTION_MIN_WIN_RATE = 0.55
TRAIN_STALL_SECONDS = 300
EXPORT_STALL_SECONDS = 300
DEVICE = torch.device("cpu")

ARCHETYPE_DEFAULTS = {
    "hunter": {"aggression": 0.9, "accuracy_focus": 0.4, "crystal_priority": 0.1, "ability_usage": 0.7, "retreat_threshold": 0.1},
    "sniper": {"aggression": 0.2, "accuracy_focus": 0.95, "crystal_priority": 0.2, "ability_usage": 0.3, "retreat_threshold": 0.4},
    "collector": {"aggression": 0.1, "accuracy_focus": 0.3, "crystal_priority": 0.95, "ability_usage": 0.2, "retreat_threshold": 0.6},
    "survivor": {"aggression": 0.3, "accuracy_focus": 0.5, "crystal_priority": 0.4, "ability_usage": 0.5, "retreat_threshold": 0.7},
    "berserker": {"aggression": 1.0, "accuracy_focus": 0.2, "crystal_priority": 0.0, "ability_usage": 1.0, "retreat_threshold": 0.0},
    "tactician": {"aggression": 0.5, "accuracy_focus": 0.6, "crystal_priority": 0.3, "ability_usage": 0.5, "retreat_threshold": 0.35},
    "flanker": {"aggression": 0.6, "accuracy_focus": 0.7, "crystal_priority": 0.2, "ability_usage": 0.6, "retreat_threshold": 0.3},
    "guardian": {"aggression": 0.7, "accuracy_focus": 0.5, "crystal_priority": 0.4, "ability_usage": 0.8, "retreat_threshold": 0.25},
}

PROMOTION_HISTORY = LOGS_DIR / "promotions.jsonl"
EVALUATION_HISTORY = LOGS_DIR / "evaluations.jsonl"


def _clamp01(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _strategy_from_any(raw, archetype_id=DEFAULT_ARCHETYPE_ID) -> dict:
    base = dict(ARCHETYPE_DEFAULTS.get(archetype_id, ARCHETYPE_DEFAULTS[DEFAULT_ARCHETYPE_ID]))
    if isinstance(raw, dict):
        for key in STRATEGY_KEYS:
            if key in raw:
                base[key] = _clamp01(raw[key])
    elif isinstance(raw, list) and len(raw) == STRATEGY_DIM:
        for index, key in enumerate(STRATEGY_KEYS):
            base[key] = _clamp01(raw[index])
    return base


def _strategy_vector(raw, archetype_id=DEFAULT_ARCHETYPE_ID) -> list[float]:
    strategy = _strategy_from_any(raw, archetype_id)
    return [strategy[key] for key in STRATEGY_KEYS]


def _normalize_version(version_like) -> str:
    return str(version_like).split("+", maxsplit=1)[0]


def _safe_mean(values):
    return float(np.mean(values)) if values else 0.0


def _sanitize_tensor(tensor: torch.Tensor, clamp_min=None, clamp_max=None) -> torch.Tensor:
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    if clamp_min is not None or clamp_max is not None:
        tensor = torch.clamp(tensor, min=clamp_min, max=clamp_max)
    return tensor


def _read_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return copy.deepcopy(default)


def _append_jsonl(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def _is_valid_float_list(lst, expected_len):
    if not isinstance(lst, list) or len(lst) != expected_len:
        return False
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in lst)


class ConditionedActorCritic(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, agent_count=1, archetype_count=1):
        super().__init__()
        self.agent_embedding = nn.Embedding(max(1, agent_count), AGENT_EMBED_DIM)
        self.archetype_embedding = nn.Embedding(max(1, archetype_count), ARCHETYPE_EMBED_DIM)
        input_dim = state_dim + STRATEGY_DIM + AGENT_EMBED_DIM + ARCHETYPE_EMBED_DIM
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim := HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state, agent_index, archetype_index, strategy):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if agent_index.dim() == 0:
            agent_index = agent_index.unsqueeze(0)
        if archetype_index.dim() == 0:
            archetype_index = archetype_index.unsqueeze(0)
        if strategy.dim() == 1:
            strategy = strategy.unsqueeze(0)

        agent_embedding = self.agent_embedding(agent_index)
        archetype_embedding = self.archetype_embedding(archetype_index)
        joined = _sanitize_tensor(
            torch.cat([state, strategy, agent_embedding, archetype_embedding], dim=-1),
            clamp_min=-INPUT_CLAMP,
            clamp_max=INPUT_CLAMP,
        )
        shared = _sanitize_tensor(self.shared(joined), clamp_min=-INPUT_CLAMP, clamp_max=INPUT_CLAMP)
        mean = torch.tanh(self.actor_mean(shared))
        log_std = torch.clamp(self.actor_log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        value = self.critic(shared)
        return mean, log_std, value

    def resize_embeddings(self, agent_count: int, archetype_count: int):
        self.agent_embedding = self._resize_embedding(self.agent_embedding, agent_count)
        self.archetype_embedding = self._resize_embedding(self.archetype_embedding, archetype_count)

    @staticmethod
    def _resize_embedding(layer: nn.Embedding, new_count: int):
        old_weight = layer.weight.data.clone()
        target = nn.Embedding(max(new_count, 1), old_weight.shape[1])
        nn.init.normal_(target.weight, mean=0.0, std=0.02)
        target.weight.data[: old_weight.shape[0]] = old_weight
        return target


class BoundActorOnly(nn.Module):
    def __init__(self, policy: ConditionedActorCritic, agent_index: int, archetype_index: int, strategy_values: list[float]):
        super().__init__()
        self.policy = policy
        self.register_buffer("bound_agent_index", torch.tensor([agent_index], dtype=torch.long))
        self.register_buffer("bound_archetype_index", torch.tensor([archetype_index], dtype=torch.long))
        self.register_buffer("bound_strategy", torch.tensor([strategy_values], dtype=torch.float32))

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        batch = state.shape[0]
        agent_index = self.bound_agent_index.expand(batch)
        archetype_index = self.bound_archetype_index.expand(batch)
        strategy = self.bound_strategy.expand(batch, -1)
        mean, _, _ = self.policy(state, agent_index, archetype_index, strategy)
        return torch.tanh(mean)


@dataclass
class ExportRequest:
    family_id: str
    model_version: int
    reason: str


class AgentRecord:
    def __init__(self, agent_id: str, policy_family=DEFAULT_POLICY_FAMILY, archetype_id=DEFAULT_ARCHETYPE_ID):
        self.agent_id = agent_id
        self.policy_family = policy_family
        self.archetype_id = archetype_id
        self.total_episodes = 0
        self.total_steps = 0
        self.best_score = 0.0
        self.train_steps = 0
        self.model_version = 0
        self.score_history: list[float] = []
        self.kd_history: list[float] = []
        self.damage_history: list[float] = []
        self.last_strategy = {"strategy": _strategy_from_any(None, archetype_id)}
        self.last_reward_totals = {}

    def apply_binding(self, binding: dict):
        self.policy_family = binding.get("policy_family", self.policy_family)
        self.archetype_id = binding.get("archetype_id", self.archetype_id)
        if "strategy" in binding:
            self.last_strategy = {"strategy": _strategy_from_any(binding["strategy"], self.archetype_id)}

    def note_model_version(self, version: int, train_steps: int):
        self.model_version = int(version)
        self.train_steps = int(train_steps)

    def note_step(self, amount=1):
        self.total_steps += amount

    def record_episode(self, summary: dict):
        score = float(summary.get("score", 0))
        kills = float(summary.get("kills", 0))
        deaths = float(summary.get("deaths", 0))
        damage = float(summary.get("damageDealt", 0))
        self.total_episodes += 1
        self.best_score = max(self.best_score, score)
        self.score_history.append(score)
        self.kd_history.append(kills / max(deaths, 1.0))
        self.damage_history.append(damage)
        self.last_reward_totals = dict(summary.get("reward_totals", {}))
        for history in [self.score_history, self.kd_history, self.damage_history]:
            if len(history) > 50:
                history.pop(0)

    def restore(self, payload: dict):
        self.total_episodes = int(payload.get("episodes", self.total_episodes))
        self.total_steps = int(payload.get("total_steps", self.total_steps))
        self.train_steps = int(payload.get("train_steps", self.train_steps))
        self.model_version = int(payload.get("model_version", self.model_version))
        self.best_score = float(payload.get("best_score", self.best_score))
        self.score_history = list(payload.get("score_history", self.score_history))
        self.kd_history = list(payload.get("kd_history", self.kd_history))
        self.damage_history = list(payload.get("damage_history", self.damage_history))
        self.last_strategy = dict(payload.get("last_strategy", self.last_strategy))
        self.last_reward_totals = dict(payload.get("last_reward_totals", self.last_reward_totals))
        self.policy_family = payload.get("policy_family", self.policy_family)
        self.archetype_id = payload.get("archetype_id", self.archetype_id)

    def get_stats(self, aliases: dict | None = None):
        return {
            "agent_id": self.agent_id,
            "policy_family": self.policy_family,
            "archetype_id": self.archetype_id,
            "episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "train_steps": self.train_steps,
            "model_version": self.model_version,
            "best_score": round(self.best_score, 1),
            "avg_score_50": round(_safe_mean(self.score_history), 1),
            "avg_kd_50": round(_safe_mean(self.kd_history), 2),
            "avg_damage_50": round(_safe_mean(self.damage_history), 1),
            "buffer_size": 0,
            "score_history": self.score_history[-20:],
            "kd_history": [round(value, 2) for value in self.kd_history[-20:]],
            "aliases": aliases or {},
            "last_strategy": self.last_strategy,
            "last_reward_totals": self.last_reward_totals,
        }


class PolicyFamily:
    def __init__(self, family_id: str):
        self.family_id = family_id
        self.model = ConditionedActorCritic()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.model_version = 0
        self.train_steps = 0
        self.last_train_time = 0.0
        self.agent_slots: dict[str, int] = {}
        self.archetype_slots: dict[str, int] = {}
        self.bound_agents: set[str] = set()
        self.states: list[list[float]] = []
        self.actions: list[list[float]] = []
        self.rewards: list[float] = []
        self.reward_components: list[dict] = []
        self.values: list[float] = []
        self.log_probs: list[float] = []
        self.dones: list[float] = []
        self.agent_indices: list[int] = []
        self.archetype_indices: list[int] = []
        self.strategy_vectors: list[list[float]] = []
        self.last_eval_report = None

    def ensure_slots(self, agent_id: str, archetype_id: str):
        changed = False
        if agent_id not in self.agent_slots:
            self.agent_slots[agent_id] = len(self.agent_slots)
            changed = True
        if archetype_id not in self.archetype_slots:
            self.archetype_slots[archetype_id] = len(self.archetype_slots)
            changed = True
        if changed:
            self.model.resize_embeddings(len(self.agent_slots), len(self.archetype_slots))
            self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        return self.agent_slots[agent_id], self.archetype_slots[archetype_id]

    def bind_agent(self, agent_id: str, archetype_id: str):
        self.bound_agents.add(agent_id)
        return self.ensure_slots(agent_id, archetype_id)

    def add_transition(self, state, action, reward, reward_components, value, log_prob, done, agent_index, archetype_index, strategy_values):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.reward_components.append(reward_components or {})
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.dones.append(float(done))
        self.agent_indices.append(int(agent_index))
        self.archetype_indices.append(int(archetype_index))
        self.strategy_vectors.append(strategy_values)

    def clear_buffer(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.reward_components.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.agent_indices.clear()
        self.archetype_indices.clear()
        self.strategy_vectors.clear()

    def has_enough_experience(self):
        return len(self.states) >= BATCH_SIZE

    def has_finite_parameters(self):
        for tensor in self.model.state_dict().values():
            if not torch.isfinite(tensor).all():
                return False
        return True

    def get_summary(self, aliases: dict | None = None):
        return {
            "family_id": self.family_id,
            "model_version": self.model_version,
            "train_steps": self.train_steps,
            "bound_agents": sorted(self.bound_agents),
            "agent_count": len(self.bound_agents),
            "archetypes": sorted(self.archetype_slots.keys()),
            "aliases": aliases or {},
            "last_eval_report": self.last_eval_report,
        }


class PPOTrainer:
    def __init__(self):
        self.agents: dict[str, AgentRecord] = {}
        self.bindings: dict[str, dict] = {}
        self.families: dict[str, PolicyFamily] = {}
        self.lock = threading.RLock()
        self.start_time = time.time()
        self.total_experiences = 0
        self.total_train_steps = 0
        self.errors = 0
        self.last_export_error = None
        self.last_train_error = None
        self.last_training_at = None
        self.last_export_at = None
        self.pending_exports: dict[tuple[str, int], threading.Event] = {}
        self.export_queue: queue.Queue[ExportRequest] = queue.Queue()
        self.alias_defaults = {"latest": 0, "candidate": 0, "champion": 0}

        POLICIES_DIR.mkdir(parents=True, exist_ok=True)
        AGENTS_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        torch.set_num_threads(1)
        self._validate_runtime_contracts()

        self.export_thread = threading.Thread(target=self._export_worker, daemon=True)
        self.export_thread.start()
        print(f"[Trainer] Device: {DEVICE} (shared conditioned CPU training)")

    def _load_family_checkpoint_payload(self, checkpoint_path: Path) -> dict:
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        if ckpt.get("schema_version") not in (MODEL_SCHEMA_VERSION, None):
            raise RuntimeError("checkpoint schema mismatch")
        return ckpt

    def _hydrate_family_from_checkpoint(self, family: PolicyFamily, ckpt: dict):
        family.model.resize_embeddings(
            max(len(ckpt.get("agent_slots", {})), 1),
            max(len(ckpt.get("archetype_slots", {})), 1),
        )
        family.model.load_state_dict(ckpt["model_state"])
        family.optimizer = optim.Adam(family.model.parameters(), lr=LEARNING_RATE)
        optimizer_state = ckpt.get("optimizer_state")
        if optimizer_state:
            family.optimizer.load_state_dict(optimizer_state)
        family.model_version = int(ckpt.get("model_version", 0))
        family.train_steps = int(ckpt.get("train_steps", 0))
        family.agent_slots = dict(ckpt.get("agent_slots", {}))
        family.archetype_slots = dict(ckpt.get("archetype_slots", {}))
        family.bound_agents = set(ckpt.get("bound_agents", [])) or set(family.agent_slots.keys())
        family.last_eval_report = ckpt.get("last_eval_report")

    def _validate_runtime_contracts(self):
        if sys.version_info < (3, 12):
            raise RuntimeError(f"Python 3.12+ required, found {sys.version_info.major}.{sys.version_info.minor}")

        required = {
            "torch": "2.11.0",
            "numpy": "2.4.4",
        }
        for package_name, expected in required.items():
            module = __import__(package_name)
            actual = getattr(module, "__version__", "unknown")
            if _normalize_version(actual) != expected:
                raise RuntimeError(f"{package_name} version mismatch: expected {expected}, found {actual}")

        import onnx  # pylint: disable=import-outside-toplevel
        import onnxscript  # pylint: disable=import-outside-toplevel

        if _normalize_version(onnx.__version__) != "1.21.0":
            raise RuntimeError(f"onnx version mismatch: expected 1.21.0, found {onnx.__version__}")
        if _normalize_version(onnxscript.__version__) != "0.6.2":
            raise RuntimeError(f"onnxscript version mismatch: expected 0.6.2, found {onnxscript.__version__}")

    # Path helpers
    def _family_dir(self, family_id: str) -> Path:
        return POLICIES_DIR / family_id

    def _family_versions_dir(self, family_id: str) -> Path:
        return self._family_dir(family_id) / "versions"

    def _family_version_dir(self, family_id: str, version: int) -> Path:
        return self._family_versions_dir(family_id) / f"v{version:06d}"

    def _family_aliases_path(self, family_id: str) -> Path:
        return self._family_dir(family_id) / "aliases.json"

    def _family_checkpoint_path(self, family_id: str, version: int) -> Path:
        return self._family_version_dir(family_id, version) / "checkpoint.pt"

    def _family_metadata_path(self, family_id: str, version: int) -> Path:
        return self._family_version_dir(family_id, version) / "metadata.json"

    def _family_eval_path(self, family_id: str, version: int) -> Path:
        return self._family_version_dir(family_id, version) / "eval.json"

    def _agent_dir(self, agent_id: str) -> Path:
        return AGENTS_DIR / agent_id

    def _binding_path(self, agent_id: str) -> Path:
        return self._agent_dir(agent_id) / "binding.json"

    def _agent_aliases_path(self, agent_id: str) -> Path:
        return self._agent_dir(agent_id) / "aliases.json"

    def _agent_versions_dir(self, agent_id: str) -> Path:
        return self._agent_dir(agent_id) / "versions"

    def _agent_version_dir(self, agent_id: str, version: int) -> Path:
        return self._agent_versions_dir(agent_id) / f"v{version:06d}"

    def _agent_policy_path(self, agent_id: str, version: int) -> Path:
        return self._agent_version_dir(agent_id, version) / "policy.onnx"

    def _agent_metadata_path(self, agent_id: str, version: int) -> Path:
        return self._agent_version_dir(agent_id, version) / "metadata.json"

    def _history_path(self, agent_id: str) -> Path:
        return LOGS_DIR / f"{agent_id}.jsonl"

    # Registry helpers
    def _default_binding(self, agent_id: str, policy_family=None, archetype_id=None) -> dict:
        digits = re.sub(r"\D", "", agent_id)
        index = int(digits or "0")
        resolved_arch = archetype_id or list(ARCHETYPE_DEFAULTS.keys())[index % len(ARCHETYPE_DEFAULTS)]
        return {
            "agent_id": agent_id,
            "policy_family": policy_family or DEFAULT_POLICY_FAMILY,
            "archetype_id": resolved_arch,
            "strategy": _strategy_from_any(None, resolved_arch),
            "updated_at": time.time(),
        }

    def _read_family_aliases(self, family_id: str) -> dict:
        return _read_json(self._family_aliases_path(family_id), self.alias_defaults)

    def _write_family_aliases(self, family_id: str, aliases: dict):
        path = self._family_aliases_path(family_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(aliases, indent=2))

    def _read_agent_aliases(self, agent_id: str) -> dict:
        return _read_json(self._agent_aliases_path(agent_id), self.alias_defaults)

    def _write_agent_aliases(self, agent_id: str, aliases: dict):
        path = self._agent_aliases_path(agent_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(aliases, indent=2))

    def _read_binding(self, agent_id: str) -> dict | None:
        path = self._binding_path(agent_id)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                return None
        return None

    def _write_binding(self, agent_id: str, binding: dict):
        path = self._binding_path(agent_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(binding, indent=2))

    def get_or_create_family(self, family_id: str) -> PolicyFamily:
        family = self.families.get(family_id)
        if family:
            return family
        family = PolicyFamily(family_id)
        self.families[family_id] = family
        self._write_family_aliases(family_id, self._read_family_aliases(family_id))
        return family

    def get_or_create_agent(self, agent_id: str, binding: dict | None = None) -> AgentRecord:
        agent = self.agents.get(agent_id)
        if agent:
            if binding:
                agent.apply_binding(binding)
            return agent
        binding = binding or self._default_binding(agent_id)
        agent = AgentRecord(agent_id, binding["policy_family"], binding["archetype_id"])
        agent.apply_binding(binding)
        self.agents[agent_id] = agent
        self._write_agent_aliases(agent_id, self._read_agent_aliases(agent_id))
        return agent

    def ensure_binding(self, agent_id: str, policy_family=None, archetype_id=None, strategy_vector=None) -> dict:
        binding = self.bindings.get(agent_id) or self._read_binding(agent_id) or self._default_binding(
            agent_id, policy_family, archetype_id
        )
        if policy_family:
            binding["policy_family"] = policy_family
        if archetype_id:
            binding["archetype_id"] = archetype_id
        if strategy_vector is not None:
            binding["strategy"] = _strategy_from_any(strategy_vector, binding["archetype_id"])
        else:
            binding["strategy"] = _strategy_from_any(binding.get("strategy"), binding["archetype_id"])
        binding["updated_at"] = time.time()

        self.bindings[agent_id] = binding
        self._write_binding(agent_id, binding)

        family = self.get_or_create_family(binding["policy_family"])
        family.bind_agent(agent_id, binding["archetype_id"])

        agent = self.get_or_create_agent(agent_id, binding)
        agent.apply_binding(binding)

        family_aliases = self._read_family_aliases(binding["policy_family"])
        aliases = self._read_agent_aliases(agent_id)
        changed = False
        for alias in self.alias_defaults:
          # Keep manual champion history, but hydrate unset aliases from family aliases.
            if aliases.get(alias) in (None, 0) and family_aliases.get(alias) not in (None, 0):
                aliases[alias] = family_aliases[alias]
                changed = True
        if changed:
            self._write_agent_aliases(agent_id, aliases)
            agent.note_model_version(aliases.get("latest", family.model_version), family.train_steps)

        return binding

    def resolve_version(self, agent_id: str, alias: str | None = None, version: str | None = None) -> int | None:
        if version is not None:
            if isinstance(version, str) and version.startswith("v"):
                version = version[1:]
            return int(version)
        aliases = self._read_agent_aliases(agent_id)
        chosen = alias or "latest"
        return aliases.get(chosen)

    def resolve_family_version(self, family_id: str, alias: str | None = None, version: str | None = None) -> int | None:
        if version is not None:
            if isinstance(version, str) and version.startswith("v"):
                version = version[1:]
            return int(version)
        aliases = self._read_family_aliases(family_id)
        return aliases.get(alias or "latest")

    # Training
    def process_experience(self, agent_id: str, transitions: list, policy_family=None, archetype_id=None, strategy_vector=None):
        binding = self.ensure_binding(agent_id, policy_family, archetype_id, strategy_vector)
        family = self.get_or_create_family(binding["policy_family"])
        agent = self.get_or_create_agent(agent_id, binding)
        agent_index, archetype_index = family.ensure_slots(agent_id, binding["archetype_id"])
        strategy_values = _strategy_vector(binding["strategy"], binding["archetype_id"])

        accepted = 0
        for transition in transitions:
            state = transition.get("state")
            action = transition.get("action")
            if not _is_valid_float_list(state, STATE_DIM):
                continue
            if not _is_valid_float_list(action, ACTION_DIM):
                continue

            state = [max(-INPUT_CLAMP, min(INPUT_CLAMP, float(value))) for value in state]
            action = [max(-ACTION_CLAMP, min(ACTION_CLAMP, float(value))) for value in action]

            reward = float(transition.get("reward", 0))
            if not math.isfinite(reward):
                reward = 0.0
            reward = max(-20.0, min(20.0, reward))
            done = 1.0 if transition.get("done") else 0.0

            state_t = torch.FloatTensor(state).unsqueeze(0)
            action_t = torch.FloatTensor(action).unsqueeze(0)
            agent_t = torch.LongTensor([agent_index])
            archetype_t = torch.LongTensor([archetype_index])
            strategy_t = torch.FloatTensor([strategy_values])
            with torch.no_grad():
                action_mean, log_std, value = family.model(state_t, agent_t, archetype_t, strategy_t)
                if not torch.isfinite(action_mean).all() or not torch.isfinite(log_std).all() or not torch.isfinite(value).all():
                    continue
                std = log_std.exp().clamp(min=1e-6)
                dist = torch.distributions.Normal(action_mean, std)
                log_prob = dist.log_prob(action_t).sum(-1)
                if not torch.isfinite(log_prob).all():
                    continue

            family.add_transition(
                state,
                action,
                reward,
                transition.get("reward_components", {}),
                value.item(),
                log_prob.item(),
                done,
                agent_index,
                archetype_index,
                strategy_values,
            )
            agent.note_step()
            accepted += 1

        self.total_experiences += accepted

        if family.has_enough_experience():
            try:
                self._train_family(family.family_id)
            except Exception as exc:
                self.errors += 1
                self.last_train_error = str(exc)
                print(f"[Trainer] Training failed for {family.family_id}: {exc}")
                family.clear_buffer()

        return accepted

    def _train_family(self, family_id: str):
        family = self.families[family_id]
        t0 = time.time()
        snapshot = copy.deepcopy(family.model.state_dict())

        try:
            states = _sanitize_tensor(torch.FloatTensor(np.array(family.states)), clamp_min=-INPUT_CLAMP, clamp_max=INPUT_CLAMP)
            actions = _sanitize_tensor(torch.FloatTensor(np.array(family.actions)), clamp_min=-ACTION_CLAMP, clamp_max=ACTION_CLAMP)
            agent_indices = torch.LongTensor(np.array(family.agent_indices))
            archetype_indices = torch.LongTensor(np.array(family.archetype_indices))
            strategies = _sanitize_tensor(torch.FloatTensor(np.array(family.strategy_vectors)), clamp_min=0.0, clamp_max=1.0)
            old_log_probs = _sanitize_tensor(torch.FloatTensor(family.log_probs), clamp_min=-100.0, clamp_max=100.0)
            rewards = [max(-20.0, min(20.0, float(reward) if math.isfinite(reward) else 0.0)) for reward in family.rewards]
            values = [float(value) if math.isfinite(value) else 0.0 for value in family.values]
            dones = [1.0 if done else 0.0 for done in family.dones]

            advantages = [0.0] * len(rewards)
            gae = 0.0
            for idx in reversed(range(len(rewards))):
                next_value = values[idx + 1] if idx < len(rewards) - 1 else 0.0
                delta = rewards[idx] + GAMMA * next_value * (1 - dones[idx]) - values[idx]
                gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[idx]) * gae
                if not math.isfinite(gae):
                    gae = 0.0
                advantages[idx] = gae

            advantages_t = _sanitize_tensor(torch.FloatTensor(advantages), clamp_min=-100.0, clamp_max=100.0)
            returns = _sanitize_tensor(advantages_t + torch.FloatTensor(values), clamp_min=-100.0, clamp_max=100.0)
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
            advantages_t = _sanitize_tensor(advantages_t, clamp_min=-20.0, clamp_max=20.0)

            for _ in range(PPO_EPOCHS):
                indices = torch.randperm(len(states))
                for start in range(0, len(states), 64):
                    idx = indices[start : start + 64]
                    b_states = states[idx]
                    b_actions = actions[idx]
                    b_agent = agent_indices[idx]
                    b_archetype = archetype_indices[idx]
                    b_strategy = strategies[idx]
                    b_old_lp = old_log_probs[idx]
                    b_adv = advantages_t[idx]
                    b_ret = returns[idx]

                    action_mean, log_std, value = family.model(b_states, b_agent, b_archetype, b_strategy)
                    action_mean = _sanitize_tensor(action_mean, clamp_min=-ACTION_CLAMP, clamp_max=ACTION_CLAMP)
                    log_std = _sanitize_tensor(log_std, clamp_min=LOG_STD_MIN, clamp_max=LOG_STD_MAX)
                    value = _sanitize_tensor(value.squeeze(-1), clamp_min=-100.0, clamp_max=100.0)
                    std = log_std.exp().clamp(min=1e-6, max=7.0)
                    dist = torch.distributions.Normal(action_mean, std)
                    new_lp = _sanitize_tensor(dist.log_prob(b_actions).sum(-1), clamp_min=-100.0, clamp_max=100.0)
                    entropy = _sanitize_tensor(dist.entropy().sum(-1).mean(), clamp_min=0.0, clamp_max=100.0)

                    ratio = _sanitize_tensor((new_lp - b_old_lp).exp(), clamp_min=0.0, clamp_max=20.0)
                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * b_adv
                    value_loss = (b_ret - value).pow(2).mean()
                    loss = -torch.min(surr1, surr2).mean() + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite PPO loss")

                    family.optimizer.zero_grad()
                    loss.backward()
                    for parameter in family.model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.data = torch.nan_to_num(parameter.grad.data, nan=0.0, posinf=1.0, neginf=-1.0)
                    nn.utils.clip_grad_norm_(family.model.parameters(), 0.5)
                    family.optimizer.step()

                    if not family.has_finite_parameters():
                        raise RuntimeError("non-finite model parameters after optimizer step")
        except Exception:
            family.model.load_state_dict(snapshot)
            family.optimizer = optim.Adam(family.model.parameters(), lr=LEARNING_RATE)
            raise

        family.clear_buffer()
        family.train_steps += 1
        family.model_version += 1
        family.last_train_time = time.time() - t0
        self.last_training_at = time.time()
        self.total_train_steps += 1
        self.last_train_error = None

        self._save_family_checkpoint(family_id)
        self.schedule_export(family_id, "train", block=False)

        aliases = self._read_family_aliases(family_id)
        for agent_id in sorted(family.bound_agents):
            aliases_for_agent = self._read_agent_aliases(agent_id)
            self.agents[agent_id].note_model_version(aliases_for_agent.get("latest", family.model_version), family.train_steps)
            self._append_agent_history(agent_id)

        print(
            f"[Train] {family_id}: v{family.model_version} train_steps={family.train_steps} "
            f"bound_agents={len(family.bound_agents)} ({family.last_train_time:.2f}s)"
        )

    # Export + persistence
    def schedule_export(self, family_id: str, reason: str, block: bool = False):
        family = self.get_or_create_family(family_id)
        key = (family_id, family.model_version)
        with self.lock:
            if self._family_export_complete(family_id, family.model_version):
                return
            waiter = self.pending_exports.get(key)
            if waiter is None:
                waiter = threading.Event()
                self.pending_exports[key] = waiter
                self.export_queue.put(ExportRequest(family_id, family.model_version, reason))
        if block:
            waiter.wait(timeout=60)

    def _family_export_complete(self, family_id: str, version: int) -> bool:
        if not self._family_metadata_path(family_id, version).exists():
            return False
        family = self.families.get(family_id)
        if not family:
            return False
        for agent_id in family.bound_agents:
            if not self._agent_policy_path(agent_id, version).exists():
                return False
        return True

    def _save_family_checkpoint(self, family_id: str):
        family = self.families[family_id]
        version_dir = self._family_version_dir(family_id, family.model_version)
        version_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "family_id": family_id,
                "model_state": family.model.state_dict(),
                "optimizer_state": family.optimizer.state_dict(),
                "model_version": family.model_version,
                "schema_version": MODEL_SCHEMA_VERSION,
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "strategy_dim": STRATEGY_DIM,
                "train_steps": family.train_steps,
                "agent_slots": family.agent_slots,
                "archetype_slots": family.archetype_slots,
                "bound_agents": sorted(family.bound_agents),
                "last_eval_report": family.last_eval_report,
            },
            self._family_checkpoint_path(family_id, family.model_version),
        )

    def _write_family_metadata(self, family_id: str, version: int, extra: dict | None = None):
        family = self.families[family_id]
        version_dir = self._family_version_dir(family_id, version)
        version_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "family_id": family_id,
            "model_version": version,
            "schema_version": MODEL_SCHEMA_VERSION,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "strategy_dim": STRATEGY_DIM,
            "updated_at": time.time(),
            "summary": family.get_summary(self._read_family_aliases(family_id)),
        }
        if extra:
            metadata.update(extra)
        self._family_metadata_path(family_id, version).write_text(json.dumps(metadata, indent=2))

        if not self._family_eval_path(family_id, version).exists():
            self._family_eval_path(family_id, version).write_text(
                json.dumps(
                    {
                        "family_id": family_id,
                        "model_version": version,
                        "candidate": version == self.resolve_family_version(family_id, alias="candidate"),
                        "champion": version == self.resolve_family_version(family_id, alias="champion"),
                        "updated_at": time.time(),
                        "reports": [],
                    },
                    indent=2,
                )
            )

    def _write_agent_metadata(self, agent_id: str, version: int):
        binding = self.bindings[agent_id]
        agent = self.agents[agent_id]
        version_dir = self._agent_version_dir(agent_id, version)
        version_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "agent_id": agent_id,
            "policy_family": binding["policy_family"],
            "archetype_id": binding["archetype_id"],
            "strategy": _strategy_from_any(binding.get("strategy"), binding["archetype_id"]),
            "model_version": version,
            "schema_version": MODEL_SCHEMA_VERSION,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "aliases": self._read_agent_aliases(agent_id),
            "family_aliases": self._read_family_aliases(binding["policy_family"]),
            "stats": agent.get_stats(self._read_agent_aliases(agent_id)),
            "updated_at": time.time(),
        }
        self._agent_metadata_path(agent_id, version).write_text(json.dumps(metadata, indent=2))

    def _export_worker(self):
        while True:
            request = self.export_queue.get()
            key = (request.family_id, request.model_version)
            try:
                with self.lock:
                    family = self.families.get(request.family_id)
                    if family is None or family.model_version != request.model_version:
                        continue
                    family_model = copy.deepcopy(family.model).cpu()
                    bound_agents = sorted(family.bound_agents)
                    agent_slots = dict(family.agent_slots)
                    archetype_slots = dict(family.archetype_slots)
                    family_aliases = self._read_family_aliases(request.family_id)

                family_model.eval()
                self._save_family_checkpoint(request.family_id)
                family_aliases["latest"] = request.model_version
                if family_aliases.get("candidate", 0) == 0:
                    family_aliases["candidate"] = request.model_version
                if family_aliases.get("champion", 0) == 0:
                    family_aliases["champion"] = family_aliases.get("candidate", request.model_version)
                self._write_family_aliases(request.family_id, family_aliases)
                self._write_family_metadata(
                    request.family_id,
                    request.model_version,
                    {
                        "aliases": family_aliases,
                        "export_reason": request.reason,
                        "bound_agents": bound_agents,
                    },
                )

                for agent_id in bound_agents:
                    binding = self.ensure_binding(agent_id)
                    strategy_values = _strategy_vector(binding.get("strategy"), binding["archetype_id"])
                    wrapper = BoundActorOnly(
                        copy.deepcopy(family_model),
                        agent_slots[agent_id],
                        archetype_slots[binding["archetype_id"]],
                        strategy_values,
                    )
                    wrapper.eval()
                    policy_path = self._agent_policy_path(agent_id, request.model_version)
                    policy_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path = policy_path.with_suffix(".tmp")
                    dummy = torch.randn(1, STATE_DIM)
                    torch.onnx.export(
                        wrapper,
                        dummy,
                        str(tmp_path),
                        input_names=["state"],
                        output_names=["action"],
                        opset_version=17,
                        dynamo=False,
                    )
                    os.replace(tmp_path, policy_path)

                    agent_aliases = self._read_agent_aliases(agent_id)
                    agent_aliases["latest"] = request.model_version
                    if agent_aliases.get("candidate", 0) == 0:
                        agent_aliases["candidate"] = request.model_version
                    if agent_aliases.get("champion", 0) == 0:
                        agent_aliases["champion"] = agent_aliases.get("candidate", request.model_version)
                    self._write_agent_aliases(agent_id, agent_aliases)
                    self.agents[agent_id].note_model_version(request.model_version, self.families[request.family_id].train_steps)
                    self._write_agent_metadata(agent_id, request.model_version)

                self.last_export_error = None
                self.last_export_at = time.time()
            except Exception as exc:
                self.errors += 1
                self.last_export_error = str(exc)
                print(f"[Trainer] Export failed for {request.family_id}: {exc}")
            finally:
                waiter = self.pending_exports.pop(key, None)
                if waiter:
                    waiter.set()
                self.export_queue.task_done()

    def _append_agent_history(self, agent_id: str):
        agent = self.agents[agent_id]
        payload = {**agent.get_stats(self._read_agent_aliases(agent_id)), "timestamp": time.time()}
        _append_jsonl(self._history_path(agent_id), payload)

    # Migration + loading
    def load_existing_agents(self):
        self._load_families()
        self._load_agents_from_bindings()
        if not self.families:
            self._migrate_legacy_layout()
        for family_id in sorted(self.families):
            self.schedule_export(family_id, "startup", block=False)

    def _load_families(self):
        for family_dir in sorted(POLICIES_DIR.glob("*")):
            versions_dir = family_dir / "versions"
            if not family_dir.is_dir() or not versions_dir.exists():
                continue

            aliases = _read_json(self._family_aliases_path(family_dir.name), self.alias_defaults)
            preferred = int(aliases.get("latest", 0) or 0)
            checkpoint = self._family_checkpoint_path(family_dir.name, preferred) if preferred > 0 else None

            if not checkpoint or not checkpoint.exists():
                checkpoints = sorted(versions_dir.glob("v*/checkpoint.pt"))
                checkpoint = checkpoints[-1] if checkpoints else None

            if not checkpoint or not checkpoint.exists():
                continue

            try:
                self._load_family_checkpoint(checkpoint)
            except Exception as exc:
                print(f"[Trainer] Skipping family checkpoint {checkpoint}: {exc}")

    def _load_family_checkpoint(self, checkpoint_path: Path):
        ckpt = self._load_family_checkpoint_payload(checkpoint_path)
        family_id = ckpt["family_id"]
        family = PolicyFamily(family_id)
        self._hydrate_family_from_checkpoint(family, ckpt)
        self.families[family_id] = family
        self._write_family_metadata(family_id, family.model_version, {"aliases": self._read_family_aliases(family_id)})
        print(f"[Trainer] Loaded family {family_id} (v{family.model_version}, agents={len(family.bound_agents)})")

    def _load_agents_from_bindings(self):
        for binding_path in sorted(AGENTS_DIR.glob("*/binding.json")):
            try:
                binding = json.loads(binding_path.read_text())
            except Exception:
                continue
            agent_id = binding_path.parent.name
            self.bindings[agent_id] = binding
            family = self.get_or_create_family(binding["policy_family"])
            family.bind_agent(agent_id, binding["archetype_id"])
            agent = self.get_or_create_agent(agent_id, binding)
            self._restore_agent_history(agent_id, agent)

    def _restore_agent_history(self, agent_id: str, agent: AgentRecord):
        entries = []
        path = self._history_path(agent_id)
        if path.exists():
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue
        if entries:
            agent.restore(entries[-1])
            return

        aliases = self._read_agent_aliases(agent_id)
        latest = aliases.get("latest", 0)
        metadata_path = self._agent_metadata_path(agent_id, latest)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            agent.restore(metadata.get("stats", {}))

    def _migrate_legacy_layout(self):
        legacy_dirs = [
            path
            for path in MODELS_DIR.iterdir()
            if path.is_dir() and path.name not in {"policies", "agents"}
        ]
        if not legacy_dirs:
            return

        legacy_state_dicts = []
        migrated_agents = 0
        max_version = 0
        for agent_dir in sorted(legacy_dirs):
            checkpoints = sorted((agent_dir / "versions").glob("v*/checkpoint.pt"))
            if not checkpoints:
                continue
            checkpoint_path = checkpoints[-1]
            try:
                ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            except Exception:
                continue
            agent_id = ckpt.get("agent_id", agent_dir.name)
            binding = self.ensure_binding(agent_id)
            agent = self.get_or_create_agent(agent_id, binding)
            agent.restore(
                {
                    "episodes": ckpt.get("total_episodes", 0),
                    "total_steps": ckpt.get("total_steps", 0),
                    "train_steps": ckpt.get("model_version", 0),
                    "model_version": ckpt.get("model_version", 0),
                    "best_score": ckpt.get("best_score", 0),
                    "score_history": ckpt.get("score_history", []),
                    "kd_history": ckpt.get("kd_history", []),
                    "damage_history": ckpt.get("damage_history", []),
                    "last_strategy": ckpt.get("last_strategy", {"strategy": binding["strategy"]}),
                    "last_reward_totals": ckpt.get("last_reward_totals", {}),
                    "policy_family": binding["policy_family"],
                    "archetype_id": binding["archetype_id"],
                }
            )
            legacy_state_dicts.append(ckpt["model_state"])
            max_version = max(max_version, int(ckpt.get("model_version", 0)))
            migrated_agents += 1

        if not legacy_state_dicts:
            return

        family = self.get_or_create_family(DEFAULT_POLICY_FAMILY)
        self._initialize_family_from_legacy(family, legacy_state_dicts)
        family.model_version = max_version
        family.train_steps = max_version
        family.bound_agents = set(self.agents.keys())
        self._save_family_checkpoint(DEFAULT_POLICY_FAMILY)
        aliases = {"latest": max_version, "candidate": max_version, "champion": max_version}
        self._write_family_aliases(DEFAULT_POLICY_FAMILY, aliases)
        for agent_id in self.agents:
            self._write_agent_aliases(agent_id, dict(aliases))
            self.agents[agent_id].note_model_version(max_version, family.train_steps)
            self._append_agent_history(agent_id)
        self.schedule_export(DEFAULT_POLICY_FAMILY, "legacy_migration", block=True)
        print(f"[Trainer] Migrated {migrated_agents} legacy agents into {DEFAULT_POLICY_FAMILY}")

    def _initialize_family_from_legacy(self, family: PolicyFamily, state_dicts: list[dict]):
        def average_tensor(key):
            tensors = [state[key].detach().cpu().float() for state in state_dicts if key in state]
            if not tensors:
                return None
            return torch.stack(tensors, dim=0).mean(dim=0)

        averaged = {
            "shared.0.weight": average_tensor("shared.0.weight"),
            "shared.0.bias": average_tensor("shared.0.bias"),
            "shared.2.weight": average_tensor("shared.2.weight"),
            "shared.2.bias": average_tensor("shared.2.bias"),
            "actor_mean.weight": average_tensor("actor_mean.weight"),
            "actor_mean.bias": average_tensor("actor_mean.bias"),
            "actor_log_std": average_tensor("actor_log_std"),
            "critic.weight": average_tensor("critic.weight"),
            "critic.bias": average_tensor("critic.bias"),
        }

        with torch.no_grad():
            if averaged["shared.0.weight"] is not None:
                family.model.shared[0].weight.zero_()
                family.model.shared[0].weight[:, : STATE_DIM] = averaged["shared.0.weight"]
            if averaged["shared.0.bias"] is not None:
                family.model.shared[0].bias.copy_(averaged["shared.0.bias"])
            if averaged["shared.2.weight"] is not None:
                family.model.shared[2].weight.copy_(averaged["shared.2.weight"])
            if averaged["shared.2.bias"] is not None:
                family.model.shared[2].bias.copy_(averaged["shared.2.bias"])
            if averaged["actor_mean.weight"] is not None:
                family.model.actor_mean.weight.copy_(averaged["actor_mean.weight"])
            if averaged["actor_mean.bias"] is not None:
                family.model.actor_mean.bias.copy_(averaged["actor_mean.bias"])
            if averaged["actor_log_std"] is not None:
                family.model.actor_log_std.copy_(averaged["actor_log_std"])
            if averaged["critic.weight"] is not None:
                family.model.critic.weight.copy_(averaged["critic.weight"])
            if averaged["critic.bias"] is not None:
                family.model.critic.bias.copy_(averaged["critic.bias"])

    # Evaluation + promotion
    def record_evaluation(self, payload: dict):
        family_id = payload["family_id"]
        candidate_version = int(payload.get("candidate_version", self.resolve_family_version(family_id, alias="candidate") or 0))
        payload = {
            **payload,
            "family_id": family_id,
            "candidate_version": candidate_version,
            "recorded_at": time.time(),
        }
        _append_jsonl(EVALUATION_HISTORY, payload)

        family = self.get_or_create_family(family_id)
        family.last_eval_report = payload

        eval_path = self._family_eval_path(family_id, candidate_version)
        existing = _read_json(
            eval_path,
            {
                "family_id": family_id,
                "model_version": candidate_version,
                "candidate": True,
                "champion": candidate_version == self.resolve_family_version(family_id, alias="champion"),
                "updated_at": time.time(),
                "reports": [],
            },
        )
        existing["updated_at"] = time.time()
        existing.setdefault("reports", []).append(payload)
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(json.dumps(existing, indent=2))

    def get_evaluation_history(self, family_id: str | None = None, limit: int = 50):
        if not EVALUATION_HISTORY.exists():
            return []
        entries = []
        for line in EVALUATION_HISTORY.read_text().splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if family_id and item.get("family_id") != family_id:
                continue
            entries.append(item)
        return entries[-limit:]

    def promote_candidate(self, family_id: str, version: int | None = None):
        family = self.get_or_create_family(family_id)
        target = int(version if version is not None else family.model_version)
        aliases = self._read_family_aliases(family_id)
        aliases["candidate"] = target
        aliases["latest"] = max(aliases.get("latest", 0), target)
        self._write_family_aliases(family_id, aliases)
        for agent_id in family.bound_agents:
            agent_aliases = self._read_agent_aliases(agent_id)
            agent_aliases["candidate"] = target
            agent_aliases["latest"] = max(agent_aliases.get("latest", 0), target)
            self._write_agent_aliases(agent_id, agent_aliases)
        return {"ok": True, "family_id": family_id, "aliases": aliases}

    def latest_passed_candidate_version(self, family_id: str) -> int:
        for entry in reversed(self.get_evaluation_history(family_id, limit=250)):
            version = int(entry.get("candidate_version", 0) or 0)
            if entry.get("passed") and version > 0:
                return version
        return 0

    def restore_family_version(
        self,
        family_id: str,
        version: int,
        note: str | None = None,
        update_latest: bool = True,
        update_candidate: bool = True,
    ):
        version = int(version or 0)
        checkpoint_path = self._family_checkpoint_path(family_id, version)
        if version <= 0 or not checkpoint_path.exists():
            return {"ok": False, "error": f"checkpoint not found for {family_id} v{version}"}

        with self.lock:
            ckpt = self._load_family_checkpoint_payload(checkpoint_path)
            family = self.families.get(family_id) or PolicyFamily(family_id)
            self._hydrate_family_from_checkpoint(family, ckpt)
            self.families[family_id] = family

            aliases = self._read_family_aliases(family_id)
            previous_latest = aliases.get("latest", 0)
            previous_candidate = aliases.get("candidate", 0)
            if update_latest:
                aliases["latest"] = version
            if update_candidate:
                aliases["candidate"] = version
            self._write_family_aliases(family_id, aliases)

            for agent_id in sorted(family.bound_agents):
                agent_aliases = self._read_agent_aliases(agent_id)
                if update_latest:
                    agent_aliases["latest"] = version
                if update_candidate:
                    agent_aliases["candidate"] = version
                self._write_agent_aliases(agent_id, agent_aliases)
                binding = self.ensure_binding(agent_id)
                agent = self.get_or_create_agent(agent_id, binding)
                agent.note_model_version(agent_aliases.get("latest", version), family.train_steps)

            _append_jsonl(
                PROMOTION_HISTORY,
                {
                    "action": "restore",
                    "family_id": family_id,
                    "version": version,
                    "previous_latest": previous_latest,
                    "previous_candidate": previous_candidate,
                    "restored_at": time.time(),
                    "note": note,
                },
            )

        self._write_family_metadata(
            family_id,
            version,
            {
                "aliases": self._read_family_aliases(family_id),
                "restore_note": note,
            },
        )
        self.schedule_export(family_id, "restore", block=True)
        return {"ok": True, "family_id": family_id, "version": version, "aliases": self._read_family_aliases(family_id)}

    def approve_champion(self, family_id: str, note: str | None = None):
        family = self.get_or_create_family(family_id)
        aliases = self._read_family_aliases(family_id)
        candidate_version = aliases.get("candidate", 0)
        latest_eval = family.last_eval_report or {}
        if latest_eval.get("candidate_version") != candidate_version or not latest_eval.get("passed"):
            return {"ok": False, "error": "candidate has not passed evaluation gates"}

        aliases["champion"] = candidate_version
        self._write_family_aliases(family_id, aliases)
        payload = {
            "family_id": family_id,
            "candidate_version": candidate_version,
            "approved_at": time.time(),
            "note": note,
        }
        _append_jsonl(PROMOTION_HISTORY, {"action": "approve", **payload})

        for agent_id in family.bound_agents:
            agent_aliases = self._read_agent_aliases(agent_id)
            agent_aliases["champion"] = candidate_version
            self._write_agent_aliases(agent_id, agent_aliases)
        return {"ok": True, "family_id": family_id, "aliases": aliases}

    def reject_champion(self, family_id: str, note: str | None = None):
        aliases = self._read_family_aliases(family_id)
        payload = {
            "family_id": family_id,
            "candidate_version": aliases.get("candidate", 0),
            "rejected_at": time.time(),
            "note": note,
        }
        _append_jsonl(PROMOTION_HISTORY, {"action": "reject", **payload})
        return {"ok": True, **payload}

    # Management
    def reset_agent(self, agent_id: str):
        binding = self.ensure_binding(agent_id)
        self.agents[agent_id] = AgentRecord(agent_id, binding["policy_family"], binding["archetype_id"])
        self.agents[agent_id].apply_binding(binding)
        family = self.get_or_create_family(binding["policy_family"])
        self.agents[agent_id].note_model_version(self._read_agent_aliases(agent_id).get("latest", family.model_version), family.train_steps)
        self._append_agent_history(agent_id)

    def clone_agent(self, target_id: str, source_id: str, mutation=0.02):
        source_binding = self.ensure_binding(source_id)
        target_binding = self.ensure_binding(
            target_id,
            source_binding["policy_family"],
            source_binding["archetype_id"],
            source_binding["strategy"],
        )
        self.agents[target_id] = AgentRecord(target_id, target_binding["policy_family"], target_binding["archetype_id"])
        self.agents[target_id].apply_binding(target_binding)
        family = self.get_or_create_family(target_binding["policy_family"])
        family.bind_agent(target_id, target_binding["archetype_id"])
        self.agents[target_id].note_model_version(self._read_agent_aliases(target_id).get("latest", family.model_version), family.train_steps)
        self._append_agent_history(target_id)
        return True

    def record_strategy(self, agent_id: str, payload: dict):
        binding = self.ensure_binding(agent_id, payload.get("policy_family"), payload.get("archetype_id"), payload.get("strategy"))
        agent = self.get_or_create_agent(agent_id, binding)
        agent.last_strategy = {
            "analysis": payload.get("analysis"),
            "plan": payload.get("plan"),
            "strategy": _strategy_from_any(payload.get("strategy"), binding["archetype_id"]),
            "diff": payload.get("diff", {}),
            "personality": payload.get("personality", {}),
            "updated_at": time.time(),
        }
        version = self.resolve_version(agent_id, alias="latest") or self.get_or_create_family(binding["policy_family"]).model_version
        if version is not None:
            self._write_agent_metadata(agent_id, int(version))

    def get_all_stats(self):
        stats = []
        for agent_id, agent in self.agents.items():
            aliases = self._read_agent_aliases(agent_id)
            family = self.families.get(agent.policy_family)
            if family:
                agent.note_model_version(aliases.get("latest", family.model_version), family.train_steps)
            stats.append(agent.get_stats(aliases))
        stats.sort(key=lambda item: item["avg_score_50"], reverse=True)
        return stats

    def get_family_summaries(self):
        families = []
        for family_id, family in self.families.items():
            aliases = self._read_family_aliases(family_id)
            families.append(
                {
                    **family.get_summary(aliases),
                    "champion_history": [
                        item
                        for item in self.get_promotion_history(family_id, limit=25)
                        if item.get("action") == "approve"
                    ],
                }
            )
        families.sort(key=lambda item: item["family_id"])
        return families

    def get_promotion_history(self, family_id: str | None = None, limit: int = 50):
        if not PROMOTION_HISTORY.exists():
            return []
        entries = []
        for line in PROMOTION_HISTORY.read_text().splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if family_id and item.get("family_id") != family_id:
                continue
            entries.append(item)
        return entries[-limit:]

    def get_system_info(self):
        uptime = time.time() - self.start_time
        return {
            "device": str(DEVICE),
            "uptime_seconds": int(uptime),
            "uptime_human": f"{int(uptime // 3600)}h{int((uptime % 3600) // 60)}m",
            "total_experiences": self.total_experiences,
            "total_train_steps": self.total_train_steps,
            "errors": self.errors,
            "num_agents": len(self.agents),
            "num_families": len(self.families),
            "export_queue_depth": self.export_queue.qsize(),
            "last_export_error": self.last_export_error,
            "last_train_error": self.last_train_error,
            "last_training_at": self.last_training_at,
            "last_export_at": self.last_export_at,
            "default_policy_family": DEFAULT_POLICY_FAMILY,
            "versions": {
                "python": sys.version.split(" ", maxsplit=1)[0],
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
        }

    def is_ready(self):
        if not self.export_thread.is_alive():
            return False
        if self.last_export_error or self.last_train_error:
            return False
        now = time.time()
        if self.export_queue.qsize() > 0 and self.last_export_at and now - self.last_export_at > EXPORT_STALL_SECONDS:
            return False
        if any(family.has_enough_experience() for family in self.families.values()):
            if self.last_training_at is None:
                return False
            if now - self.last_training_at > TRAIN_STALL_SECONDS:
                return False
        return True

    def wait_for_model(self, agent_id: str, alias: str | None = None, version: str | None = None, family_hint=None, archetype_hint=None, timeout: float = 30.0):
        binding = self.ensure_binding(agent_id, family_hint, archetype_hint)
        family = self.get_or_create_family(binding["policy_family"])
        resolved = self.resolve_version(agent_id, alias=alias, version=version)
        if resolved is None:
            resolved = family.model_version
            aliases = self._read_agent_aliases(agent_id)
            aliases["latest"] = resolved
            if aliases.get("candidate", 0) == 0:
                aliases["candidate"] = resolved
            self._write_agent_aliases(agent_id, aliases)
        key = (binding["policy_family"], resolved)
        if self._agent_policy_path(agent_id, resolved).exists():
            return resolved
        self.schedule_export(binding["policy_family"], "serve", block=False)
        waiter = self.pending_exports.get(key)
        if waiter:
            waiter.wait(timeout=timeout)
        return resolved if self._agent_policy_path(agent_id, resolved).exists() else None


def create_app(trainer: PPOTrainer):
    from flask import Flask, Response, jsonify, request, send_file

    app = Flask(__name__)

    @app.after_request
    def add_cors(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    @app.route("/experience", methods=["POST"])
    def receive_experience():
        data = request.get_json() or {}
        agent_id = data["agent_id"]
        accepted = trainer.process_experience(
            agent_id,
            data.get("transitions", []),
            policy_family=data.get("policy_family"),
            archetype_id=data.get("archetype_id"),
            strategy_vector=data.get("strategy_vector"),
        )
        agent = trainer.agents.get(agent_id)
        return jsonify(
            {
                "ok": True,
                "accepted": accepted,
                "model_version": agent.model_version if agent else 0,
            }
        )

    @app.route("/episode", methods=["POST"])
    def record_episode():
        data = request.get_json() or {}
        binding = trainer.ensure_binding(
            data["agent_id"],
            data.get("policy_family"),
            data.get("archetype_id"),
            data.get("strategy_vector"),
        )
        agent = trainer.get_or_create_agent(data["agent_id"], binding)
        agent.record_episode(data)
        trainer._append_agent_history(agent.agent_id)
        return jsonify({"ok": True})

    @app.route("/model/<agent_id>", methods=["GET"])
    def get_model(agent_id):
        resolved = trainer.wait_for_model(
            agent_id,
            alias=request.args.get("alias"),
            version=request.args.get("version"),
            family_hint=request.args.get("policy_family"),
            archetype_hint=request.args.get("archetype_id"),
        )
        if resolved is None:
            return jsonify({"error": "Model not ready"}), 503
        return send_file(
            str(trainer._agent_policy_path(agent_id, resolved).resolve()),
            mimetype="application/octet-stream",
        )

    @app.route("/model/<agent_id>/version", methods=["GET"])
    def get_model_version(agent_id):
        trainer.ensure_binding(
            agent_id,
            request.args.get("policy_family"),
            request.args.get("archetype_id"),
        )
        resolved = trainer.resolve_version(
            agent_id,
            alias=request.args.get("alias"),
            version=request.args.get("version"),
        )
        return jsonify({"version": resolved if resolved is not None else 0})

    @app.route("/model/<agent_id>/metadata", methods=["GET"])
    def get_model_metadata(agent_id):
        trainer.ensure_binding(
            agent_id,
            request.args.get("policy_family"),
            request.args.get("archetype_id"),
        )
        resolved = trainer.resolve_version(
            agent_id,
            alias=request.args.get("alias"),
            version=request.args.get("version"),
        )
        if resolved is None:
            return jsonify({"error": "Unknown model"}), 404
        metadata_path = trainer._agent_metadata_path(agent_id, resolved)
        if not metadata_path.exists():
            return jsonify({"error": "Metadata not found"}), 404
        return jsonify(json.loads(metadata_path.read_text()))

    @app.route("/model/<agent_id>/aliases", methods=["GET"])
    def get_model_aliases(agent_id):
        trainer.ensure_binding(
            agent_id,
            request.args.get("policy_family"),
            request.args.get("archetype_id"),
        )
        return jsonify(trainer._read_agent_aliases(agent_id))

    @app.route("/model/<agent_id>/alias/<alias>", methods=["POST"])
    def set_model_alias(agent_id, alias):
        version = int((request.get_json() or {}).get("version", 0))
        aliases = trainer._read_agent_aliases(agent_id)
        aliases[alias] = version
        trainer._write_agent_aliases(agent_id, aliases)
        return jsonify({"ok": True, "aliases": aliases})

    @app.route("/families", methods=["GET"])
    def list_families():
        return jsonify(trainer.get_family_summaries())

    @app.route("/family/<family_id>/status", methods=["GET"])
    def family_status(family_id):
        family = trainer.get_or_create_family(family_id)
        return jsonify(
            {
                **family.get_summary(trainer._read_family_aliases(family_id)),
                "champion_history": trainer.get_promotion_history(family_id, limit=25),
                "evaluation_history": trainer.get_evaluation_history(family_id, limit=25),
            }
        )

    @app.route("/family/<family_id>/bindings", methods=["GET"])
    def family_bindings(family_id):
        bindings = [
            binding
            for binding in trainer.bindings.values()
            if binding.get("policy_family") == family_id
        ]
        bindings.sort(key=lambda item: item["agent_id"])
        return jsonify(bindings)

    @app.route("/eval/report", methods=["POST"])
    def record_eval_report():
        trainer.record_evaluation(request.get_json() or {})
        return jsonify({"ok": True})

    @app.route("/eval/history", methods=["GET"])
    def evaluation_history():
        limit = max(1, min(200, int(request.args.get("limit", "50"))))
        return jsonify(trainer.get_evaluation_history(request.args.get("family_id"), limit))

    @app.route("/promotion/candidate/<family_id>", methods=["POST"])
    def candidate_promotion(family_id):
        body = request.get_json(silent=True) or {}
        result = trainer.promote_candidate(family_id, body.get("version"))
        return jsonify(result)

    @app.route("/promotion/champion/<family_id>/approve", methods=["POST"])
    def champion_approve(family_id):
        body = request.get_json(silent=True) or {}
        result = trainer.approve_champion(family_id, body.get("note"))
        return jsonify(result), (200 if result.get("ok") else 409)

    @app.route("/promotion/champion/<family_id>/reject", methods=["POST"])
    def champion_reject(family_id):
        body = request.get_json(silent=True) or {}
        return jsonify(trainer.reject_champion(family_id, body.get("note")))

    @app.route("/family/<family_id>/restore", methods=["POST"])
    def restore_family(family_id):
        body = request.get_json(silent=True) or {}
        version = (
            body.get("version")
            or trainer.latest_passed_candidate_version(family_id)
            or trainer.resolve_family_version(family_id, alias="champion")
        )
        result = trainer.restore_family_version(
            family_id,
            version,
            note=body.get("note"),
            update_latest=body.get("update_latest", True),
            update_candidate=body.get("update_candidate", True),
        )
        return jsonify(result), (200 if result.get("ok") else 404)

    @app.route("/stats", methods=["GET"])
    def get_stats():
        return jsonify({**trainer.get_system_info(), "agents": trainer.get_all_stats(), "families": trainer.get_family_summaries()})

    @app.route("/select", methods=["POST"])
    def legacy_select():
        family_id = (request.get_json(silent=True) or {}).get("family_id", DEFAULT_POLICY_FAMILY)
        result = trainer.promote_candidate(family_id)
        return jsonify({"ok": True, "legacy": True, **result, "agents": trainer.get_all_stats(), "families": trainer.get_family_summaries()})

    @app.route("/agent/<agent_id>/reset", methods=["POST"])
    def reset_agent(agent_id):
        trainer.reset_agent(agent_id)
        return jsonify({"ok": True})

    @app.route("/agent/<target_id>/clone/<source_id>", methods=["POST"])
    def clone_agent(target_id, source_id):
        data = request.get_json(silent=True) or {}
        ok = trainer.clone_agent(target_id, source_id, data.get("mutation", 0.02))
        return jsonify({"ok": ok})

    @app.route("/agent/<agent_id>/history", methods=["GET"])
    def agent_history(agent_id):
        path = trainer._history_path(agent_id)
        if not path.exists():
            return jsonify([])
        entries = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
        return jsonify(entries[-100:])

    @app.route("/agent/<agent_id>/strategy", methods=["POST"])
    def record_strategy(agent_id):
        trainer.record_strategy(agent_id, request.get_json() or {})
        return jsonify({"ok": True})

    @app.route("/health", methods=["GET"])
    @app.route("/healthz", methods=["GET"])
    def health():
        return jsonify({"status": "ok", **trainer.get_system_info()})

    @app.route("/readyz", methods=["GET"])
    def ready():
        ready_flag = trainer.is_ready()
        return (
            jsonify({"ok": ready_flag, **trainer.get_system_info()}),
            200 if ready_flag else 503,
        )

    @app.route("/metrics", methods=["GET"])
    def metrics():
        info = trainer.get_system_info()
        lines = [
            "# TYPE chainer_trainer_up gauge",
            "chainer_trainer_up 1",
            "# TYPE chainer_trainer_ready gauge",
            f"chainer_trainer_ready {1 if trainer.is_ready() else 0}",
            f"chainer_trainer_total_experiences {info['total_experiences']}",
            f"chainer_trainer_total_train_steps {info['total_train_steps']}",
            f"chainer_trainer_errors {info['errors']}",
            f"chainer_trainer_export_queue_depth {info['export_queue_depth']}",
            f"chainer_trainer_num_families {info['num_families']}",
        ]
        return Response("\n".join(lines) + "\n", mimetype="text/plain")

    return app


if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.load_existing_agents()
    app = create_app(trainer)
    port = int(os.environ.get("TRAINER_PORT", 5555))
    print(f"[Trainer] Starting on port {port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
