"""
Chainer Agent — PPO Training Service

Stable single-process trainer with:
  - per-agent PPO policies
  - filesystem-backed model registry with aliases
  - single-threaded ONNX export queue
  - readiness/metrics endpoints
"""

from __future__ import annotations

import copy
import json
import math
import os
import queue
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
HIDDEN_DIM = 128
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
PPO_EPOCHS = 4
MODEL_SCHEMA_VERSION = 2
MODELS_DIR = Path("models")
LOGS_DIR = Path("training_logs")
DEVICE = torch.device("cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        shared = self.shared(state)
        return self.actor_mean(shared), self.actor_log_std, self.critic(shared)


class ActorOnly(nn.Module):
    def __init__(self, ac: ActorCritic):
        super().__init__()
        self.shared = ac.shared
        self.actor_mean = ac.actor_mean

    def forward(self, state):
        return torch.tanh(self.actor_mean(self.shared(state)))


class Agent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.model = ActorCritic()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.states: list[list[float]] = []
        self.actions: list[list[float]] = []
        self.rewards: list[float] = []
        self.reward_components: list[dict] = []
        self.values: list[float] = []
        self.log_probs: list[float] = []
        self.dones: list[float] = []
        self.total_episodes = 0
        self.total_steps = 0
        self.best_score = 0.0
        self.train_steps = 0
        self.last_train_time = 0.0
        self.score_history: list[float] = []
        self.kd_history: list[float] = []
        self.damage_history: list[float] = []
        self.model_version = 0
        self.created_at = time.time()
        self.last_strategy = {}
        self.last_reward_totals = {}

    def add_experience(self, state, action, reward, reward_components, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.reward_components.append(reward_components or {})
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.dones.append(float(done))
        self.total_steps += 1

    def has_enough_experience(self):
        return len(self.states) >= BATCH_SIZE

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

    def clear_buffer(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.reward_components.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def get_stats(self, aliases: dict | None = None):
        avg_score = float(np.mean(self.score_history)) if self.score_history else 0.0
        avg_kd = float(np.mean(self.kd_history)) if self.kd_history else 0.0
        avg_damage = float(np.mean(self.damage_history)) if self.damage_history else 0.0
        return {
            "agent_id": self.agent_id,
            "episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "train_steps": self.train_steps,
            "model_version": self.model_version,
            "best_score": round(self.best_score, 1),
            "avg_score_50": round(avg_score, 1),
            "avg_kd_50": round(avg_kd, 2),
            "avg_damage_50": round(avg_damage, 1),
            "buffer_size": len(self.states),
            "score_history": self.score_history[-20:],
            "kd_history": [round(v, 2) for v in self.kd_history[-20:]],
            "aliases": aliases or {},
            "last_strategy": self.last_strategy,
            "last_reward_totals": self.last_reward_totals,
        }


@dataclass
class ExportRequest:
    agent_id: str
    model_version: int
    reason: str


def _is_valid_float_list(lst, expected_len):
    if not isinstance(lst, list) or len(lst) != expected_len:
        return False
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in lst)


class PPOTrainer:
    def __init__(self):
        self.agents: dict[str, Agent] = {}
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
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(1)
        self.export_thread = threading.Thread(target=self._export_worker, daemon=True)
        self.export_thread.start()
        print(f"[Trainer] Device: {DEVICE} (stable CPU training)")

    # Registry helpers
    def _agent_dir(self, agent_id: str) -> Path:
        return MODELS_DIR / agent_id

    def _versions_dir(self, agent_id: str) -> Path:
        return self._agent_dir(agent_id) / "versions"

    def _version_dir(self, agent_id: str, version: int) -> Path:
        return self._versions_dir(agent_id) / f"v{version:06d}"

    def _aliases_path(self, agent_id: str) -> Path:
        return self._agent_dir(agent_id) / "aliases.json"

    def _history_path(self, agent_id: str) -> Path:
        return LOGS_DIR / f"{agent_id}.jsonl"

    def _read_aliases(self, agent_id: str) -> dict:
        path = self._aliases_path(agent_id)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        return dict(self.alias_defaults)

    def _write_aliases(self, agent_id: str, aliases: dict):
        agent_dir = self._agent_dir(agent_id)
        agent_dir.mkdir(parents=True, exist_ok=True)
        self._aliases_path(agent_id).write_text(json.dumps(aliases, indent=2))

    def set_alias(self, agent_id: str, alias: str, version: int):
        aliases = self._read_aliases(agent_id)
        aliases[alias] = int(version)
        self._write_aliases(agent_id, aliases)

    def resolve_version(self, agent_id: str, alias: str | None = None, version: str | None = None) -> int | None:
        if version is not None:
            if isinstance(version, str) and version.startswith("v"):
                version = version[1:]
            return int(version)
        aliases = self._read_aliases(agent_id)
        chosen = alias or "latest"
        return aliases.get(chosen)

    def _policy_path(self, agent_id: str, version: int) -> Path:
        return self._version_dir(agent_id, version) / "policy.onnx"

    def _checkpoint_path(self, agent_id: str, version: int) -> Path:
        return self._version_dir(agent_id, version) / "checkpoint.pt"

    def _metadata_path(self, agent_id: str, version: int) -> Path:
        return self._version_dir(agent_id, version) / "metadata.json"

    def _eval_path(self, agent_id: str, version: int) -> Path:
        return self._version_dir(agent_id, version) / "eval.json"

    def _write_metadata(self, agent: Agent, version: int, extra: dict | None = None):
        version_dir = self._version_dir(agent.agent_id, version)
        version_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "agent_id": agent.agent_id,
            "model_version": version,
            "schema_version": MODEL_SCHEMA_VERSION,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "updated_at": time.time(),
            "stats": agent.get_stats(self._read_aliases(agent.agent_id)),
        }
        if extra:
            metadata.update(extra)
        self._metadata_path(agent.agent_id, version).write_text(json.dumps(metadata, indent=2))
        if not self._eval_path(agent.agent_id, version).exists():
            self._eval_path(agent.agent_id, version).write_text(json.dumps({
                "candidate": version == self.resolve_version(agent.agent_id, alias="candidate"),
                "champion": version == self.resolve_version(agent.agent_id, alias="champion"),
                "updated_at": time.time(),
            }, indent=2))

    # Lifecycle
    def get_or_create_agent(self, agent_id: str) -> Agent:
        created = False
        with self.lock:
            agent = self.agents.get(agent_id)
            if agent:
                return agent
            agent = Agent(agent_id)
            self.agents[agent_id] = agent
            self._write_aliases(agent_id, dict(self.alias_defaults))
            created = True
        if created:
            self.schedule_export(agent_id, "bootstrap", block=True)
            print(f"[Trainer] Created agent: {agent_id}")
        return agent

    def process_experience(self, agent_id: str, transitions: list):
        agent = self.get_or_create_agent(agent_id)
        accepted = 0

        for transition in transitions:
            state = transition.get("state")
            action = transition.get("action")
            if not _is_valid_float_list(state, STATE_DIM):
                continue
            if not _is_valid_float_list(action, ACTION_DIM):
                continue

            reward = float(transition.get("reward", 0))
            if not math.isfinite(reward):
                reward = 0.0
            reward = max(-20.0, min(20.0, reward))

            done = 1.0 if transition.get("done") else 0.0

            state_t = torch.FloatTensor(state).unsqueeze(0)
            action_t = torch.FloatTensor(action).unsqueeze(0)
            with torch.no_grad():
                action_mean, log_std, value = agent.model(state_t)
                std = log_std.exp().clamp(min=1e-6)
                dist = torch.distributions.Normal(action_mean, std)
                log_prob = dist.log_prob(action_t).sum(-1)

            agent.add_experience(
                state,
                action,
                reward,
                transition.get("reward_components", {}),
                value.item(),
                log_prob.item(),
                done,
            )
            accepted += 1

        self.total_experiences += accepted

        if agent.has_enough_experience():
            try:
                self._train(agent_id)
            except Exception as exc:
                self.errors += 1
                self.last_train_error = str(exc)
                print(f"[Trainer] Training failed for {agent_id}: {exc}")
                agent.clear_buffer()

        return accepted

    def _train(self, agent_id: str):
        agent = self.agents[agent_id]
        t0 = time.time()

        states = torch.FloatTensor(np.array(agent.states))
        actions = torch.FloatTensor(np.array(agent.actions))
        old_log_probs = torch.FloatTensor(agent.log_probs)
        rewards = list(agent.rewards)
        values = list(agent.values)
        dones = list(agent.dones)

        advantages = [0.0] * len(rewards)
        gae = 0.0
        for idx in reversed(range(len(rewards))):
            next_value = values[idx + 1] if idx < len(rewards) - 1 else 0.0
            delta = rewards[idx] + GAMMA * next_value * (1 - dones[idx]) - values[idx]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[idx]) * gae
            advantages[idx] = gae

        advantages_t = torch.FloatTensor(advantages)
        returns = advantages_t + torch.FloatTensor(values)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        for _ in range(PPO_EPOCHS):
            indices = torch.randperm(len(states))
            for start in range(0, len(states), 64):
                idx = indices[start:start + 64]
                b_states = states[idx]
                b_actions = actions[idx]
                b_old_lp = old_log_probs[idx]
                b_adv = advantages_t[idx]
                b_ret = returns[idx]

                action_mean, log_std, value = agent.model(b_states)
                std = log_std.exp().clamp(min=1e-6)
                dist = torch.distributions.Normal(action_mean, std)
                new_lp = dist.log_prob(b_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = (new_lp - b_old_lp).exp()
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * b_adv
                value_loss = (b_ret - value.squeeze(-1)).pow(2).mean()
                loss = -torch.min(surr1, surr2).mean() + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite PPO loss")

                agent.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.model.parameters(), 0.5)
                agent.optimizer.step()

        agent.clear_buffer()
        agent.train_steps += 1
        agent.model_version += 1
        agent.last_train_time = time.time() - t0
        self.last_training_at = time.time()
        self.total_train_steps += 1

        self._save_checkpoint(agent_id)
        self.schedule_export(agent_id, "train", block=False)

        stats = agent.get_stats(self._read_aliases(agent_id))
        print(
            f"[Train] {agent_id}: v{agent.model_version} score={stats['avg_score_50']} "
            f"kd={stats['avg_kd_50']} ep={agent.total_episodes} ({agent.last_train_time:.2f}s)"
        )
        log_entry = {**stats, "timestamp": time.time()}
        with self._history_path(agent_id).open("a") as handle:
            handle.write(json.dumps(log_entry) + "\n")

    # Registry + export
    def schedule_export(self, agent_id: str, reason: str, block: bool = False):
        agent = self.get_or_create_agent(agent_id)
        key = (agent_id, agent.model_version)
        with self.lock:
            if self._policy_path(agent_id, agent.model_version).exists():
                return
            waiter = self.pending_exports.get(key)
            if waiter is None:
                waiter = threading.Event()
                self.pending_exports[key] = waiter
                self.export_queue.put(ExportRequest(agent_id, agent.model_version, reason))
        if block:
            waiter.wait(timeout=30)

    def _export_worker(self):
        while True:
            request = self.export_queue.get()
            key = (request.agent_id, request.model_version)
            try:
                with self.lock:
                    agent = self.agents.get(request.agent_id)
                    if agent is None:
                        continue
                    if agent.model_version != request.model_version:
                        continue
                    actor = ActorOnly(copy.deepcopy(agent.model))
                    aliases = self._read_aliases(request.agent_id)
                actor.eval()
                version_dir = self._version_dir(request.agent_id, request.model_version)
                version_dir.mkdir(parents=True, exist_ok=True)
                policy_path = self._policy_path(request.agent_id, request.model_version)
                tmp_path = policy_path.with_suffix(".tmp")
                dummy = torch.randn(1, STATE_DIM)

                torch.onnx.export(
                    actor,
                    dummy,
                    str(tmp_path),
                    input_names=["state"],
                    output_names=["action"],
                    opset_version=17,
                    dynamo=False,
                )
                os.replace(tmp_path, policy_path)
                self._write_metadata(agent, request.model_version, {
                    "aliases": aliases,
                    "export_reason": request.reason,
                })

                aliases["latest"] = request.model_version
                aliases["candidate"] = request.model_version
                if aliases.get("champion", 0) == 0 and request.model_version == 0:
                    aliases["champion"] = 0
                self._write_aliases(request.agent_id, aliases)

                self.last_export_error = None
                self.last_export_at = time.time()
            except Exception as exc:
                self.errors += 1
                self.last_export_error = str(exc)
                print(f"[Trainer] Export failed for {request.agent_id}: {exc}")
            finally:
                waiter = self.pending_exports.pop(key, None)
                if waiter:
                    waiter.set()
                self.export_queue.task_done()

    def _save_checkpoint(self, agent_id: str):
        agent = self.agents[agent_id]
        version_dir = self._version_dir(agent_id, agent.model_version)
        version_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": agent.model.state_dict(),
                "optimizer_state": agent.optimizer.state_dict(),
                "agent_id": agent_id,
                "model_version": agent.model_version,
                "schema_version": MODEL_SCHEMA_VERSION,
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "total_episodes": agent.total_episodes,
                "total_steps": agent.total_steps,
                "best_score": agent.best_score,
                "score_history": agent.score_history,
                "kd_history": agent.kd_history,
                "damage_history": agent.damage_history,
                "last_strategy": agent.last_strategy,
                "last_reward_totals": agent.last_reward_totals,
            },
            self._checkpoint_path(agent_id, agent.model_version),
        )

    def load_existing_agents(self):
        for checkpoint in sorted(MODELS_DIR.glob("*/versions/v*/checkpoint.pt")):
            try:
                self.load_checkpoint(checkpoint)
            except Exception as exc:
                print(f"[Trainer] Skipping incompatible checkpoint {checkpoint}: {exc}")

        legacy_checkpoints = MODELS_DIR / "checkpoints"
        if legacy_checkpoints.exists():
            for checkpoint in sorted(legacy_checkpoints.glob("*.pt")):
                try:
                    self.load_checkpoint(checkpoint)
                except Exception:
                    continue

    def load_checkpoint(self, checkpoint_path: Path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        if ckpt.get("schema_version") not in (MODEL_SCHEMA_VERSION, None):
            raise RuntimeError("checkpoint schema mismatch")
        if ckpt.get("state_dim", STATE_DIM) != STATE_DIM:
            raise RuntimeError("state dimension mismatch")
        if ckpt.get("action_dim", ACTION_DIM) != ACTION_DIM:
            raise RuntimeError("action dimension mismatch")

        agent_id = ckpt["agent_id"]
        agent = self.agents.get(agent_id) or Agent(agent_id)
        agent.model.load_state_dict(ckpt["model_state"])
        agent.optimizer.load_state_dict(ckpt["optimizer_state"])
        agent.model_version = int(ckpt.get("model_version", 0))
        agent.total_episodes = int(ckpt.get("total_episodes", 0))
        agent.total_steps = int(ckpt.get("total_steps", 0))
        agent.best_score = float(ckpt.get("best_score", 0))
        agent.score_history = list(ckpt.get("score_history", []))
        agent.kd_history = list(ckpt.get("kd_history", []))
        agent.damage_history = list(ckpt.get("damage_history", []))
        agent.last_strategy = dict(ckpt.get("last_strategy", {}))
        agent.last_reward_totals = dict(ckpt.get("last_reward_totals", {}))
        self.agents[agent_id] = agent
        self._write_metadata(agent, agent.model_version, {"aliases": self._read_aliases(agent_id)})
        print(f"[Trainer] Loaded {agent_id} (v{agent.model_version}, {agent.total_episodes} ep)")

    # Management
    def get_all_stats(self):
        stats = [
            agent.get_stats(self._read_aliases(agent.agent_id))
            for agent in self.agents.values()
        ]
        stats.sort(key=lambda item: item["avg_score_50"], reverse=True)
        return stats

    def natural_selection(self, num_cull=5):
        with self.lock:
            if len(self.agents) < num_cull * 2:
                return {"error": f"Not enough agents ({len(self.agents)})"}

            ranked = sorted(
                self.agents.values(),
                key=lambda agent: (
                    (np.mean(agent.score_history) if agent.score_history else 0)
                    + (np.mean(agent.kd_history) * 100 if agent.kd_history else 0)
                ),
                reverse=True,
            )

            strong = ranked[:num_cull]
            weak = ranked[-num_cull:]
            results = {"strong": [], "weak": [], "clones": []}

            for agent in strong:
                score = np.mean(agent.score_history) if agent.score_history else 0
                kd = np.mean(agent.kd_history) if agent.kd_history else 0
                results["strong"].append({"id": agent.agent_id, "score": round(score), "kd": round(kd, 2)})

            for index, weak_agent in enumerate(weak):
                parent = strong[index % len(strong)]
                child = Agent(weak_agent.agent_id)
                child.model.load_state_dict(parent.model.state_dict())
                with torch.no_grad():
                    for param in child.model.parameters():
                        param.add_(torch.randn_like(param) * 0.02)
                child.optimizer = optim.Adam(child.model.parameters(), lr=LEARNING_RATE)
                child.model_version = parent.model_version + 1
                child.last_strategy = dict(parent.last_strategy)
                self.agents[weak_agent.agent_id] = child
                self._save_checkpoint(weak_agent.agent_id)
                self.schedule_export(weak_agent.agent_id, "selection", block=False)
                self.set_alias(weak_agent.agent_id, "latest", child.model_version)
                self.set_alias(weak_agent.agent_id, "candidate", child.model_version)
                results["clones"].append({"child": weak_agent.agent_id, "parent": parent.agent_id})

            with (LOGS_DIR / "selection.jsonl").open("a") as handle:
                handle.write(json.dumps({"timestamp": time.time(), **results}) + "\n")

            return results

    def reset_agent(self, agent_id: str):
        with self.lock:
            self.agents[agent_id] = Agent(agent_id)
            self._write_aliases(agent_id, dict(self.alias_defaults))
            self._save_checkpoint(agent_id)
            self.schedule_export(agent_id, "reset", block=False)

    def clone_agent(self, target_id: str, source_id: str, mutation=0.02):
        with self.lock:
            source = self.agents.get(source_id)
            if not source:
                return False
            target = Agent(target_id)
            target.model.load_state_dict(source.model.state_dict())
            with torch.no_grad():
                for param in target.model.parameters():
                    param.add_(torch.randn_like(param) * mutation)
            target.optimizer = optim.Adam(target.model.parameters(), lr=LEARNING_RATE)
            target.model_version = source.model_version + 1
            target.last_strategy = dict(source.last_strategy)
            self.agents[target_id] = target
            self._save_checkpoint(target_id)
            self.schedule_export(target_id, "clone", block=False)
            self.set_alias(target_id, "latest", target.model_version)
            self.set_alias(target_id, "candidate", target.model_version)
            return True

    def record_strategy(self, agent_id: str, payload: dict):
        agent = self.get_or_create_agent(agent_id)
        agent.last_strategy = {
            "analysis": payload.get("analysis"),
            "plan": payload.get("plan"),
            "strategy": payload.get("strategy", {}),
            "diff": payload.get("diff", {}),
            "personality": payload.get("personality", {}),
            "updated_at": time.time(),
        }
        version = self.resolve_version(agent_id, alias="latest") or agent.model_version
        self._write_metadata(agent, version, {"llm": agent.last_strategy, "aliases": self._read_aliases(agent_id)})

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
            "export_queue_depth": self.export_queue.qsize(),
            "last_export_error": self.last_export_error,
            "last_train_error": self.last_train_error,
            "last_training_at": self.last_training_at,
            "last_export_at": self.last_export_at,
        }

    def is_ready(self):
        return self.export_thread.is_alive() and self.last_export_error is None

    def wait_for_model(self, agent_id: str, alias: str | None = None, version: str | None = None, timeout: float = 30.0):
        resolved = self.resolve_version(agent_id, alias=alias, version=version)
        if resolved is None:
            return None
        key = (agent_id, resolved)
        if self._policy_path(agent_id, resolved).exists():
            return resolved
        self.schedule_export(agent_id, "serve", block=False)
        waiter = self.pending_exports.get(key)
        if waiter:
            waiter.wait(timeout=timeout)
        return resolved if self._policy_path(agent_id, resolved).exists() else None


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
        data = request.get_json()
        agent_id = data["agent_id"]
        accepted = trainer.process_experience(agent_id, data["transitions"])
        agent = trainer.agents.get(agent_id)
        return jsonify({
            "ok": True,
            "accepted": accepted,
            "model_version": agent.model_version if agent else 0,
        })

    @app.route("/episode", methods=["POST"])
    def record_episode():
        data = request.get_json()
        agent = trainer.get_or_create_agent(data["agent_id"])
        agent.record_episode(data)
        return jsonify({"ok": True})

    @app.route("/model/<agent_id>", methods=["GET"])
    def get_model(agent_id):
        trainer.get_or_create_agent(agent_id)
        alias = request.args.get("alias")
        version = request.args.get("version")
        resolved = trainer.wait_for_model(agent_id, alias=alias, version=version)
        if resolved is None:
            return jsonify({"error": "Model not ready"}), 503
        return send_file(str(trainer._policy_path(agent_id, resolved).resolve()), mimetype="application/octet-stream")

    @app.route("/model/<agent_id>/version", methods=["GET"])
    def get_model_version(agent_id):
        resolved = trainer.resolve_version(
            agent_id,
            alias=request.args.get("alias"),
            version=request.args.get("version"),
        )
        return jsonify({"version": resolved if resolved is not None else 0})

    @app.route("/model/<agent_id>/metadata", methods=["GET"])
    def get_model_metadata(agent_id):
        resolved = trainer.resolve_version(
            agent_id,
            alias=request.args.get("alias"),
            version=request.args.get("version"),
        )
        if resolved is None:
            return jsonify({"error": "Unknown model"}), 404
        metadata_path = trainer._metadata_path(agent_id, resolved)
        if not metadata_path.exists():
            return jsonify({"error": "Metadata not found"}), 404
        return jsonify(json.loads(metadata_path.read_text()))

    @app.route("/model/<agent_id>/aliases", methods=["GET"])
    def get_model_aliases(agent_id):
        return jsonify(trainer._read_aliases(agent_id))

    @app.route("/model/<agent_id>/alias/<alias>", methods=["POST"])
    def set_model_alias(agent_id, alias):
        version = int((request.get_json() or {}).get("version", 0))
        trainer.set_alias(agent_id, alias, version)
        return jsonify({"ok": True, "aliases": trainer._read_aliases(agent_id)})

    @app.route("/stats", methods=["GET"])
    def get_stats():
        return jsonify({**trainer.get_system_info(), "agents": trainer.get_all_stats()})

    @app.route("/select", methods=["POST"])
    def run_selection():
        data = request.get_json() or {}
        result = trainer.natural_selection(data.get("num_cull", 5))
        return jsonify({"ok": True, **result, "agents": trainer.get_all_stats()})

    @app.route("/agent/<agent_id>/reset", methods=["POST"])
    def reset_agent(agent_id):
        trainer.reset_agent(agent_id)
        return jsonify({"ok": True})

    @app.route("/agent/<target_id>/clone/<source_id>", methods=["POST"])
    def clone_agent(target_id, source_id):
        data = request.get_json() or {}
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
        ready = trainer.is_ready()
        return (
            jsonify({"ok": ready, **trainer.get_system_info()}),
            200 if ready else 503,
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
