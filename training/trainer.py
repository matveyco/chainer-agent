"""
Chainer Agent — PPO Training Service (Robust)

Per-agent policy networks trained via Proximal Policy Optimization.
Each agent has its own persistent neural network, experience buffer, and identity.

Exposes HTTP API:
  POST /experience          — receive experience batch from Node.js bot
  GET  /model/:agent_id     — serve ONNX model for inference
  GET  /stats               — training stats
  POST /select              — trigger natural selection
  POST /agent/:id/reset     — reset an agent
  POST /agent/:id/clone/:src — clone source into target
  GET  /health              — health check
"""

import os
import json
import math
import time
import traceback
import threading
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────

STATE_DIM = 18
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
MODELS_DIR = Path("models")
LOGS_DIR = Path("training_logs")

# Force CPU — our models are tiny (20K params), training takes <10ms per batch.
# CPU avoids ALL CUDA inplace/device/HPU backend crashes.
DEVICE = torch.device("cpu")


# ── Neural Network ──────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
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


# ── Agent ───────────────────────────────────────────────────────────────

class Agent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.model = ActorCritic()        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.states, self.actions, self.rewards = [], [], []
        self.values, self.log_probs, self.dones = [], [], []
        self.total_episodes = 0
        self.total_steps = 0
        self.best_score = 0
        self.train_steps = 0
        self.last_train_time = 0
        self.score_history = []
        self.kd_history = []
        self.model_version = 0
        self.created_at = time.time()

    def add_experience(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.dones.append(float(done))
        self.total_steps += 1

    def has_enough_experience(self):
        return len(self.states) >= BATCH_SIZE

    def record_episode(self, score, kills, deaths):
        self.total_episodes += 1
        self.best_score = max(self.best_score, score)
        self.score_history.append(score)
        if len(self.score_history) > 50:
            self.score_history.pop(0)
        kd = kills / max(deaths, 1)
        self.kd_history.append(kd)
        if len(self.kd_history) > 50:
            self.kd_history.pop(0)

    def clear_buffer(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def get_stats(self):
        avg_score = float(np.mean(self.score_history)) if self.score_history else 0
        avg_kd = float(np.mean(self.kd_history)) if self.kd_history else 0
        return {
            "agent_id": self.agent_id,
            "episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "train_steps": self.train_steps,
            "model_version": self.model_version,
            "best_score": round(self.best_score, 1),
            "avg_score_50": round(avg_score, 1),
            "avg_kd_50": round(avg_kd, 2),
            "buffer_size": len(self.states),
            "score_history": self.score_history[-20:],
            "kd_history": [round(k, 2) for k in self.kd_history[-20:]],
        }


def _is_valid_float_list(lst, expected_len):
    """Check list has correct length and no NaN/Inf."""
    if not isinstance(lst, list) or len(lst) != expected_len:
        return False
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in lst)


# ── PPO Trainer ─────────────────────────────────────────────────────────

class PPOTrainer:
    def __init__(self):
        self.agents: dict[str, Agent] = {}
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.total_experiences = 0
        self.total_train_steps = 0
        self.errors = 0
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[Trainer] Device: {DEVICE} (CPU training — stable, no CUDA crashes)")

    def get_or_create_agent(self, agent_id: str) -> Agent:
        with self.lock:
            if agent_id not in self.agents:
                self.agents[agent_id] = Agent(agent_id)
                print(f"[Trainer] Created agent: {agent_id}")
                try:
                    self._export_onnx(agent_id)
                except Exception as e:
                    print(f"[Trainer] ONNX export failed for new {agent_id}: {e}")
            return self.agents[agent_id]

    def process_experience(self, agent_id: str, transitions: list):
        """Process experience batch with full validation and error recovery."""
        agent = self.get_or_create_agent(agent_id)
        accepted = 0

        for t in transitions:
            try:
                state = t.get("state")
                action = t.get("action")

                # Validate data
                if not _is_valid_float_list(state, STATE_DIM):
                    continue
                if not _is_valid_float_list(action, ACTION_DIM):
                    continue

                reward = float(t.get("reward", 0))
                if not math.isfinite(reward):
                    reward = 0.0
                # Clamp extreme rewards
                reward = max(-10.0, min(10.0, reward))

                done = 1.0 if t.get("done") else 0.0

                # Compute value and log_prob on same device as model
                state_t = torch.FloatTensor(state).unsqueeze(0)                action_t = torch.FloatTensor(action).unsqueeze(0)
                with torch.no_grad():
                    action_mean, log_std, value = agent.model(state_t)
                    std = log_std.exp().clamp(min=1e-6)
                    dist = torch.distributions.Normal(action_mean, std)
                    log_prob = dist.log_prob(action_t).sum(-1)

                agent.add_experience(
                    state, action, reward,
                    value.item(), log_prob.item(), done,
                )
                accepted += 1

                if done > 0.5:
                    agent.record_episode(
                        t.get("score", 0),
                        t.get("kills", 0),
                        t.get("deaths", 0),
                    )
            except Exception as e:
                self.errors += 1
                continue

        self.total_experiences += accepted

        # Train if buffer is full
        if agent.has_enough_experience():
            try:
                self._train(agent_id)
            except Exception as e:
                print(f"[Trainer] Training failed for {agent_id}: {e}")
                agent.clear_buffer()

        return accepted

    def _train(self, agent_id: str):
        """PPO training step for a single agent."""
        agent = self.agents[agent_id]
        t0 = time.time()

        states = torch.FloatTensor(np.array(agent.states))        actions = torch.FloatTensor(np.array(agent.actions))        old_log_probs = torch.FloatTensor(agent.log_probs)        rewards = list(agent.rewards)
        values = list(agent.values)
        dones = list(agent.dones)

        # GAE advantages (computed on CPU, then moved to device)
        advantages = [0.0] * len(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = values[t + 1] if t < len(rewards) - 1 else 0.0
            delta = rewards[t] + GAMMA * next_val * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae

        advantages = torch.FloatTensor(advantages)        returns = advantages + torch.FloatTensor(values)        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for epoch in range(PPO_EPOCHS):
            indices = torch.randperm(len(states))
            for start in range(0, len(states), 64):
                idx = indices[start:start + 64]
                b_states = states[idx]
                b_actions = actions[idx]
                b_old_lp = old_log_probs[idx]
                b_adv = advantages[idx]
                b_ret = returns[idx]

                action_mean, log_std, value = agent.model(b_states)
                std = log_std.exp().clamp(min=1e-6)
                dist = torch.distributions.Normal(action_mean, std)
                new_lp = dist.log_prob(b_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = (new_lp - b_old_lp).exp()
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * b_adv
                loss = (
                    -torch.min(surr1, surr2).mean()
                    + VALUE_COEF * (b_ret - value.squeeze(-1)).pow(2).mean()
                    - ENTROPY_COEF * entropy
                )

                agent.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.model.parameters(), 0.5)
                agent.optimizer.step()

        agent.clear_buffer()
        agent.train_steps += 1
        agent.model_version += 1
        agent.last_train_time = time.time() - t0
        self.total_train_steps += 1

        self._export_onnx(agent_id)
        self._save_checkpoint(agent_id)

        stats = agent.get_stats()
        print(
            f"[Train] {agent_id}: v{agent.model_version} "
            f"score={stats['avg_score_50']} kd={stats['avg_kd_50']} "
            f"ep={agent.total_episodes} ({agent.last_train_time:.1f}s)"
        )

        log_entry = {**stats, "timestamp": time.time()}
        with open(LOGS_DIR / f"{agent_id}.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def _export_onnx(self, agent_id: str):
        agent = self.agents[agent_id]
        import copy
        actor = ActorOnly(copy.deepcopy(agent.model))
        actor.eval()
        dummy = torch.randn(1, STATE_DIM)
        onnx_path = MODELS_DIR / f"{agent_id}.onnx"

        torch.onnx.export(
            actor, dummy, str(onnx_path),
            input_names=["state"], output_names=["action"],
            dynamic_axes={"state": {0: "batch"}, "action": {0: "batch"}},
            opset_version=18,
        )

        # Force inline weights
        data_path = MODELS_DIR / f"{agent_id}.onnx.data"
        if data_path.exists():
            import onnx
            model = onnx.load(str(onnx_path))
            onnx.save(model, str(onnx_path))
            data_path.unlink()

    def _save_checkpoint(self, agent_id: str):
        agent = self.agents[agent_id]
        ckpt_dir = MODELS_DIR / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        torch.save({
            "model_state": agent.model.state_dict(),
            "optimizer_state": agent.optimizer.state_dict(),
            "agent_id": agent_id,
            "model_version": agent.model_version,
            "total_episodes": agent.total_episodes,
            "total_steps": agent.total_steps,
            "best_score": agent.best_score,
            "score_history": agent.score_history,
            "kd_history": agent.kd_history,
        }, ckpt_dir / f"{agent_id}.pt")

    def load_checkpoint(self, agent_id: str):
        ckpt_path = MODELS_DIR / "checkpoints" / f"{agent_id}.pt"
        if not ckpt_path.exists():
            return False
        agent = self.get_or_create_agent(agent_id)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        agent.model.load_state_dict(ckpt["model_state"])
        agent.optimizer.load_state_dict(ckpt["optimizer_state"])
        agent.model_version = ckpt.get("model_version", 0)
        agent.total_episodes = ckpt.get("total_episodes", 0)
        agent.total_steps = ckpt.get("total_steps", 0)
        agent.best_score = ckpt.get("best_score", 0)
        agent.score_history = ckpt.get("score_history", [])
        agent.kd_history = ckpt.get("kd_history", [])
        # Skip ONNX export on load — export lazily when model is first requested
        print(f"[Trainer] Loaded {agent_id} (v{agent.model_version}, {agent.total_episodes} ep)")
        return True

    def get_all_stats(self):
        stats = [a.get_stats() for a in self.agents.values()]
        stats.sort(key=lambda s: s["avg_score_50"], reverse=True)
        return stats

    def natural_selection(self, num_cull=5):
        with self.lock:
            if len(self.agents) < num_cull * 2:
                return {"error": f"Not enough agents ({len(self.agents)})"}

            ranked = sorted(
                self.agents.values(),
                key=lambda a: (
                    (np.mean(a.score_history) if a.score_history else 0)
                    + (np.mean(a.kd_history) * 100 if a.kd_history else 0)
                ),
                reverse=True,
            )

            strong = ranked[:num_cull]
            weak = ranked[-num_cull:]

            print(f"\n{'='*60}")
            print(f"[Selection] Culling {num_cull} weakest, cloning strongest")
            results = {"strong": [], "weak": [], "clones": []}

            for a in strong:
                s = np.mean(a.score_history) if a.score_history else 0
                k = np.mean(a.kd_history) if a.kd_history else 0
                results["strong"].append({"id": a.agent_id, "score": round(s), "kd": round(k, 2)})
                print(f"  STRONG {a.agent_id}: score={s:.0f} kd={k:.2f}")

            for i, weak_agent in enumerate(weak):
                parent = strong[i % len(strong)]
                child_id = weak_agent.agent_id

                new_agent = Agent(child_id)
                new_agent.model.load_state_dict(parent.model.state_dict())

                with torch.no_grad():
                    for param in new_agent.model.parameters():
                        param.add_(torch.randn_like(param) * 0.02)

                new_agent.optimizer = optim.Adam(new_agent.model.parameters(), lr=LEARNING_RATE)
                new_agent.model_version = parent.model_version

                self.agents[child_id] = new_agent
                try:
                    self._export_onnx(child_id)
                    self._save_checkpoint(child_id)
                except Exception as e:
                    print(f"  Export failed for {child_id}: {e}")

                results["clones"].append({"child": child_id, "parent": parent.agent_id})
                print(f"  {child_id} <- cloned from {parent.agent_id}")

            print(f"{'='*60}\n")

            with open(LOGS_DIR / "selection.jsonl", "a") as f:
                f.write(json.dumps({"timestamp": time.time(), **results}) + "\n")

            return results

    def reset_agent(self, agent_id: str):
        """Reset an agent to a fresh random network."""
        with self.lock:
            self.agents[agent_id] = Agent(agent_id)
            self._export_onnx(agent_id)
            self._save_checkpoint(agent_id)
            print(f"[Trainer] Reset {agent_id}")

    def clone_agent(self, target_id: str, source_id: str, mutation=0.02):
        """Clone source agent into target with optional mutation."""
        with self.lock:
            source = self.agents.get(source_id)
            if not source:
                return False
            new_agent = Agent(target_id)
            new_agent.model.load_state_dict(source.model.state_dict())
            if mutation > 0:
                with torch.no_grad():
                    for p in new_agent.model.parameters():
                        p.add_(torch.randn_like(p) * mutation)
            new_agent.optimizer = optim.Adam(new_agent.model.parameters(), lr=LEARNING_RATE)
            new_agent.model_version = source.model_version
            self.agents[target_id] = new_agent
            self._export_onnx(target_id)
            self._save_checkpoint(target_id)
            print(f"[Trainer] Cloned {source_id} -> {target_id}")
            return True

    def get_system_info(self):
        uptime = time.time() - self.start_time
        gpu_info = "CPU (20K param models — fast enough)"
        try:
            if torch.cuda.is_available():
                gpu_info = f"CPU training, GPU available: {torch.cuda.get_device_name(0)}"
        except:
            pass
        return {
            "device": str(DEVICE),
            "gpu": gpu_info,
            "uptime_seconds": int(uptime),
            "uptime_human": f"{int(uptime//3600)}h{int((uptime%3600)//60)}m",
            "total_experiences": self.total_experiences,
            "total_train_steps": self.total_train_steps,
            "errors": self.errors,
            "num_agents": len(self.agents),
        }


# ── HTTP API ────────────────────────────────────────────────────────────

def create_app(trainer: PPOTrainer):
    from flask import Flask, request, jsonify, send_file
    from flask import Response

    app = Flask(__name__)

    @app.after_request
    def add_cors(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    @app.route("/experience", methods=["POST"])
    def receive_experience():
        try:
            data = request.get_json()
            agent_id = data["agent_id"]
            transitions = data["transitions"]
            accepted = trainer.process_experience(agent_id, transitions)
            agent = trainer.agents.get(agent_id)
            return jsonify({
                "ok": True,
                "accepted": accepted,
                "model_version": agent.model_version if agent else 0,
            })
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/model/<agent_id>", methods=["GET"])
    def get_model(agent_id):
        model_path = MODELS_DIR / f"{agent_id}.onnx"
        if not model_path.exists():
            agent = trainer.get_or_create_agent(agent_id)
            try:
                trainer._export_onnx(agent_id)
            except Exception as e:
                return jsonify({"error": f"Export failed: {e}"}), 503
        if not model_path.exists():
            return jsonify({"error": "Model not ready"}), 503
        return send_file(str(model_path.resolve()), mimetype="application/octet-stream")

    @app.route("/model/<agent_id>/version", methods=["GET"])
    def get_model_version(agent_id):
        agent = trainer.agents.get(agent_id)
        return jsonify({"version": agent.model_version if agent else 0})

    @app.route("/stats", methods=["GET"])
    def get_stats():
        return jsonify({
            **trainer.get_system_info(),
            "agents": trainer.get_all_stats(),
        })

    @app.route("/episode", methods=["POST"])
    def record_episode():
        data = request.get_json()
        agent = trainer.get_or_create_agent(data["agent_id"])
        agent.record_episode(data.get("score", 0), data.get("kills", 0), data.get("deaths", 0))
        return jsonify({"ok": True})

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
        log_path = LOGS_DIR / f"{agent_id}.jsonl"
        if not log_path.exists():
            return jsonify([])
        entries = []
        for line in log_path.read_text().strip().split("\n"):
            if line:
                try:
                    entries.append(json.loads(line))
                except:
                    pass
        return jsonify(entries[-100:])  # Last 100 entries

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", **trainer.get_system_info()})

    return app


if __name__ == "__main__":
    trainer = PPOTrainer()

    ckpt_dir = MODELS_DIR / "checkpoints"
    if ckpt_dir.exists():
        for f in sorted(ckpt_dir.glob("*.pt")):
            trainer.load_checkpoint(f.stem)

    app = create_app(trainer)
    port = int(os.environ.get("TRAINER_PORT", 5555))
    print(f"[Trainer] Starting on port {port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
