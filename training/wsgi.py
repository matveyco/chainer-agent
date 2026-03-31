import os

from training.trainer import PPOTrainer, create_app


trainer = PPOTrainer()
trainer.load_existing_agents()
app = create_app(trainer)


if __name__ == "__main__":
    port = int(os.environ.get("TRAINER_PORT", "5555"))
    app.run(host="0.0.0.0", port=port, threaded=True)
