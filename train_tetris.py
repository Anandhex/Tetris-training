#!/usr/bin/env python3
import argparse
import sys
import json

from tetris_client import UnityTetrisClient
from tetris_trainer import TetrisTrainer  # make sure this points to your updated trainer

def main(args):
    # 1) Load JSON config
    try:
        with open(args.config, 'r') as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load config '{args.config}': {e}", file=sys.stderr)
        sys.exit(1)

    reward_config = cfg.get('reward_config', {})
     # 2) Extract agent-specific parameters
    agent_params = {}
    
    # Get general parameters that apply to all agents
    general_params = cfg.get('agent_params', {})
    agent_params.update(general_params)
    
    # Get agent-specific parameters based on selected agent
    if args.agent.startswith('sb3_'):
        algorithm = args.agent.split('_')[1].lower()  # ppo, dqn, a2c
        sb3_params = cfg.get('sb3_params', {})
        
        # Get algorithm-specific parameters
        algo_params = sb3_params.get(algorithm, {})
        agent_params.update(algo_params)
        
        # Get common SB3 parameters
        common_sb3_params = sb3_params.get('common', {})
        agent_params.update(common_sb3_params)

    # 2) Configuration summary
    use_curriculum = args.curriculum and not args.no_curriculum
    print("[CONFIG]")
    print(f"  Host           = {args.host}")
    print(f"  Port           = {args.port}")
    print(f"  Agent          = {args.agent}")
    print(f"  Curriculum     = {use_curriculum}")
    print(f"  Episodes       = {args.episodes}")
    print("  Reward config  =")
    for k, v in reward_config.items():
        print(f"    {k}: {v}")
    print(" Agent parameters =")
    for k, v in agent_params.items():
        print(f"   {k}: {v}")    
    print("--------------------------------------------------")

    # 3) Initialize Unity client

    print("[INFO] Creating trainer...")
    trainer = TetrisTrainer(
        agent_type=args.agent,
        reward_config=reward_config,
        curriculum=use_curriculum,
        host=args.host,
        port=args.port,
        **agent_params
    )

    # 5) Run training
    print(f"[INFO] Starting training for {args.episodes} episodes...")
    trainer.train(args.episodes)
    print("[INFO] Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Tetris with JSON-driven rewards, agents & curriculum"
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to JSON config file"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Unity host address"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=12345,
        help="Unity socket port"
    )
    parser.add_argument(
        "--agent",
        choices=["dqn", "dqn_noise", "greedy",'sb3_ppo', 'sb3_dqn', 'sb3_a2c',"nuna"],
        default="dqn",
        help="Which agent to use"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable curriculum learning"
    )
    group.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Disable curriculum learning (start at final stage)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes"
    )

    args = parser.parse_args()
    print("[START] train_tetris.py invoked with arguments:", args)
    print("==================================================")
    main(args)
