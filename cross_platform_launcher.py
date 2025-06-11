#!/usr/bin/env python3
"""
cross_platform_launcher.py

Launch multiple Unity + Python training instances on Windows, macOS, or Linux
based on a single config.json that specifies per-OS Unity paths and global reward_config.
For each instance, merges global + instance-specific reward_config,
writes a temp config, and passes it (so train_tetris picks up the right rewards).
"""

import json
import subprocess
import os
import sys
import platform
import tempfile

def main(config_path='config.json'):
    # 1) Load master config
    try:
        with open(config_path, 'r') as f:
            master = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Config '{config_path}' not found.", file=sys.stderr)
        sys.exit(1)

    # 2) Get global settings
    system    = platform.system()       # 'Windows', 'Linux', 'Darwin'
    unity     = master['unity_paths'].get(system)
    host      = master.get('host', '127.0.0.1')
    log_dir   = master.get('log_dir', 'logs')
    global_rc = master.get('reward_config', {})

    if not unity:
        print(f"[ERROR] No Unity path for OS '{system}' in config.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(log_dir, exist_ok=True)
    processes = []

    # 3) Loop over instances
    for inst in master.get('instances', []):
        name       = inst.get('name')
        port       = str(inst.get('port'))
        agent      = inst.get('agent')
        curriculum = inst.get('curriculum', False)
        episodes   = str(inst.get('episodes'))

        if not (name and port and agent):
            print(f"[WARNING] Skipping invalid instance: {inst}", file=sys.stderr)
            continue

        print(f"[{system}] Launching '{name}' on port {port} with agent={agent}, curriculum={curriculum}")

        # 4) Prepare per-instance reward_config
        inst_rc = inst.get('reward_config', {})
        merged_rc = {**global_rc, **inst_rc}
        # write it to a temp file (in log_dir so you can inspect it)
        cfg_run = {'reward_config': merged_rc}
        cfg_path_run = os.path.join(log_dir, f"{name}_config.json")
        with open(cfg_path_run, 'w') as cf:
            json.dump(cfg_run, cf, indent=2)

        # 5) Start Unity player
        unity_cmd = [unity, '-port', port, '-host', host]
        uni_log = open(os.path.join(log_dir, f"{name}_unity.log"), 'w')
        p1 = subprocess.Popen(unity_cmd, stdout=uni_log, stderr=uni_log)
        processes.append((f"{name}_unity", p1, uni_log))

        # 6) Start Python trainer, passing --config cfg_path_run
        py_cmd = [
            sys.executable, 'train_tetris.py',
            '--config', cfg_path_run,
            '--host',   host,
            '--port',   port,
            '--agent',  agent,
            '--episodes', episodes
        ]
        if curriculum:
            py_cmd.append('--curriculum')
        else:
            py_cmd.append('--no-curriculum')

        py_log = open(os.path.join(log_dir, f"{name}_train.log"), 'w')
        p2 = subprocess.Popen(py_cmd, stdout=py_log, stderr=py_log)
        processes.append((f"{name}_train", p2, py_log))

    # 7) Wait for all processes
    for label, proc, logf in processes:
        proc.wait()
        logf.close()
        print(f"[{system}] Process '{label}' exited with code {proc.returncode}")

if __name__ == '__main__':
    cfg = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    main(cfg)
