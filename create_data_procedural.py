import argparse
import json
import random
from pathlib import Path


INSTRUCTION = (
    "你是一个具身智能无人机集群的战术指挥中枢，请将用户的自然语言战术意图"
    "转换为结构化 JSON 调度指令。只输出 JSON，不要额外解释。"
)


def _default_output_path() -> Path:
    llm_dir = Path(__file__).resolve().parent
    return llm_dir / "data" / "marl_llm_dataset_procedural.json"


def _one_item(rng: random.Random) -> dict:
    task_type = rng.choice(
        [
            "simple_tag",
            "regional_capture",
            "escort_and_navigate",
            "urban_search",
            "survival_evasion",
            "tactical_breach",
        ]
    )

    if task_type == "simple_tag":
        predators = rng.randint(2, 6)
        prey = rng.randint(1, 3)
        capture_reward = rng.choice([40.0, 60.0, 80.0, 100.0])
        collision_penalty = rng.choice([-10.0, -20.0, -50.0, -100.0])
        input_text = (
            f"演练开始，敌方释放{prey}架诱饵机，我方出动{predators}架拦截机进行追击。"
            "要求左右包抄，先压缩逃逸空间再捕获；尽量避免友机碰撞。"
        )
        output_obj = {
            "task_type": "simple_tag",
            "agents_config": [
                {"role": "predator", "count": predators, "strategy": "pincer_and_compress"},
                {"role": "prey", "count": prey, "strategy": "evasion"},
            ],
            "rl_parameters": {
                "capture_reward": capture_reward,
                "collision_penalty": collision_penalty,
            },
        }
    elif task_type == "regional_capture":
        n = rng.randint(3, 8)
        input_text = (
            "启动区域控制任务，派遣无人机在 Alpha 区进行网格化巡逻搜索；"
            "任意一架发现目标后其余立刻收缩合围。"
        )
        output_obj = {
            "task_type": "regional_capture",
            "agents_config": [{"role": "patrol", "count": n, "strategy": "grid_search_and_surround"}],
            "rl_parameters": {
                "coverage_reward": 20.0,
                "cooperative_capture_reward": 80.0,
                "collision_penalty": -10.0,
            },
        }
    elif task_type == "escort_and_navigate":
        escorts = rng.randint(2, 5)
        input_text = (
            "我方运输机穿越峡谷，派护航无人机组成三角队形伴飞。"
            "途中允许开启强化避障算法，优先保证安全，其次保持队形。"
        )
        output_obj = {
            "task_type": "escort_and_navigate",
            "agents_config": [
                {"role": "vip", "count": 1, "strategy": "follow_waypoints"},
                {"role": "escort", "count": escorts, "strategy": "triangle_formation"},
            ],
            "rl_parameters": {
                "formation_keeping_reward": 15.0,
                "obstacle_avoidance_penalty": -30.0,
            },
        }
    elif task_type == "urban_search":
        n = rng.randint(3, 10)
        input_text = (
            "目标进入城市密集建筑区，所有无人机切换低空静默模式，单机独立搜索。"
            "发现目标后上报并保持跟踪，不要依赖队友通信。"
        )
        output_obj = {
            "task_type": "urban_search",
            "agents_config": [{"role": "searcher", "count": n, "strategy": "independent_stealth"}],
            "rl_parameters": {
                "discovery_reward": 50.0,
                "comm_dependency_penalty": -5.0,
                "altitude_limit": "low",
            },
        }
    elif task_type == "survival_evasion":
        input_text = (
            "集群遭遇强电磁干扰，立刻解散当前队形，所有无人机执行随机规避动作。"
            "优先保证生存，碰撞惩罚设为最高。"
        )
        output_obj = {
            "task_type": "survival_evasion",
            "agents_config": [{"role": "all", "count": "all", "strategy": "random_evasion_no_comm"}],
            "rl_parameters": {
                "survival_reward_per_step": 1.0,
                "collision_penalty": -100.0,
            },
        }
    else:
        decoys = rng.randint(2, 4)
        input_text = (
            "对敌方基地进行模拟突防：多架无人机作为诱饵吸引火力，"
            "主攻机趁机隐蔽渗透突防。成功突防奖励最高，诱饵损失惩罚较低。"
        )
        output_obj = {
            "task_type": "tactical_breach",
            "agents_config": [
                {"role": "decoy", "count": decoys, "strategy": "draw_fire"},
                {"role": "attacker", "count": 1, "strategy": "stealth_penetration"},
            ],
            "rl_parameters": {
                "breach_success_reward": 200.0,
                "decoy_sacrifice_penalty": -10.0,
            },
        }

    return {
        "instruction": INSTRUCTION,
        "input": input_text,
        "output": json.dumps(output_obj, ensure_ascii=False),
    }


def generate_dataset(count: int, *, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [_one_item(rng) for _ in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a procedural SFT dataset (no API).")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(_default_output_path()))
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = generate_dataset(args.count, seed=args.seed)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(dataset)} items to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

