"""Command-line entry point; run `python main.py examples/tension.json`."""

import argparse
from pathlib import Path
import sys

from laminate import Laminate, analyze
from reporting import write_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="层合板比例加载渐进失效分析（MPa、mm）")
    parser.add_argument("input", type=Path, help="JSON 输入文件")
    parser.add_argument("--output", type=Path, default=Path("results"), help="输出目录，默认 results")
    args = parser.parse_args(argv)
    try:
        model = Laminate.from_json(args.input)
        result = analyze(model)
        write_outputs(result, args.output)
    except (OSError, ValueError, ArithmeticError) as exc:
        print(f"计算失败：{exc}", file=sys.stderr)
        return 1
    print(f"计算完成：{model.title}")
    if result.first_failure_load_factor is not None:
        print(f"首层失效加载倍数：{result.first_failure_load_factor:.9g}")
    if result.ultimate_load_factor is not None:
        print(f"整体失效加载倍数：{result.ultimate_load_factor:.9g}")
    else:
        print(result.reason)
        if result.terminal_load_factor is not None:
            print(f"计算中止加载倍数：{result.terminal_load_factor:.9g}（未达到所有层完全失效）")
    print(f"共 {len(result.events)} 次层损伤，结果已保存至：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
