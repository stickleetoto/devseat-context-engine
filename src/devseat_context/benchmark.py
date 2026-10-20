"""Run: python -m devseat_context.benchmark benchmarks/fixtures.json"""
import argparse
from pathlib import Path

from .evaluation import evaluate_cases, load_cases


def main() -> None:
    parser = argparse.ArgumentParser(description='Small exact-evidence retrieval benchmark')
    parser.add_argument('fixtures', nargs='?', default=str(Path(__file__).resolve().parents[2] / 'benchmarks' / 'fixtures.json'))
    args = parser.parse_args()
    scores = evaluate_cases(load_cases(args.fixtures))
    for score in scores:
        print(f'{score.case_id:>15} {score.mode:<25} '
              f'literal_recall={score.literal_recall:.2f} '
              f'char_reduction={score.character_reduction:.2%} '
              f'ms={score.elapsed_ms:.2f}')
    for mode in sorted(set(s.mode for s in scores)):
        group = [s for s in scores if s.mode == mode]
        hits = sum(s.hit_count for s in group)
        total = sum(s.expected_count for s in group)
        print(f'{mode}: micro_literal_recall={hits / total:.3f} ({hits}/{total})')


if __name__ == '__main__':
    main()
