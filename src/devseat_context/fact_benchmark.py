"""Compare v0.3 and v0.4 on tiny public synthetic fact-preservation cases.

Gold strings are read AFTER running both compression methods; they are not
provided to either algorithm. All budgets here count characters, not tokens.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

from .document import compress_document
from .engine import BudgetExceeded
from .facts import compress_fact_aware


def evaluate_fact_cases(cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        source = case['text']
        query = case['query']
        budget = case['budget_characters']
        gold = case['expected_literals']
        if any(not isinstance(x, str) or not x or x not in source for x in gold):
            raise ValueError('invalid gold literal in ' + case['id'])
        if not isinstance(budget, int) or budget < 0:
            raise ValueError('invalid budget in ' + case['id'])
        for mode in ('v0.3_span', 'v0.4_fact_strict'):
            start = perf_counter()
            try:
                if mode == 'v0.3_span':
                    output = compress_document(source, budget, query=query, token_counter=len).text
                    conflicts = 0
                else:
                    result = compress_fact_aware(source, budget, query=query, token_counter=len)
                    output = result.text
                    conflicts = len(result.audit.conflicts)
                state = 'ok'
            except BudgetExceeded:
                output = ''
                conflicts = 0
                state = 'fail_closed'
            elapsed_ms = (perf_counter() - start) * 1000
            # Answers are never generated: only literal hit counts are measured.
            rows.append({
                'id': case['id'], 'mode': mode, 'status': state,
                'literal_hits': (sum(x in output for x in gold) if state == 'ok' else 0),
                'literal_total': len(gold),
                'output_chars': len(output), 'source_chars': len(source),
                'budget_characters': budget,
                'conflict_candidates': conflicts, 'elapsed_ms': round(elapsed_ms, 3),
            })
            if state == 'ok' and len(output) > budget:
                raise AssertionError('budget exceeded')
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('fixtures', nargs='?', default=str(
        Path(__file__).resolve().parents[2] / 'benchmarks' / 'fact_fixtures.json'
    ))
    args = parser.parse_args()
    cases = json.loads(Path(args.fixtures).read_text(encoding='utf-8'))
    rows = evaluate_fact_cases(cases)
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    for mode in ('v0.3_span', 'v0.4_fact_strict'):
        group = [x for x in rows if x['mode'] == mode]
        valid = [x for x in group if x['status'] == 'ok']
        hits = sum(x['literal_hits'] for x in valid)
        total = sum(x['literal_total'] for x in valid)
        failures = len(group) - len(valid)
        print(f'{mode}: {hits}/{total} visible literal hits on successful cases; '
              f'fail_closed={failures}/{len(group)}')
    print('NOTE: Tiny hand-made synthetic cases; NOT model accuracy or token savings.')


if __name__ == '__main__':
    main()
