import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description='Five alternating Thin/native rounds for three YOLO families')
    for family in ('v5m', 'v8s', 'v11s'):
        parser.add_argument(f'--{family}-model', type=Path, required=True)
        parser.add_argument(f'--{family}-fixture', type=Path, required=True)
        parser.add_argument(f'--{family}-model-format', choices=('fused-module', 'ultralytics'),
                            default='ultralytics' if family == 'v8s' else 'fused-module')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--samples', type=int, default=101)
    args = parser.parse_args()
    if args.rounds < 1 or args.warmup < 0 or args.samples < 1:
        parser.error('rounds/samples must be positive; warmup must be nonnegative')
    families = ('v5m', 'v8s', 'v11s')
    for family in families:
        for kind in ('model', 'fixture'):
            path = getattr(args, f'{family}_{kind}')
            if not path.is_file():
                raise FileNotFoundError(path)
    if args.v5m_model_format != 'fused-module':
        parser.error('Classic v5 requires an upstream-constructed fused-module artifact')
    args.output.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name('benchmark_family_capture.py')
    rounds = {family: [] for family in families}
    for number in range(1, args.rounds + 1):
        order = ('thin', 'native') if number % 2 else ('native', 'thin')
        for family in families:
            row = dict(round=number, order=list(order))
            for backend in order:
                directory = args.output / family / f'{number}-{backend}'
                directory.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, str(worker), '--family', family, '--backend', backend,
                           '--model', str(getattr(args, f'{family}_model').resolve()),
                           '--model-format', getattr(args, f'{family}_model_format'),
                           '--fixture', str(getattr(args, f'{family}_fixture').resolve()),
                           '--output', str(directory.resolve()), '--warmup', str(args.warmup),
                           '--samples', str(args.samples)]
                print(f'Round {number}: {family} {backend}', flush=True)
                with (directory / 'run.log').open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                row[backend] = json.loads((directory / 'result.json').read_text())
            row['thin_native_latency_ratio'] = row['thin']['median_ms'] / row['native']['median_ms']
            rounds[family].append(row)
    summary = {}
    for family, rows in rounds.items():
        thin = statistics.median(row['thin']['median_ms'] for row in rows)
        native = statistics.median(row['native']['median_ms'] for row in rows)
        summary[family] = dict(thin_median_of_round_medians_ms=thin,
                               native_median_of_round_medians_ms=native,
                               thin_native_latency_ratio=thin / native)
    result = dict(scope='synchronized whole-model graph replay at 640x640; no input copy or NMS',
                  fresh_processes=args.rounds * len(families) * 2,
                  rounds_per_family=args.rounds, warmup=args.warmup, samples_per_process=args.samples,
                  aggregation='median of per-round medians; no outlier removal',
                  summary=summary, rounds=rounds)
    (args.output / 'family_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
