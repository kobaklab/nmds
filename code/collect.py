"""Stage 3: join runtimes and metrics of all runs into results/all_runs.csv
(one row per dataset x method x seed), and report what is still missing.

  uv run python code/collect.py            # write the CSV and print the status
  uv run python code/collect.py --status   # only print the status
"""
import argparse

import pipeline
import plan
from paths import RESULTS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", action="store_true", help="only print what is missing")
    args = parser.parse_args()

    jobs = plan.main_jobs()  # always everything, so all_runs.csv is never a partial table
    df, report = pipeline.collect(jobs)
    if not args.status:
        RESULTS.mkdir(parents=True, exist_ok=True)
        df.to_csv(RESULTS / "all_runs.csv", index=False)
        print(f"Wrote {len(df)} rows to {RESULTS / 'all_runs.csv'}")
    pipeline.print_report(report, len(jobs))


if __name__ == "__main__":
    main()
