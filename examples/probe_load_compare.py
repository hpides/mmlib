import argparse

from examples.probe_store import _generate_probe_training_summary
from mmlib.probe import ProbeSummary, ProbeInfo


def main(args):
    summary = _generate_probe_training_summary()

    loaded_summary = ProbeSummary(summary_path=args.path)

    common = [ProbeInfo.LAYER_NAME]
    compare = [ProbeInfo.INPUT_TENSOR, ProbeInfo.OUTPUT_TENSOR, ProbeInfo.GRAD_INPUT_TENSOR,
               ProbeInfo.GRAD_OUTPUT_TENSOR]

    summary.compare_to(loaded_summary, common, compare)


def parse_args():
    parser = argparse.ArgumentParser(description='Create a summary, load another one and compare.')
    parser.add_argument('--path', help='path to load the summary data from', default='./summary')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
