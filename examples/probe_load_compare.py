import argparse

from examples.probe_store import _generate_probe_training_summary
from mmlib.probe import ProbeSummary, ProbeInfo


def main(args):
    # we use the functionality from the probe_store script to generate a summary for the GoogLeNet
    summary = _generate_probe_training_summary()
    # We load the summary from the path given in the args
    loaded_summary = ProbeSummary(summary_path=args.path)
    # we specify the fields both summaries have in common (they are excluded form comparing)
    common = [ProbeInfo.LAYER_NAME, ProbeInfo.FORWARD_INDEX]
    # we define the fields we want to compare; in this case different kind of tensors for teh forward an backward pass
    compare = [ProbeInfo.INPUT_TENSOR, ProbeInfo.OUTPUT_TENSOR, ProbeInfo.GRAD_INPUT_TENSOR,
               ProbeInfo.GRAD_OUTPUT_TENSOR]
    # haven created one summary and loaded one we compare them and print the comparison to the console
    summary.compare_to(loaded_summary, common, compare)


def parse_args():
    parser = argparse.ArgumentParser(description='Create a summary, load another one and compare.')
    parser.add_argument('--path', help='path to load the summary data from', default='./summary')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
