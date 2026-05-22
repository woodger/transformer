from app.cli.help import build_parser


def parse_args():
    return build_parser().parse_args()
