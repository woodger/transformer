from app.cli.formatting import (
    COMMAND_HELP,
    FlightServiceHelpFormatter,
    HelpFormatter,
)
from app.cli.options import add_hidden_help_argument, nonnegative_int
from app.cli.parsers import SubparserTarget
from app.service.bootstrap.config import HOST_DEFAULT, PORT_DEFAULT

_FLIGHT_SERVE_DESCRIPTION = (
    "Run the durable Arrow Flight service for fit and predict jobs."
)


def add_service_parsers(subparsers: SubparserTarget) -> None:
    flight = subparsers.add_parser(
        "flight",
        help="Arrow Flight service commands.",
        formatter_class=HelpFormatter,
    )
    flight_commands = flight.add_subparsers(
        dest="flight_action",
        required=True,
        title="Commands",
        metavar="COMMAND",
    )
    service = flight_commands.add_parser(
        "serve",
        add_help=False,
        help=COMMAND_HELP["flight serve"],
        description=_FLIGHT_SERVE_DESCRIPTION,
        formatter_class=FlightServiceHelpFormatter,
        usage="%(prog)s [options]",
    )
    add_hidden_help_argument(service)
    service.add_argument(
        "--host",
        metavar="HOST",
        default=None,
        help=f"Listen host. (default: {HOST_DEFAULT})",
    )
    service.add_argument(
        "--port",
        type=nonnegative_int,
        default=None,
        help=f"Listen port. (default: {PORT_DEFAULT})",
    )
    service.add_argument(
        "--allow-plaintext",
        action="store_true",
        default=None,
        help="Allow serving without TLS.",
    )
    service.add_argument(
        "--tls-cert-file",
        default=None,
        metavar="FILE",
        help="Server certificate PEM.\nRequires --tls-key-file.",
    )
    service.add_argument(
        "--tls-key-file",
        default=None,
        metavar="FILE",
        help="Server private key PEM.\nRequires --tls-cert-file.",
    )
    service.add_argument(
        "--tls-ca-file",
        default=None,
        metavar="FILE",
        help="Client CA PEM.\nRequires server TLS.",
    )
    service.add_argument(
        "--tls-require-client-cert",
        action="store_true",
        default=None,
        help=(
            "Require client certificates.\n"
            "Requires --tls-ca-file and server TLS."
        ),
    )
    service.set_defaults(data=None, metrics_name=None)


__all__ = ["add_service_parsers"]
