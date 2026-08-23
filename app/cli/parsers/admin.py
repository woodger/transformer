from app.cli.formatting import HelpFormatter
from app.cli.options import add_hidden_help_argument
from app.cli.parsers import SubparserTarget


def add_admin_parsers(subparsers: SubparserTarget) -> None:
    _add_auth_parsers(subparsers)
    _add_database_parsers(subparsers)
    _add_model_parsers(subparsers)


def _add_auth_parsers(subparsers: SubparserTarget) -> None:
    auth = subparsers.add_parser(
        "auth",
        help="OAuth client administration commands.",
        formatter_class=HelpFormatter,
    )
    auth_commands = auth.add_subparsers(
        dest="auth_action",
        required=True,
        title="Commands",
        metavar="COMMAND",
    )
    clients = auth_commands.add_parser(
        "clients",
        help="Transformer-managed Hydra OAuth clients.",
        formatter_class=HelpFormatter,
    )
    client_commands = clients.add_subparsers(
        dest="clients_action",
        required=True,
        title="Commands",
        metavar="COMMAND",
    )
    create = client_commands.add_parser(
        "create",
        add_help=False,
        description="Create Hydra OAuth client credentials for Transformer.",
        formatter_class=HelpFormatter,
    )
    add_hidden_help_argument(create)
    create.add_argument(
        "client_id",
        metavar="CLIENT_ID",
        help="Stable OAuth client ID and Transformer owner identity.",
    )
    create.set_defaults(data=None, metrics_name=None)

    client_list = client_commands.add_parser(
        "list",
        add_help=False,
        description="List Transformer-managed Hydra OAuth clients.",
        formatter_class=HelpFormatter,
    )
    add_hidden_help_argument(client_list)
    client_list.set_defaults(data=None, metrics_name=None)

    delete = client_commands.add_parser(
        "delete",
        add_help=False,
        description=(
            "Delete one Transformer-managed Hydra client and its access tokens."
        ),
        formatter_class=HelpFormatter,
    )
    add_hidden_help_argument(delete)
    delete.add_argument(
        "client_id",
        metavar="CLIENT_ID",
        help="Hydra OAuth client ID returned by auth clients create.",
    )
    delete.set_defaults(data=None, metrics_name=None)


def _add_database_parsers(subparsers: SubparserTarget) -> None:
    database = subparsers.add_parser(
        "db",
        help="Database schema commands.",
        formatter_class=HelpFormatter,
    )
    database_commands = database.add_subparsers(
        dest="db_action",
        required=True,
        title="Commands",
        metavar="COMMAND",
    )
    migrations = database_commands.add_parser(
        "migrations",
        help="Database schema migration commands.",
        formatter_class=HelpFormatter,
    )
    migration_commands = migrations.add_subparsers(
        dest="migrations_action",
        required=True,
        title="Commands",
        metavar="COMMAND",
    )
    for name, description in (
        ("status", "Read the current and expected schema revisions."),
        ("apply", "Apply all pending schema migrations."),
        ("rollback", "Revert the latest applied schema migration."),
    ):
        command = migration_commands.add_parser(
            name,
            add_help=False,
            description=description,
            formatter_class=HelpFormatter,
        )
        add_hidden_help_argument(command)
        command.set_defaults(data=None, metrics_name=None)


def _add_model_parsers(subparsers: SubparserTarget) -> None:
    models = subparsers.add_parser(
        "models",
        help="Published model administration commands.",
        formatter_class=HelpFormatter,
    )
    model_commands = models.add_subparsers(
        dest="models_action",
        required=True,
        title="Commands",
        metavar="COMMAND",
    )
    model_list = model_commands.add_parser(
        "list",
        add_help=False,
        help="List available published model generations.",
        description="List available published model generations.",
        formatter_class=HelpFormatter,
    )
    add_hidden_help_argument(model_list)
    model_list.add_argument(
        "--deleted",
        action="store_true",
        help="List models being deleted and completed deletion records.",
    )
    model_list.set_defaults(data=None, metrics_name=None)

    delete = model_commands.add_parser(
        "delete",
        add_help=False,
        help="Request deletion of one exact model generation.",
        description="Request deletion of one exact published model generation.",
        formatter_class=HelpFormatter,
    )
    add_hidden_help_argument(delete)
    delete.add_argument(
        "model_ref",
        metavar="MODEL_REF",
        help="Exact published model reference; aliases are not accepted.",
    )
    delete.set_defaults(data=None, metrics_name=None, deleted=False)


__all__ = ["add_admin_parsers"]
