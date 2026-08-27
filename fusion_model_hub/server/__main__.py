import argparse
import asyncio
import json
import logging
import sys

import uvicorn

logger = logging.getLogger(__name__)


def _run_export(args):
    from ..db.database import get_engine, init_db
    from .config import Settings
    from .deps import init_deps

    settings = Settings(data_dir=args.data_dir, db_url=args.db_url or "")
    engine = get_engine(settings.db_url)

    async def _do_export():
        await init_db(engine)
        init_deps(settings, engine)
        from .deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            from ..db import crud

            model_ids = [x.strip() for x in args.models.split(",") if x.strip()] if args.models else []
            models, _ = await crud.list_models(session, page=1, page_size=10000)
            if model_ids:
                models = [m for m in models if m.id in model_ids]
            tenants = await crud.list_tenants(session)
            webhooks = await crud.list_webhooks(session)
            data = {
                "version": "1.0",
                "models": [
                    {
                        "id": m.id,
                        "name": m.name,
                        "tenant_id": m.tenant_id,
                        "description": m.description,
                        "model_type": m.model_type.value,
                        "architecture": m.architecture,
                        "params_size": m.params_size,
                        "license": m.license,
                        "author": m.author,
                        "language": m.language,
                        "task_types": m.task_types,
                        "owner": m.owner,
                        "hf_repo": m.hf_repo,
                        "tags": [{"key": t.key, "value": t.value} for t in m.tags],
                    }
                    for m in models
                ],
                "tenants": [{"id": t.id, "name": t.name, "display_name": t.display_name} for t in tenants],
                "webhooks": [
                    {"id": w.id, "name": w.name, "url": w.url, "events": w.events, "tenant_id": w.tenant_id}
                    for w in webhooks
                ],
            }
        output = args.output or "-"
        content = json.dumps(data, indent=2, ensure_ascii=False)
        if output == "-":
            print(content)
        else:
            with open(output, "w") as f:
                f.write(content)
            logger.info(
                "Exported %d models, %d tenants, %d webhooks to %s",
                len(models),
                len(tenants),
                len(webhooks),
                output,
            )

    asyncio.run(_do_export())


def _run_import(args):
    from ..db import crud
    from ..db.database import get_engine, init_db
    from ..db.models import ModelType
    from .config import Settings
    from .deps import init_deps

    settings = Settings(data_dir=args.data_dir, db_url=args.db_url or "")
    engine = get_engine(settings.db_url)

    async def _do_import():
        await init_db(engine)
        init_deps(settings, engine)
        from .deps import get_session_factory

        sf = get_session_factory()
        input_file = args.input or "-"
        if input_file == "-":
            content = sys.stdin.read()
        else:
            with open(input_file) as f:
                content = f.read()
        data = json.loads(content)
        count = 0
        async with sf() as session:
            for t in data.get("tenants", []):
                existing = await crud.get_tenant_by_name(session, t.get("name", ""))
                if not existing:
                    await crud.create_tenant(session, name=t["name"], display_name=t.get("display_name", ""))
                    count += 1
            for m in data.get("models", []):
                existing = await crud.get_model_by_name(session, m.get("name", ""))
                if not existing:
                    try:
                        mt = ModelType(m.get("model_type", "llm"))
                    except ValueError:
                        mt = ModelType.LLM
                    new_m = await crud.create_model(
                        session,
                        name=m["name"],
                        tenant_id=m.get("tenant_id", ""),
                        description=m.get("description", ""),
                        model_type=mt,
                        architecture=m.get("architecture", ""),
                        params_size=m.get("params_size", ""),
                        license=m.get("license", ""),
                        author=m.get("author", ""),
                        language=m.get("language", ""),
                        task_types=m.get("task_types", ""),
                        owner=m.get("owner", ""),
                        hf_repo=m.get("hf_repo", ""),
                    )
                    tags = m.get("tags", [])
                    if tags:
                        await crud.set_tags(session, new_m.id, tags)
                    count += 1
            for w in data.get("webhooks", []):
                await crud.create_webhook(
                    session,
                    name=w["name"],
                    url=w["url"],
                    events=w.get("events", ""),
                    tenant_id=w.get("tenant_id", ""),
                )
                count += 1
        logger.info("Imported %d items", count)

    asyncio.run(_do_import())


def _run_migrate(args):
    try:
        from alembic.config import Config

        from alembic import command
    except ImportError:
        print("alembic not installed. Run: pip install alembic")
        sys.exit(1)
    alembic_cfg = Config()
    alembic_cfg.set_main_option("script_location", "alembic")
    alembic_cfg.set_main_option("sqlalchemy.url", args.db_url or "sqlite:///data/fmh.db")
    if args.revision:
        command.upgrade(alembic_cfg, args.revision)
        logger.info("Migrated to revision: %s", args.revision)
    else:
        command.upgrade(alembic_cfg, "head")
        logger.info("Migrated to head")


def _run_restore(args):
    from ..db.database import get_engine, init_db
    from .backup import restore_from_backup
    from .config import Settings
    from .deps import init_deps

    settings = Settings(data_dir=args.data_dir, db_url=args.db_url or "")
    engine = get_engine(settings.db_url)

    async def _do_restore():
        await init_db(engine)
        init_deps(settings, engine)
        result = await restore_from_backup(args.input)
        logger.info(
            "Restore result: models=%d versions=%d skipped=%d",
            result["models_restored"],
            result["versions_restored"],
            result["skipped"],
        )
        print(json.dumps(result, indent=2))

    asyncio.run(_do_restore())


def main():
    parser = argparse.ArgumentParser(description="Fusion Model Hub — Model repository & management server")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=11444, help="Bind port (default: 11444)")
    serve_parser.add_argument("--data-dir", default="", help="Data directory (default: ./data)")
    serve_parser.add_argument("--db-url", default="", help="Database URL")
    serve_parser.add_argument("--mlx-url", default="http://localhost:11434", help="Fusion-MLX API URL")
    serve_parser.add_argument("--log-level", default="INFO", help="Log level")
    serve_parser.add_argument("--tls-certfile", default="", help="TLS certificate file path")
    serve_parser.add_argument("--tls-keyfile", default="", help="TLS private key file path")
    # E-E10: `serve` runs uvicorn single-worker by default. The adapt/recommend/
    # hardware engine singletons and the in-process _running_executions /
    # _running_tasks dicts are per-process — running uvicorn with --workers>1
    # (e.g. behind gunicorn) gives each worker its own copy, so GET
    # /adapt/execute/{id} and GET /quantize/running return inconsistent results
    # depending on which worker handles the request. For correctness of those
    # stateful endpoints, run a single worker and scale horizontally with more
    # processes fronted by a state-aware gateway instead. Migrating the task
    # registry to an external coordinator (Redis/DB) is the durable fix and is
    # tracked separately; do not silently enable multi-worker here.

    export_parser = subparsers.add_parser("export", help="Export data to JSON")
    export_parser.add_argument("--data-dir", default="", help="Data directory")
    export_parser.add_argument("--db-url", default="", help="Database URL")
    export_parser.add_argument("--output", "-o", default="-", help="Output file (default: stdout)")
    export_parser.add_argument("--models", default="", help="Comma-separated model IDs to export (default: all)")

    import_parser = subparsers.add_parser("import", help="Import data from JSON")
    import_parser.add_argument("--data-dir", default="", help="Data directory")
    import_parser.add_argument("--db-url", default="", help="Database URL")
    import_parser.add_argument("--input", "-i", default="-", help="Input file (default: stdin)")

    migrate_parser = subparsers.add_parser("migrate", help="Run database migrations")
    migrate_parser.add_argument("--db-url", default="", help="Database URL")
    migrate_parser.add_argument("--revision", default="", help="Target revision (default: head)")

    # P1-22: restore path for the auto-backup schema (models + versions). The
    # `import` subcommand consumes a different schema (tenants/webhooks), so it
    # cannot load a backup file — this is the matching reader.
    restore_parser = subparsers.add_parser("restore", help="Restore models+versions from a backup JSON")
    restore_parser.add_argument("--data-dir", default="", help="Data directory")
    restore_parser.add_argument("--db-url", default="", help="Database URL")
    restore_parser.add_argument("--input", "-i", required=True, help="Backup JSON file to restore from")

    args = parser.parse_args()
    command = args.command or "serve"

    if command == "serve":
        from .config import Settings

        settings = Settings(
            host=args.host,
            port=args.port,
            data_dir=args.data_dir,
            db_url=args.db_url,
            mlx_url=args.mlx_url,
            log_level=args.log_level,
            tls_certfile=args.tls_certfile,
            tls_keyfile=args.tls_keyfile,
        )
        ssl_kwargs = {}
        if settings.tls_certfile and settings.tls_keyfile:
            ssl_kwargs["ssl_certfile"] = settings.tls_certfile
            ssl_kwargs["ssl_keyfile"] = settings.tls_keyfile
            logger.info("TLS enabled: cert=%s", settings.tls_certfile)
        uvicorn.run(
            "fusion_model_hub.server.app:create_app",
            factory=True,
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level.lower(),
            **ssl_kwargs,
        )
    elif command == "export":
        _run_export(args)
    elif command == "import":
        _run_import(args)
    elif command == "migrate":
        _run_migrate(args)
    elif command == "restore":
        _run_restore(args)


if __name__ == "__main__":
    main()
