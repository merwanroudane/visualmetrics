"""Command-line interface.

Built on argparse so the base install stays dependency-light:

    visualmetrics                 # launch the GUI
    visualmetrics list --domain econometrics
    visualmetrics search "power"
    visualmetrics show inference.power
    visualmetrics run inference.power --set alpha=0.01 --export out.html
    visualmetrics proofs
    visualmetrics proof regression.fwl.theorem --checks
    visualmetrics doctor
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .version import __version__

__all__ = ["main", "build_parser", "build_doctor_report", "format_doctor_report"]


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------
def build_doctor_report() -> dict[str, Any]:
    """Collect everything ``visualmetrics doctor`` needs to know."""
    import platform

    from .backends.capabilities import capability_report
    from .config import cache_path, config_path
    from .core.registry import registry
    from .i18n.translator import available_languages

    def writable(path) -> bool:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".vm_write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return True
        except OSError:
            return False

    caps = capability_report()
    core: dict[str, Any] = {}
    for name in ("numpy", "scipy", "pandas", "platformdirs"):
        try:
            module = __import__(name)
            core[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            core[name] = f"MISSING ({type(exc).__name__})"

    return {
        "visualmetrics": __version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "core": core,
        "capabilities": caps,
        "languages": available_languages(),
        "catalog": registry.stats(),
        "config_path": str(config_path()),
        "cache_path": str(cache_path()),
        "config_writable": writable(config_path().parent),
        "cache_writable": writable(cache_path()),
    }


def format_doctor_report(report: dict[str, Any]) -> str:
    lines = [
        f"VisualMetrics {report['visualmetrics']} environment check",
        "=" * 52,
        f"Python            {report['python']} on {report['platform']}",
        "",
        "Core dependencies",
    ]
    for name, version in report["core"].items():
        mark = "MISSING" if str(version).startswith("MISSING") else "OK"
        lines.append(f"  {name:<16} {mark:<8} {version}")

    lines.append("")
    lines.append("Optional capabilities")
    by_extra: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for name, info in report["capabilities"].items():
        by_extra.setdefault(str(info["extra"]), []).append((name, info))
    for extra in sorted(by_extra):
        entries = by_extra[extra]
        ok = sum(1 for _, i in entries if i["available"])
        lines.append(f"  [{extra}]  {ok}/{len(entries)} available")
        for name, info in sorted(entries):
            if info["available"]:
                lines.append(f"    {name:<14} OK       {info['version']}")
            else:
                lines.append(f"    {name:<14} MISSING  {info['reason']} - {info['purpose']}")
        if ok < len(entries):
            lines.append(f'    install:  pip install "visualmetrics[{extra}]"')

    catalog = report["catalog"]
    lines += [
        "",
        "Content",
        f"  languages       {', '.join(report['languages'])}",
        f"  concepts        {catalog['implemented']} implemented / {catalog['total']} catalogued",
        "",
        "Paths",
        f"  config          {report['config_path']}"
        f"  {'(writable)' if report['config_writable'] else '(NOT WRITABLE)'}",
        f"  cache           {report['cache_path']}"
        f"  {'(writable)' if report['cache_writable'] else '(NOT WRITABLE)'}",
        "",
        "Missing optional extras only disable the features that need them; the core works.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def _cmd_gui(args: argparse.Namespace) -> int:
    from .gui.app import launch_gui

    launch_gui(
        host=args.host,
        port=args.port,
        native=args.native,
        open_browser=not args.no_browser,
        reload=False,
    )
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    from .core.registry import registry
    from .i18n.translator import get_translator

    tr = get_translator(args.language)
    specs = registry.specs(
        domain=args.domain,
        level=args.level,
        implemented_only=not args.all,
    )
    if not specs:
        print("No concept matched those filters.")
        return 1
    if args.json:
        print(json.dumps([{"id": s.id, "domain": s.domain.value, "status": s.status.value,
                           "title": tr.t(s.title_key)} for s in specs],
                         indent=2, ensure_ascii=False))
        return 0
    current = None
    for spec in specs:
        if spec.domain is not current:
            current = spec.domain
            print()
            print(tr.t(current.label_key, current.value.replace("_", " ").title()))
            print("-" * 60)
        flag = "" if spec.status.is_implemented else "  [planned]"
        print(f"  {spec.id:<38} {tr.t(spec.title_key)}{flag}")
    stats = registry.stats()
    print()
    print(f"{stats['implemented']} implemented labs out of {stats['total']} catalogued concepts.")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    from .catalog.search import search_concepts
    from .i18n.translator import set_language

    if args.language:
        set_language(args.language)
    hits = search_concepts(args.query, limit=args.limit)
    if not hits:
        print(f"No concept matched {args.query!r}.")
        return 1
    for hit in hits:
        flag = "" if hit["implemented"] else "  [planned]"
        print(f"{hit['id']:<38} {hit['title']}{flag}")
        if args.verbose:
            print(f"    {hit['summary']}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    from .catalog.explain import explain_concept

    print(explain_concept(args.concept, language=args.language))
    return 0


def _cmd_proofs(args: argparse.Namespace) -> int:
    from .proofs import list_proofs

    for entry in list_proofs():
        mark = " " if entry.is_written else "*"
        concepts = ", ".join(entry.concept_ids)
        print(f"{mark} {entry.id:42s} {entry.title}")
        if concepts:
            print(f"    labs: {concepts}")
    if any(not e.is_written for e in list_proofs()):
        print("\n* = catalogued but not written yet")
    return 0


def _cmd_proof(args: argparse.Namespace) -> int:
    import visualmetrics as vm

    rendered = vm.proof(args.proof, language=args.language)
    if args.json:
        print(json.dumps(rendered.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(rendered.to_text())
    if args.checks:
        from .proofs import proof as _proof

        spec = _proof(args.proof)
        if not spec.checks:
            print("\nNo numerical checks are attached to this proof.")
            return 0
        print("\nNumerical checks (these test the code, not the theorem):")
        failed = 0
        for outcome in spec.run_checks():
            status = "pass" if outcome.passed else "FAIL"
            failed += not outcome.passed
            print(f"  [{status}] {outcome.id}: {outcome.detail}")
        if failed:
            return 1
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    import visualmetrics as vm
    from .core.controls import ControlKind

    spec = vm.concept(args.concept)
    params: dict[str, Any] = {}
    controls = spec.control_map
    for assignment in args.set or []:
        if "=" not in assignment:
            print(f"--set expects key=value, got {assignment!r}", file=sys.stderr)
            return 2
        key, raw = assignment.split("=", 1)
        key = key.strip()
        value: Any = raw.strip()
        control = controls.get(key)
        if control is not None and control.kind not in (ControlKind.SELECT,
                                                        ControlKind.MULTISELECT):
            try:
                value = control.coerce(value)
            except Exception:  # noqa: BLE001 - fall back to the raw string
                pass
        params[key] = value

    result = vm.lab(
        args.concept,
        scenario=args.scenario,
        seed=args.seed,
        language=args.language or vm.get_config().language,
        theme=args.theme or vm.get_config().theme,
        **params,
    )
    print(result.summary())
    if args.code:
        print()
        print(result.code)
    if args.export:
        from .export.images import export_result

        paths = export_result(result, args.export)
        for path in paths:
            print(f"exported: {path}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    report = build_doctor_report()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_doctor_report(report))
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    import visualmetrics as vm

    print(json.dumps(vm.info(), indent=2, ensure_ascii=False))
    return 0


def _cmd_paths(args: argparse.Namespace) -> int:
    from .catalog.paths import list_paths, path_payload
    from .i18n.translator import set_language

    if args.language:
        set_language(args.language)
    if args.path_id:
        payload = path_payload(args.path_id)
        print(payload["title"])
        print(payload["summary"])
        print()
        for i, step in enumerate(payload["steps"], start=1):
            flag = "" if step["status"] != "planned" else "  [planned]"
            print(f"  {i}. {step['title']:<44} {step['id']}{flag}")
        return 0
    from .i18n.translator import get_translator

    tr = get_translator()
    for path in list_paths():
        print(f"{path.id:<34} {tr.t(path.title_key)}  ({path.level})")
    return 0


def _cmd_examples(args: argparse.Namespace) -> int:
    print(EXAMPLES)
    return 0


EXAMPLES = """VisualMetrics - example session

  import visualmetrics as vm

  vm.configure(language="ar", terminology="bilingual", theme="classroom")

  # a lab with a scenario preset
  r = vm.lab("regression.simple_linear", scenario="omitted_variable_bias", n=200, seed=7)
  print(r.summary())
  r.show()

  # the Python that reproduces a GUI state
  print(r.code)

  # compare two estimators on identical data
  ols, iv = vm.compare("regression.simple_linear", "econometrics.endogeneity_iv", n=400)

  # what should I learn first?
  vm.prerequisites("causal.did")
  vm.learning_path("econometrics.endogeneity_iv")

  # cross-language search
  vm.search("القوة الإحصائية")
  vm.search("puissance")

From the shell:

  visualmetrics                       launch the GUI
  visualmetrics list --domain causal_inference
  visualmetrics search "unit root"
  visualmetrics show timeseries.stationarity
  visualmetrics run inference.power --set alpha=0.01 --set n=80 --code
  visualmetrics doctor
"""


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visualmetrics",
        description="Multilingual interactive visual laboratory for statistics, "
                    "econometrics and AI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run 'visualmetrics examples' for a worked session.",
    )
    parser.add_argument("--version", action="version", version=f"visualmetrics {__version__}")
    parser.add_argument("-l", "--language", choices=("en", "ar", "fr"),
                        help="language for this command")
    sub = parser.add_subparsers(dest="command")

    gui = sub.add_parser("gui", help="launch the local GUI")
    gui.add_argument("--host", default=None)
    gui.add_argument("--port", type=int, default=None)
    gui.add_argument("--native", action="store_true", help="open a native desktop window")
    gui.add_argument("--no-browser", action="store_true")
    gui.set_defaults(func=_cmd_gui)

    lst = sub.add_parser("list", help="list catalogued concepts")
    lst.add_argument("--domain", default=None)
    lst.add_argument("--level", default=None, choices=("beginner", "intermediate",
                                                       "advanced", "phd"))
    lst.add_argument("--all", action="store_true", help="include planned concepts")
    lst.add_argument("--json", action="store_true")
    lst.set_defaults(func=_cmd_list)

    search = sub.add_parser("search", help="search concepts in any supported language")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=12)
    search.add_argument("-v", "--verbose", action="store_true")
    search.set_defaults(func=_cmd_search)

    show = sub.add_parser("show", help="explain one concept")
    show.add_argument("concept")
    show.set_defaults(func=_cmd_show)

    run = sub.add_parser("run", help="run a lab and print its numeric read-out")
    run.add_argument("concept")
    run.add_argument("--scenario", default=None)
    run.add_argument("--set", action="append", metavar="KEY=VALUE")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--theme", default=None)
    run.add_argument("--code", action="store_true", help="also print the generated Python")
    run.add_argument("--export", default=None, metavar="PATH",
                     help="export the figures (.html, .png, .svg or .json)")
    run.set_defaults(func=_cmd_run)

    proofs = sub.add_parser("proofs", help="list the catalogued proofs")
    proofs.set_defaults(func=_cmd_proofs)

    proof = sub.add_parser("proof", help="print one proof step by step")
    proof.add_argument("proof")
    proof.add_argument("--checks", action="store_true",
                       help="also run the numerical checks of the proved identities")
    proof.add_argument("--json", action="store_true")
    proof.set_defaults(func=_cmd_proof)

    doctor = sub.add_parser("doctor", help="check the environment and optional extras")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=_cmd_doctor)

    info = sub.add_parser("info", help="version, catalog and configuration as JSON")
    info.set_defaults(func=_cmd_info)

    paths = sub.add_parser("paths", help="list or show curated learning paths")
    paths.add_argument("path_id", nargs="?", default=None)
    paths.set_defaults(func=_cmd_paths)

    examples = sub.add_parser("examples", help="print a worked example session")
    examples.set_defaults(func=_cmd_examples)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.language:
        from .i18n.translator import set_language

        set_language(args.language)
    if getattr(args, "func", None) is None:
        # bare `visualmetrics` launches the GUI
        args.host = args.port = None
        args.native = False
        args.no_browser = False
        return _cmd_gui(args)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        from .core.exceptions import VisualMetricsError

        if isinstance(exc, VisualMetricsError):
            from .i18n.translator import get_translator

            print(f"error: {exc.localized(get_translator())}", file=sys.stderr)
            if isinstance(exc, ImportError):
                return 3
            return 1
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
