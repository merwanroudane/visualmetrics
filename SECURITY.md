# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | ✅ |

Until 1.0, only the latest release receives fixes.

## Reporting a vulnerability

Email **merwanroudane920@gmail.com** with `[SECURITY]` in the subject. Please do
not open a public issue for a vulnerability.

Include what you can: the version, how to reproduce it, and what an attacker
could achieve. You should get an acknowledgement within a few days, an
assessment within two weeks, and credit in the changelog unless you prefer
otherwise.

## What is in scope

VisualMetrics is a local teaching tool, so the realistic risks are narrow:

- **The local web server.** `visualmetrics gui` binds to `127.0.0.1` by
  default. Anything that lets a remote page reach it, or that executes content
  from a page it renders, is in scope.
- **Loading files.** Configuration files (`load_config`) and exported reports
  are parsed with the standard library and `yaml.safe_load`. Any path that
  allows code execution, or that writes outside the path you asked for, is in
  scope.
- **Plugins.** Third-party concept plugins are discovered through entry points
  and run in-process. A way for a plugin to escalate beyond what installing it
  already implies is in scope.
- **Dependency vulnerabilities** that this package's usage makes exploitable.

## What is not in scope

- Running `visualmetrics gui --host 0.0.0.0` on an untrusted network. That is
  an explicit choice to expose a development server; do not make it.
- Installing a malicious plugin, or executing generated code you did not read.
  Installing a package already grants it execution.
- Resource exhaustion from parameters you chose yourself - a large Monte Carlo
  study is slow on purpose.

## A note on the numbers

A wrong statistical result is not a security issue, but it is treated with the
same seriousness. Report it as a normal issue and it will be prioritised.
