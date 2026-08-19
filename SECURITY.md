# Security policy

## Supported versions

GPN is currently in the `0.9` pre-release series. Security fixes are applied to the
latest published `0.9.x` release or pre-release; older `0.x` lines are unsupported.

## Reporting a vulnerability

Email `gbenegas@berkeley.edu` for sensitive dependency, serialization,
model-loading, or code-execution issues. Do not open a public issue until
maintainers have assessed disclosure and remediation.

Scientific correctness problems—such as an unexpected score, coordinate mismatch,
or model regression—are important but are usually not security vulnerabilities.
Report those through the public issue tracker unless they also create a security or
privacy risk.

Never include private genomic data, access tokens, proprietary model weights, or
institutional filesystem details in either kind of report.
