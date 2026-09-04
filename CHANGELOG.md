# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- Repository hardening: actions pinned to commit SHAs, Dependabot, CODEOWNERS, code of conduct, issue and pull request templates, disclaimer.
- Dependency management moved to uv with a committed lockfile; CI installs with `uv sync --locked`.

## [0.1.0] - 2026-09-04

### Added
- `mlkit.preprocessing`: `clean_dataframe`, `remove_outliers`, `split_features_target`,
  `split_data`, and the scikit-learn compatible `Preprocessor` transformer.
- `mlkit.deep`: `MLPClassifier`, `MLPRegressor`, `CNNClassifier` with early stopping,
  device auto-detection and safe save / load; `build_mlp` and `build_cnn` builders.
- Worked example on the breast cancer dataset in `examples/`.
- Test suite, GitHub Actions CI (lint and tests on Python 3.10 and 3.12), MIT license,
  security policy.
