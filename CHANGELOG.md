# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 24.11.17

## Added
- Add `SfkitLDFiltering` for simulating the SNP selection in sfkit.
- Add `BatchQRSolver` for linear regression.

## Fixed
- Avoid re-compiling when using `jax.jit`.
- Use `numpy` when calculating the statistics for getting the more accurate results.
- Ensure the sample order in covariates and phenotypes are the same.

## Removed
- The `BlockDiagonalMatrix` implementation in linear regression.
- Remove `pmap` implementation in regressions.
- Remove `examples/` using http protocol.


## [0.0.0] - 24.05.28
The first release version of RAFAEL.
