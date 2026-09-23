# Contributing to OpenJeff

Useful contributions include reproducible bug reports, independent evaluations,
clearer documentation, and focused code changes. Financial support is optional.

For a bug, include the command, package versions, expected result, and a minimal
synthetic input. Remove credentials and private data before opening an issue.

Run the numerical tests with Python 3.10 or later:

```bash
python -m unittest discover -s tests -v
python -m scripts.validate_release
python -m scripts.validate_hybrid_v2
```

Optional model-library tests skip when those dependencies are absent. The saved
score validators require no GPU or paid API. GPU inference and training have
separate pinned requirements in `configs/`; never infer a new benchmark result
from a saved-score check.

Keep model or prompt selection on development data. Refit calibration when the
scorer changes, disclose shared synthetic templates and repeated states, and
report regressions as well as improvements. Keep unrelated changes separate.

Do not submit model weights, private datasets, payment information, access tokens,
or cloud-account logs in issues or pull requests. Discuss new weight releases in
an issue before adding binaries. Preserve upstream attribution and license notices.

Contributions are provided under the repository's Apache-2.0 license; third-party
materials retain their own terms.
