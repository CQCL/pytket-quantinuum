# pytket-quantinuum

[![Slack](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white)](https://tketusers.slack.com/join/shared_invite/zt-18qmsamj9-UqQFVdkRzxnXCcKtcarLRA#)
[![Stack Exchange](https://img.shields.io/badge/StackExchange-%23ffffff.svg?style=for-the-badge&logo=StackExchange)](https://quantumcomputing.stackexchange.com/tags/pytket)

[Pytket](https://docs.quantinuum.com/tket/api-docs/index.html) is a python module for interfacing
with tket, a quantum computing toolkit and optimising compiler developed by Quantinuum.

`pytket-quantinuum` is an extension to `pytket` that allows `pytket` circuits to
be executed on Quantinuum's quantum devices.

Some useful links:
- [API Documentation](https://docs.quantinuum.com/tket/extensions/pytket-quantinuum/)

## Getting started

`pytket-quantinuum` is available for Python 3.10, 3.11 and 3.12, on Linux, MacOS
and Windows. To install, run:

```shell
pip install pytket-quantinuum
```

This will install `pytket` if it isn't already installed, and add new classes
and methods into the `pytket.extensions` namespace.

## Examples

For example usage of features, see the [pytket-quantinuum section](https://docs.quantinuum.com/systems/trainings/getting_started/pytket_quantinuum/pytket_quantinuum.html) of the documentation website.

The [knowledge articles](https://docs.quantinuum.com/systems/trainings/knowledge_articles/ka_index.html) also detail experiments using pytket-quantinuum.

## Bugs, support and feature requests

Please file bugs and feature requests on the Github
[issue tracker](https://github.com/CQCL/pytket-quantinuum/issues).

There is also a Slack channel for discussion and support. Click [here](https://tketusers.slack.com/join/shared_invite/zt-18qmsamj9-UqQFVdkRzxnXCcKtcarLRA#/shared-invite/email) to join.

## Hardware Support

In case of questions about the hardware you can contact QCsupport@quantinuum.com.

## Development

To install this extension in editable mode, simply change to this directory, and run:

```shell
pip install -e .
```

## Contributing

Pull requests are welcome. To make a PR, first fork the repo, make your proposed
changes on the `main` branch, and open a PR from your fork. If it passes
tests and is accepted after review, it will be merged in.

### Code style

#### Formatting

All code should be formatted using
[black](https://black.readthedocs.io/en/stable/), with default options. This is
checked on the CI. The CI is currently using version 20.8b1.

#### Type annotation

On the CI, [mypy](https://mypy.readthedocs.io/en/stable/) is used as a static
type checker and all submissions must pass its checks. You should therefore run
`mypy` locally on any changed files before submitting a PR. Because of the way
extension modules embed themselves into the `pytket` namespace this is a little
complicated, but it should be sufficient to run the script `modules/mypy-check`
(passing as a single argument the root directory of the module to test). The
script requires `mypy` 0.800 or above.

#### Linting

We use [pylint](https://pypi.org/project/pylint/) on the CI to check compliance
with a set of style requirements (listed in `.pylintrc`). You should run
`pylint` over any changed files before submitting a PR, to catch any issues.

### Tests

To run the tests:

1. `cd` into the `tests` directory;
2. ensure you have installed `pytest`, `hypothesis`, and any modules listed in
the `test-requirements.txt` file (all via `pip`);
3. run `pytest`.

When adding a new feature, please add a test for it. When fixing a bug, please
add a test that demonstrates the fix.
