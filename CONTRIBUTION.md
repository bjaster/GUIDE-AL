# Contributing guide lines

We appreciate all contributions! If you are planning to contribute bug-fixes or
documentation improvements, please go ahead and open a
[pull request (PR)](https://ds-juist.init.th-owl.de/ai4scada/guide_active_learning/-/merge_requests)
. If you are planning to contribute new features, please open an
[issue](https://ds-juist.init.th-owl.de/ai4scada/guide_active_learning/-/issues) and
discuss the feature with us first.

To start working on `guide_active_learning` clone the repository from GitHub and set up
the development environment

```shell
git clone https://ds-juist.init.th-owl.de/ai4scada/guide_active_learning
cd guide_active_learning
python -m pip install --user virtualenv (if not installed)
virtualenv .venv --prompt='(active_learning-dev) '
source .venv/bin/activate (on Linux) or .venv\Scripts\activate (on Windows)
pip install doit
doit install
```

Every PR is subjected to multiple checks that it has to pass before it can be merged.
The checks are performed through [doit](https://pydoit.org/). Below you can find details
and instructions how to run the checks locally.

## Code format and linting

`guide_active_learning` uses [ufmt](https://ufmt.omnilib.dev/en/stable/) to format
Python code, and [flake8](https://flake8.pycqa.org/en/stable/) to enforce
[PEP8](https://www.python.org/dev/peps/pep-0008/) compliance.

Furthermore, `guide_active_learning` is
[PEP561](https://www.python.org/dev/peps/pep-0561/) compliant and checks the type
annotations with [mypy](http://mypy-lang.org/) .

To automatically format the code, run

```shell
doit format
```

Instead of running the formatting manually, you can also add
[pre-commit](https://pre-commit.com/) hooks. By running

```shell
pre-commit install
```

once, an equivalent of `doit format` is run everytime you `git commit` something.

Everything that cannot be fixed automatically, can be checked with

```shell
doit lint
```
