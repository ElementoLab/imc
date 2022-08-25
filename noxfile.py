import nox

python_versions = [
    "3.8",
    "3.9",
    "3.10",
]

# TODO: annotate these with explanation
ignore_rules = [
    "E501",
    "F401",
    "F841",
    "W503",
    "E402",
    "E203",
    "E266",
    "E722",  # bare except
]

exclude_directories = [
    "tests",
]


@nox.session(python=python_versions)
def lint(session):
    session.install("flake8")
    session.run(
        "flake8",
        "--ignore",
        ",".join(ignore_rules),
        "--exclude",
        ",".join(exclude_directories),
        "imc/",
    )


@nox.session(python=python_versions)
def test(session):
    session.install(".[dev]")
    session.run("pytest")
