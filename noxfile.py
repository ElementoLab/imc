import nox

# TODO: annotate these with explanation
ignores = [
    "E501",
    "F401",
    "F841",
    "W503",
    "E402",
    "E203",
    "E266",
    "E722",  # bare except
]

excludes = [
    "tests",
]


@nox.session(python=["3.8", "3.9", "3.10"])
def lint(session):
    session.install("flake8")
    session.run(
        "flake8",
        "--ignore",
        ",".join(ignores),
        "--exclude",
        ",".join(excludes),
        "imc/",
    )


@nox.session(python=["3.8", "3.9", "3.10"])
def test(session):
    session.install(".[dev]")
    session.run("pytest")
