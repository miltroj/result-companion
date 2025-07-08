from invoke import task


@task
def poetry_install_dev(c):
    """Install poetry with dev dependencies."""
    c.run("poetry install --with=dev")


@task
def unittests(c):
    c.run("poetry run pytest -vv tests/unittests/")


@task
def test_coverage(c):
    c.run("poetry run pytest --cov=result_companion tests/unittests")


@task
def test_execute_e2e_tests(c):
    """TODO: implement."""
    c.run("poetry run robot --loglevel TRACE  tests/integration/e2e/")


@task
def run_result_companion_test(c):
    """Run Result Companion with predefined input and output paths."""
    c.run(
        "poetry run result-companion analyze -o examples/run_test/output.xml -r log_with_results.html"
    )
