from invoke import task

@task
def poetry_install_dev(c):
    c.run("poetry install --with=dev")


@task
def test_coverage(c):
    c.run("pytest --cov=result_companion tests/unittests")


@task
def test_execute_e2e_tests(c):
    c.run("robot --loglevel TRACE  tests/integration/e2e/")
