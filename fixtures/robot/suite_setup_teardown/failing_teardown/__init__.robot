*** Settings ***
Library    BuiltIn

Suite Setup       Log    Suite setup: PASS - tests will run
Suite Teardown    Suite Teardown Fails

*** Keywords ***
Suite Teardown Fails
    Log    Suite teardown about to fail
    Fail    Intentional suite teardown failure - tests already ran
