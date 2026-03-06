*** Settings ***
Library    BuiltIn

*** Test Cases ***
Test Runs But Suite Teardown Will Fail
    [Tags]    chunking    failing-teardown
    Log    Suite setup passed so this test runs normally

Another Test That Runs Despite Teardown Failure
    [Tags]    chunking    failing-teardown
    Log    Suite teardown failure does not prevent test execution
    Fail    Intentional test failure alongside failing suite teardown
