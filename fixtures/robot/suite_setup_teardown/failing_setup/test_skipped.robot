*** Settings ***
Library    BuiltIn

*** Test Cases ***
Test Never Reached Due To Suite Setup Failure
    [Tags]    chunking    failing-setup
    Log    This test is never executed
    Fail    Should not appear in results

Another Test Never Reached
    [Tags]    chunking    failing-setup
    Log    This test is also never executed
