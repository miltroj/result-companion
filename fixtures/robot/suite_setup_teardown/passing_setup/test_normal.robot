*** Settings ***
Library    BuiltIn

*** Test Cases ***
Test That Passes After Suite Setup
    [Tags]    chunking    passing-setup
    Log    Suite setup passed - this test should be individually analyzable

Test That Fails After Suite Setup
    [Tags]    chunking    passing-setup
    Log    Suite setup passed - this test fails intentionally
    Fail    Intentional failure to verify chunking picks up individual test
