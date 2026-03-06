*** Settings ***
Library    BuiltIn

Suite Setup       Suite Setup Passes
Suite Teardown    Log    Suite teardown: PASS

*** Keywords ***
Suite Setup Passes
    Log    Suite setup: PASS
    Set Suite Variable    ${SUITE_READY}    True
