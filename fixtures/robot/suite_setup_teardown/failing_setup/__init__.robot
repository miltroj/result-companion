*** Settings ***
Library    BuiltIn

Suite Setup       Suite Setup Fails
Suite Teardown    Log    Suite teardown runs even after setup failure

*** Keywords ***
Suite Setup Fails
    Log    Suite setup about to fail
    Fail    Intentional suite setup failure - children should be skipped by extract_analyzable_items
