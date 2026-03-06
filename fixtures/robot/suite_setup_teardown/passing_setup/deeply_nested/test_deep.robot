*** Settings ***
Library    BuiltIn

*** Test Cases ***
Deep Test That Passes
    [Tags]    chunking    deeply-nested
    Log    Three levels deep - setup passed at every level

Deep Test That Fails
    [Tags]    chunking    deeply-nested
    Log    Three levels deep - this test fails intentionally
    Fail    Intentional failure at deep nesting level
