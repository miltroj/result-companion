*** Settings ***
Library    BuiltIn


*** Variables ***
${test_var}    abcd



*** Test Cases ***
Test Neasted Test Case
    Log    ${test_var}
