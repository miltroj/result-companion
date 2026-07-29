*** Settings ***
Library    Browser    run_on_failure=Capture Embedded Screenshot
Suite Setup    New Browser    chromium    headless=${TRUE}
Suite Teardown    Close Browser

*** Variables ***
${TARGET_URL}      https://result-companion.com
${MISSING_TEXT}    Result Companion OCR harness expected text

*** Test Cases ***
Actual Result Companion Page Captures Embedded Screenshot
    [Tags]    vision    screenshot    browser
    New Page    ${TARGET_URL}
    Wait For Load State    load
    Take Screenshot    EMBED
    Get Text    body    contains    ${MISSING_TEXT}

*** Keywords ***
Capture Embedded Screenshot
    Take Screenshot    EMBED
