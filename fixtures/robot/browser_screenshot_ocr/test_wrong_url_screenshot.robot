*** Settings ***
Library    Browser    run_on_failure=Capture Embedded Screenshot
Suite Setup    New Browser    chromium    headless=${TRUE}
Suite Teardown    Close Browser

*** Variables ***
${TARGET_URL}      file://${CURDIR}/ocr_fixture_page.html
${MISSING_TEXT}    Intentional missing OCR harness text

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
