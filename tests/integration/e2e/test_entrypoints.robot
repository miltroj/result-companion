*** Settings ***
Library    BuiltIn
Library    OperatingSystem
Library    Process

*** Variables ***
${tmp_folder_name}    ${CURDIR}/../../../_tmp_artefacts/e2e
${ollama_log_file}    ${tmp_folder_name}/llm_log_ollama.html
${custom_model_log_file}    ${tmp_folder_name}/llm_log_custom_model.html
${output_file_with_failures_to_analise}    ${CURDIR}/../../../output.xml
${entrypoint_folder}    ${CURDIR}/../../../test_result_companion/entrypoints
${ollama_entrypoint}    ${entrypoint_folder}/run_ollama.py
${custom_model_entrypoint}    ${entrypoint_folder}/run_custom.py



*** Test Cases ***
Ollama Local Model Run Should Succede
    File Should Exist    ${output_file_with_failures_to_analise}
    Cleanup Test Artefacts
    ${result}    Run Process    poetry run python ${ollama_entrypoint} -o ${output_file_with_failures_to_analise} -r ${ollama_log_file}    shell=True
    File Should Exist    ${ollama_log_file}
    ${file_content}    Get File    ${ollama_log_file}
    Should Contain    ${file_content}    LLM Response

#Custom Local Model Run Should Succede
#    File Should Exist    ${output_file_with_failures_to_analise}
#    Cleanup Test Artefacts
#    ${result}    Run Process    python ${custom_model_entrypoint} -o ${output_file_with_failures_to_analise} -r ${custom_model_log_file}
#    File Should Exist    ${custom_model_log_file}
#    ${file_content}    Get File    ${custom_model_log_file}
#    Should Contain    ${file_content}    LLM Response

*** Keywords ***
Cleanup Test Artefacts
    Run Keyword And Continue On Failure     Remove File    ${ollama_log_file}
    Run Keyword And Continue On Failure     Remove File    ${custom_model_log_file}
