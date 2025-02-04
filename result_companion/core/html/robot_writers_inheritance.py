from pathlib import Path
from unittest.mock import patch

from robot.api import ExecutionResult
from robot.conf import RebotSettings
from robot.htmldata import (
    HtmlFileWriter,
)
from robot.reporting.logreportwriters import LogWriter, RobotModelWriter
from robot.reporting.resultwriter import Results, ResultWriter
from robot.utils import file_writer


LLM_LOG = Path(__file__).parent / "templates" / "llm_template.html"


class LLMHtmlTemplate:  # HtmlTemplate
    def __init__(self, path):
        self._path = path

    def __iter__(self):
        with open(self._path, "r", encoding="UTF-8") as file:
            for line in file:
                yield line


class LLMHtmlFileWriter(HtmlFileWriter):
    def __init__(self, output, model_writer):
        super().__init__(output, model_writer)

    def write(self, template: "Path|str"):
        if not isinstance(template, Path):
            template = Path(template)
        writers = self._get_writers(template.parent)
        for line in LLMHtmlTemplate(template):
            for writer in writers:
                if writer.handles(line):
                    writer.write(line)
                    break


class LLMLogWriter(LogWriter):
    usage = "LLM_log"

    def write(self, path: "Path|str", config):
        if isinstance(path, str):
            path = Path(path)
        self._write_file(path, config, LLM_LOG)
        if self._js_model.split_results:
            self._write_split_logs(path)

    def _write_file(self, path: Path, config, template):
        outfile = (
            file_writer(path, usage=self.usage) if isinstance(path, Path) else path
        )  # unit test hook
        with outfile:
            model_writer = RobotModelWriter(outfile, self._js_model, config)
            writer = LLMHtmlFileWriter(outfile, model_writer)
            writer.write(template)


class LLMResultWriter(ResultWriter):
    def __init__(self, *sources):
        super().__init__(*sources)

    def write_results(self, settings=None, **options):
        settings = settings or RebotSettings(options)
        results = Results(settings, *self._sources)
        config = dict(settings.log_config, minLevel=results.js_result.min_level)
        self._write_llm_log(results.js_result, settings.log, config)

    def _write_llm_log(self, js_result, path, config):
        self._write("Log", LLMLogWriter(js_result).write, path, config)


def create_base_llm_result_log(
    input_result_path: "Path|str", llm_output_path: "Path|str"
):
    results = ExecutionResult(input_result_path)
    writer = LLMResultWriter(results)
    with patch("robot.htmldata.htmlfilewriter.HtmlTemplate", new=LLMHtmlTemplate):
        writer.write_results(report=None, log=llm_output_path, xunit=None)
