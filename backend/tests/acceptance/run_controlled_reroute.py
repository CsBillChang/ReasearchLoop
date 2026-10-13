"""Run the AG1 controlled-PDF-failure case with the real DeepSeek provider.

The task is submitted through the HTTP API first. This script is the isolated
Worker adapter used only to label and inject one deterministic PDF failure.
"""
from __future__ import annotations
import argparse
from research_loop.config import get_settings
from research_loop.providers.pdf import PDFProvider
from research_loop.storage.files import FileStore
from research_loop.workers.executor import TaskExecutor


class ControlledPDFProvider:
    def __init__(self, delegate: PDFProvider):
        self.delegate = delegate
        self.failed_once = False

    def ingest(self, source: str, paper_id: str):
        if not self.failed_once:
            self.failed_once = True
            raise RuntimeError("CONTROLLED_PDF_FAILURE: first PDF acquisition intentionally denied for reroute acceptance")
        return self.delegate.ingest(source, paper_id)

    def pages(self, document, start_page, page_count):
        return self.delegate.pages(document, start_page, page_count)

    def close(self):
        self.delegate.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    settings = get_settings()

    def make_pdf(current_settings, files):
        return ControlledPDFProvider(PDFProvider(
            files, current_settings.parser_name, current_settings.parser_version,
            current_settings.parser_config, current_settings.request_timeout_seconds,
            current_settings.max_pdf_bytes,
        ))

    TaskExecutor(settings, pdf_factory=make_pdf).execute(args.task_id)


if __name__ == "__main__":
    main()
