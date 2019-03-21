from allennlp.commands import main as main_allennlp

from contexteval.commands.error_analysis import ErrorAnalysis


def main(prog: str = None) -> None:
    subcommand_overrides = {
        "error-analysis": ErrorAnalysis()
    }
    main_allennlp(prog, subcommand_overrides=subcommand_overrides)
