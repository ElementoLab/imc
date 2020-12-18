epilog = "https://github.com/ElementoLab/imc"
cli_config = {
    "main": {
        "prog": "imc",
        "description": "A package for the analysis of Imaging Mass Cytometry data.",
        "epilog": epilog,
    },
    "subcommands": {
        "inspect": {
            "prog": "imc inspect",
            "description": "Inspect MCD files and extract metadata.",
            "epilog": epilog,
        },
        "prepare": {
            "prog": "imc prepare",
            "description": "Prepare project directory from MCD files.",
            "epilog": epilog,
        },
        "segment": {
            "prog": "imc segment",
            "description": "Segment image stacks.",
            "epilog": epilog,
        },
    },
}
