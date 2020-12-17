from imc.types import Path

# project
DEFAULT_PROJECT_NAME = "project"
DEFAULT_SAMPLE_NAME_ATTRIBUTE = "sample_name"
DEFAULT_SAMPLE_GROUPING_ATTRIBUTEs = [DEFAULT_SAMPLE_NAME_ATTRIBUTE]
DEFAULT_TOGGLE_ATTRIBUTE = "toggle"
DEFAULT_PROCESSED_DIR_NAME = Path("processed")
DEFAULT_RESULTS_DIR_NAME = Path("results")
DEFAULT_PRJ_SINGLE_CELL_DIR = Path("single_cell")
DEFAULT_ROI_NAME_ATTRIBUTE = "roi_name"
DEFAULT_ROI_NUMBER_ATTRIBUTE = "roi_number"

# # processed directory structure
SUBFOLDERS_PER_SAMPLE = True
ROI_STACKS_DIR = Path("tiffs")
ROI_MASKS_DIR = Path("tiffs")
ROI_UNCERTAINTY_DIR = Path("uncertainty")
ROI_SINGLE_CELL_DIR = Path("single_cell")

# sample
DEFAULT_SAMPLE_NAME = "sample"
DEFAULT_ROI_NAME_ATTRIBUTE = "roi_name"
DEFAULT_ROI_NUMBER_ATTRIBUTE = "roi_number"
DEFAULT_TOGGLE_ATTRIBUTE = "toggle"

# roi
SUBFOLDERS_PER_SAMPLE = True
DEFAULT_ROI_NAME = "roi"
ROI_STACKS_DIR = Path("tiffs")
ROI_MASKS_DIR = Path("tiffs")
ROI_UNCERTAINTY_DIR = Path("uncertainty")
ROI_SINGLE_CELL_DIR = Path("single_cell")

# graphics
FIG_KWS = dict(dpi=300, bbox_inches="tight")
