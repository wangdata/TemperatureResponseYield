################################################################################
# Wetland ecosystem CH4 response workflow
################################################################################

source("analysis_pipeline.R")

run_full_workflow(
  input_file = "wetland.xlsx",
  ecosystem = "wetland",
  output_root = "outputs"
)
