################################################################################
# Paddy ecosystem CH4 response workflow
################################################################################

source("analysis_pipeline.R")

run_full_workflow(
  input_file = "paddy.xlsx",
  ecosystem = "paddy",
  output_root = "outputs"
)
