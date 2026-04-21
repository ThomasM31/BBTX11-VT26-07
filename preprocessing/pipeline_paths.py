from pathlib import Path
import os

class PipelinePaths:

    def _ensure_paths_exist(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Path):
                attr.mkdir(parents=True, exist_ok=True)

    def __init__(self, shared_dir_mode: bool, run_vars: str) -> None:

        if shared_dir_mode:
            # To read from and write to the shared folder
            base_path = Path("/data/shared/alzgene26/data")
        else:
            ### To read from and write to current users folders
            user = str(os.environ.get('USER')) or str(os.environ.get('USERNAME'))
            base_path = Path("/data/users") / user / "kand/data/"

        processed_data = "processed_data"

        # base experimental data before preprocessing
        self.conv_data_path       = base_path / "conv_data"
        # AD status, age etc
        self.metadata_path        = base_path / "supplementary_data"

        # intermediate data to save and read in pipeline
        self.gene_expr_count_path = base_path / processed_data / "expr_counts"
        self.genes_keep_path      = base_path / processed_data / "filter_genes"
        self.hvg_lists_path       = base_path / processed_data / "hvg_lists"
        self.hvg_common_path      = base_path / processed_data / "hvg_common"
        
        # to save umaps
        self.figures_path         = base_path / "figures"
        
        # to save smaller files that are faster to read
        self.test_data_path       = base_path / processed_data / "test_data"
        
        # reactome assets
        self.pathway_data_path    = base_path / 'PathwayData'
        
        # for saving fully preprocessed data
        self.compl_base = base_path / processed_data / 'completed'
        
        self.compl_full_pipe_path = self.compl_base / "full_pipeline" / run_vars
        self.compl_path_danish    = self.compl_base / "danish"
        self.compl_hvg_non_pseudo = self.compl_base / "hvg_non_pseudo" / run_vars
        self.compl_pseudo_non_hvg = self.compl_base / "pseudo_non_hvg" / run_vars

        self.extra_test_path = base_path / processed_data/ 'test_path'

        # for every path, creates path if it does not exist
        self._ensure_paths_exist()