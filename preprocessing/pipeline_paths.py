from pathlib import Path
import os

class PipelinePaths:

    def _ensure_paths_exist(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Path):
                attr.mkdir(parents=True, exist_ok=True)

    def __init__(self, 
                 shared_dir_mode: bool, 
                 full_pipeline_run_vars: str = '',
                 pseudo_non_hvg_run_vars: str = '',
                 hvg_non_pseudo_run_vars: str = ''
                 ) -> None:

        if shared_dir_mode:
            # To read from and write to the shared folder
            base_path = Path("/data/shared/alzgene26") 
            data_path = base_path / "data"
        else:
            ### To read from and write to current users folders
            user = str(os.environ.get('USER')) or str(os.environ.get('USERNAME'))
            data_path = Path("/data/users") / user / "kand/data/"

        processed_data = "processed_data"

        # base experimental data before preprocessing
        self.conv_data_path       = data_path / "conv_data"
        # AD status, age etc
        self.metadata_path        = data_path / "supplementary_data"

        # intermediate data to save and read in pipeline
        self.gene_expr_count_path = data_path / processed_data / "expr_counts"
        self.genes_keep_path      = data_path / processed_data / "filter_genes"
        self.hvg_lists_path       = data_path / processed_data / "hvg_lists"
        self.hvg_common_path      = data_path / processed_data / "hvg_common"
        
        # to save umaps
        self.figures_path         = data_path / "figures"
        self.full_pipeline_figs   = self.figures_path / "full_pipeline" / full_pipeline_run_vars
        self.danish_figs          = self.figures_path / "danish" 
        self.pseudo_non_hvg_figs   = self.figures_path / "pseudo_non_hvg" / pseudo_non_hvg_run_vars
        self.hvg_non_pseudo_figs  = self.figures_path / "hvg_non_pseudo" / hvg_non_pseudo_run_vars
        
        # to save smaller files that are faster to read
        self.test_data_path       = data_path / processed_data / "test_data"
        
        # reactome assets
        self.pathway_data_path    = base_path / "PathwayData"
        
        # for saving fully preprocessed data
        self.compl_base = data_path / processed_data / "completed"
        
        self.compl_full_pipe_path = self.compl_base / "full_pipeline" / full_pipeline_run_vars
        self.compl_path_danish    = self.compl_base / "danish"
        self.compl_hvg_non_pseudo = self.compl_base / "hvg_non_pseudo" / hvg_non_pseudo_run_vars
        self.compl_pseudo_non_hvg = self.compl_base / "pseudo_non_hvg" / pseudo_non_hvg_run_vars

        self.extra_test_path = data_path / processed_data/ "test_path"

        # for every path, creates path if it does not exist
        self._ensure_paths_exist()