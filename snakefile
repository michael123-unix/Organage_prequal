from pathlib import Path
MASTER_DIR = Path(workflow.current_basedir)


include: "ACE/snakefile",
include: "LGB/snakefile",
include: "Linear_models/snakefile"

rule compare:
	input:
		LGB_top_50_genes = MASTER_DIR / "LGB/Results/Analysis/top_50_genes_global.csv",
		ACE_top_50_genes = MASTER_DIR / "ACE/Explain_Results/top_50_attr_global.csv",
		LGB_top_50_3m = MASTER_DIR / "LGB/Results/Analysis/top_50_genes_3m.csv",
		LGB_top_50_18m = MASTER_DIR / "LGB/Results/Analysis/top_50_genes_18m.csv",
		LGB_top_50_24m = MASTER_DIR / "LGB/Results/Analysis/top_50_genes_24m.csv",
		ACE_top_50_3m = MASTER_DIR / "ACE/Explain_Results/top_50_attr_3m.csv",
		ACE_top_50_18m = MASTER_DIR / "ACE/Explain_Results/top_50_attr_18m.csv",
		ACE_top_50_24m = MASTER_DIR / "ACE/Explain_Results/top_50_attr_24m.csv",
		Log_3m = MASTER_DIR / "Linear_models/Results/top_50_genes_logistic_3m.csv",
		Log_18m = MASTER_DIR / "Linear_models/Results/top_50_genes_logistic_18m.csv",
		Log_24m = MASTER_DIR / "Linear_models/Results/top_50_genes_logistic_24m.csv"

	output:
		venn_models_3m = MASTER_DIR / "Comparison/Results/venn_models_3m.png",
		venn_models_18m = MASTER_DIR / "Comparison/Results/venn_models_18m.png",
		venn_models_24m = MASTER_DIR / "Comparison/Results/venn_models_24m.png"

	conda:
		str(MASTER_DIR / "Comparison/Envs/Comparison.yaml")

	script:
		str(MASTER_DIR / "Comparison/Scripts/Comparison.py")		
