import pandas as pd
import shap

def draw_shap():

    shap_df = pd.read_csv(figpath / f'/data/shared/alzgene26/data/figures/shap/real_shap_values_260508_0940.csv')
    raw_expr_df = pd.read_csv(figpath / f'/data/shared/alzgene26/data/figures/shap/real_expression_values_260508_0940.csv')

    alphabetical_genes = sorted(shap_df.columns.tolist())

    shap_df_sorted = shap_df[alphabetical_genes]
    raw_expr_sorted = raw_expr_df[alphabetical_genes]

    # Rebuild the object with alphabetical order
    shap_explanation_sorted = shap.Explanation(
        values=shap_df_sorted.values,
        base_values=base_value, # Ensure this is the scalar or array you calculated/saved
        data=raw_expr_sorted.values,
        feature_names=alphabetical_genes
    )

    # generate the plots
    print("Displaying Beeswarm Plot...")
    shap.plots.beeswarm(shap_explanation_sorted, show=False, max_display=11, sort=False)
    plt.savefig(figpath / f'beeswarm_{stage}_{date}.png', bbox_inches='tight')
    plt.close()

    for i in list(range(3)):
        print(f"Displaying Waterfall Plot for Patient {i}...")
        shap.plots.waterfall(shap_explanation_sorted[i], show=False, max_display=11, sort=False)
        plt.savefig(figpath / f'waterfall_{stage}_{date}_{i}.png', bbox_inches='tight')
        plt.close()

    print("Displaying Violin Plot...")
    shap.plots.violin(shap_explanation_sorted, show=False, max_display=11, sort=False)
    plt.savefig(figpath / f'violin_plot_{stage}_{date}.png', bbox_inches='tight')
    plt.close()