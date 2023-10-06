#%%
import pandas as pd 
import os 
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
# %%
gtorgan_fpath = '/home/jhubadmin/Projects/ct_organ_segmentation/organ_gtlesion_organ_segreggated_intersection_3.csv'
gtorgan_df = pd.read_csv(gtorgan_fpath)
predorgan_fpath = '/home/jhubadmin/Projects/ct_organ_segmentation/organ_unetpredlesion_organ_segreggated_intersection_3.csv'
predorgan_df = pd.read_csv(predorgan_fpath)
validids = list(predorgan_df['ImageID'])
gtorgan_valid_df = gtorgan_df[gtorgan_df['ImageID'].isin(validids)]
gtorgan_valid_df.reset_index(drop=True, inplace=True)
# %%
def plot_sum_all_columns(df_gt, df_pred, ax=None):
    col_names_gt = df_gt.columns[1:]
    col_names_pred = df_pred.columns[1:]
    colsum_values_gt = []
    colsum_values_pred = []
    for i in range(len(col_names_gt)):
        colsum_gt = df_gt[col_names_gt[i]].sum()
        colsum_pred = df_pred[col_names_pred[i]].sum()
        colsum_values_gt.append(colsum_gt)
        colsum_values_pred.append(colsum_pred)
    locs = 2*np.arange(len(col_names_gt))
    ax.bar(locs-0.375, colsum_values_gt, 0.75, alpha=0.5, color='orange', label='GT')
    ax.bar(locs+0.375, colsum_values_pred, 0.75, alpha=0.5, color='green', label='Pred')
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    col_labels = [name[:-3] for name in col_names_gt]
    ax.set_xticks(locs, col_labels, rotation=90, fontsize=15)
    ax.set_title('Sum', fontsize=20)
    ax.legend(fontsize=20)
    return ax

def plot_mean_all_columns(df, ax=None):
    col_names = df.columns[1:]
    colmean_values = []
    for name in col_names:
        colmean = df[name].mean()
        colmean_values.append(colmean)
    ax.bar(col_names, colmean_values)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xticklabels(col_names, rotation=90, fontsize=20)
    ax.set_title('Mean', fontsize=20)
    return ax

def plot_numnonzero_all_columns(df_gt, df_pred, ax=None):
    col_names_gt = df_gt.columns[1:]
    col_names_pred = df_pred.columns[1:]
    non_zero_counts_gt = df_gt.applymap(lambda x: 1 if x != 0 else 0).sum()[1:]
    non_zero_counts_pred = df_pred.applymap(lambda x: 1 if x != 0 else 0).sum()[1:]

    locs = 2*np.arange(len(col_names_gt))
    ax.bar(locs-0.375, non_zero_counts_gt, 0.75, alpha=0.5, color='orange', label='GT')
    ax.bar(locs+0.375, non_zero_counts_pred, 0.75, alpha=0.5, color='green', label='Pred')
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    col_labels = [name[:-3] for name in col_names_gt]
    ax.set_xticks(locs, col_labels, rotation=90, fontsize=15)
    ax.set_title('Sum', fontsize=20)
    ax.legend(fontsize=20)
    ax.set_title('Number of non-zero intersections', fontsize=20)
    return ax
#%%
fig, ax = plt.subplots(1, 2, figsize=(15,8))
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1)
plot_sum_all_columns(gtorgan_valid_df,predorgan_df, ax=ax[0])
plot_numnonzero_all_columns(gtorgan_valid_df,predorgan_df, ax=ax[1])
# plot_numnonzero_all_columns(gtorgan_valid_df, ax=ax[0][1])
# # plot_sum_all_columns(predorgan_df, ax=ax[1][0])
# plot_numnonzero_all_columns(predorgan_df, ax=ax[1][1])
plt.show()
#%%
# %%
new_df = pd.concat([gtorgan_valid_df,predorgan_df.iloc[:, 1:]], axis=1)
column_labels = [name[:-3] for name in gtorgan_valid_df.columns[1:]]
column_new_order = list(np.array([[f"{label}_gt", f"{label}_pred"] for label in column_labels]).flatten())
# %%
new_df_new = new_df[['ImageID'] + column_new_order]
#%%
new_df_new.to_csv('valid_gtorgan_predorgan.csv', index=False)
# %%
