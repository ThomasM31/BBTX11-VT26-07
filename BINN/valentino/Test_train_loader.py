import pandas as pd
import custom_train_test_split
import BINN.valentino.train_loader as train_loader
from BINN.valentino.train_loader import read_adata, xy_datasplit

# First mask matrix file
df1 = pd.read_csv('~/PathwayData/MaskMatrixLayers/mg_200_mc_200_mhvg1000/oligo_exc3_exc2_vasc_immune_astro_inhi_opcs_exc1_layer_0_mask.csv',index_col=0)
#print(df.shape)


row_labels1 = df1.index # This is the genes
column_labels1 = df1.columns # This is the pathways
matrix1=df1.to_numpy() # This is the mask matrix

print(row_labels1[:])
print(column_labels1)
print(df1.shape,'matrix1')

# Second mask matrix file
df2 = pd.read_csv('~/PathwayData/MaskMatrixLayers/mg_200_mc_200_mhvg1000/oligo_exc3_exc2_vasc_immune_astro_inhi_opcs_exc1_layer_1_mask.csv',index_col=0)

row_labels2 = df2.index # This is the genes
column_labels2 = df2.columns # This is the pathways
matrix2=df2.to_numpy() # This is the mask matrix

print(row_labels2[:])
print(column_labels2)
print(df2.shape,'matrix2')

# Third mask matrix file
df3 = pd.read_csv('~/PathwayData/MaskMatrixLayers/mg_200_mc_200_mhvg1000/oligo_exc3_exc2_vasc_immune_astro_inhi_opcs_exc1_layer_2_mask.csv',index_col=0)

row_labels3 = df3.index # This is pathways
column_labels3 = df3.columns # This is pathways
matrix3=df3.to_numpy() # This is the mask matrix

print(row_labels3[:])
print(column_labels3)
print(df3.shape,'matrix3')

# Fourth mask matrix file
df4 = pd.read_csv('~/PathwayData/MaskMatrixLayers/mg_200_mc_200_mhvg1000/oligo_exc3_exc2_vasc_immune_astro_inhi_opcs_exc1_layer_3_mask.csv',index_col=0)

row_labels4 = df4.index # This is pathways
column_labels4 = df4.columns # This is pathways
matrix4=df4.to_numpy() # This is the mask matrix

print(row_labels4[:])
print(column_labels4)
print(df4.shape,'matrix4')


# Fifth mask matrix file
df5 = pd.read_csv('~/PathwayData/MaskMatrixLayers/mg_200_mc_200_mhvg1000/oligo_exc3_exc2_vasc_immune_astro_inhi_opcs_exc1_layer_4_mask.csv',index_col=0)

row_labels5 = df5.index # This is pathways
column_labels5 = df5.columns # This is pathways
matrix5=df5.to_numpy() # This is the mask matrix

print(row_labels5[:])
print(column_labels5)
print(df5.shape,'matrix5')




mask_list = [matrix1, matrix2, matrix3, matrix4, matrix5]
in_features = matrix1.shape[0]
layer_list = [matrix1.shape[1], matrix2.shape[1], matrix3.shape[1], matrix4.shape[1], matrix5.shape[1]]

print(in_features)
print(layer_list)

indices = [0]
print(indices)
filepath = '/data/shared/alzgene26/data/processed_data/completed/mg_200_mc_200_mhvg1000'
train_adata, test_adata, acollection = read_adata(indices=indices, filepath=filepath, train_size=0.8)

print(train_adata)


