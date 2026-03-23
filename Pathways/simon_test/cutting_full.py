import os
import csv 
import scanpy as sc
import anndata




input_path = "/data/shared/alzgene26/PathwaysLA/binn_connectivity.csv"

output_path ="/data/shared/alzgene26/PathwaysLA/binn_cutted.csv"

temp_path = "/data/shared/alzgene26/PathwaysLA/binn_temp.csv"

#-vascular, -immune

file = "/data/shared/alzgene26"

#???
paths = {
    "Astrocytes" : "/data/shared/alzgene26/data/conv_data/",
    "Excitatory_neurons_set1" :  "/data/shared/alzgene26/data/conv_data/Excitatory_neurons_set1.h5ad",
    "Excitatory_neurons_set2" : "/data/shared/alzgene26/data/conv_data/Excitatory_neurons_set2.h5ad",
    "Excitatory_neurons_set3" : "/data/shared/alzgene26/data/conv_data/Excitatory_neurons_set3.h5ad",
    "Inhibitory_neurons": "/data/shared/alzgene26/data/conv_data/Inhibitory_neurons.h5ad",
    "Oligodendrocytes": "/data/shared/alzgene26/data/conv_data/Oligodendrocytes.h5ad",
    "OPCs": "/data/shared/alzgene26/data/conv_data/OPCs.h5ad"
}

with open(os.path.join(file,"PathwaysLA/binn_connectivity.csv") ,"r", encoding="utf-8") as f:
    reader = csv.reader(f)      # comma is the default delimiter
    data = list(reader)         # convert to a list of rows
    
print(data[1][0])


file_temp = paths["OPCs"]

#Namnen på generna (här får man ändra som fan sen)   
if os.path.exists(file_temp):
    print("file found, loading...")
    big = sc.read_h5ad(file_temp, backed = "r")
else:
    print("file not found")
    
minitest = big.var_names[0:100]

#laddan den råa datan med alla pathways 
with open(input_path,"r", newline="", encoding="utf-8") as g:
    reader = csv.reader(g)
    data = list(reader)
 
#En första fkn som tar resettar en .csv fil och lägger in relevanta paths som den hittar  
def step1():
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
    
        for i in minitest:
            for j in range(len(data)):
                trow = data[j]
                if trow[0] == i:
                    writer.writerow(trow)
    
    #Skapar en temporär fil med pathways             
    with open(output_path, "r", encoding="utf-8") as src:
        with open(temp_path, "w", encoding="utf-8") as dst:
            dst.write(src.read())
        
    

#fkn 2 som hittar alla pathways från den temporära filen och overwritar dem och lägger sedan till dem i output-filen
#tanken är att man ska iterera över denna
def step2():
   with open(temp_path, "r", newline="", encoding="utf-8") as f:
      nodes = list(csv.reader(f))
      
   messi = [i[1] for i in nodes]
   
   counter = 0
   
   with open(temp_path, "w", newline="", encoding="utf-8") as g:
      writer = csv.writer(g)
      for i in messi:
         for j in range(len(data)):
            trow = data[j]
            if trow[0] == i:
               writer.writerow(trow)
               counter += 1
               
   with open(temp_path, "r", newline="", encoding="utf-8") as temp_file:
      with open(output_path, "a", newline="", encoding="utf-8") as h:
         reader = csv.reader(temp_file)
         writer = csv.writer(h)
         for row in reader:
            writer.writerow(row)
               
   return counter

step1()

cnt = 1
while cnt > 0:
    cnt = step2()
    print(cnt)
    
    

        