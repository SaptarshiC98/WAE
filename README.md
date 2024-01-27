# WAE
Codes for the Generalisation Bounds for Wasserstein Autoencoders

Within each of the folders, you will find a .sh file. You may need to make them executable by running "chmod +x filename.sh" in your terminal to make them executable. We recommend executing these .sh files on a GPU equipped with CUDA for optimal performance. 

1. For space economy, we have only provided "data_2.pt", i.e. the dataset corresponding to d_int = 2. Executing the "data_generate.sh" in the "Data_generate" folder will regenerate the data called, "data_16.pt".
2. Copy and paste the files, "data_2.pt" and your generated data_16.pt to each of the other four folders (i.e. except "Data_generate")
3. In each of the folders, run the "run_models.sh" files to run the models on the grid as described in the paper. 
4. To regenerate the plots, you can run the code provided in the plot.ipynb file located within each respective folder. You may also run these ipynb files without completing steps 1,2 &3 as we have also provided our resulting tables as .pt files in each of the folders.
