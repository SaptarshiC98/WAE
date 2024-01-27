# WAE
Codes for the Generalisation Bounds for Wasserstein Autoencoders, used in the paper, https://openreview.net/forum?id=WjRPZsfeBO&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2024%2FConference%2FAuthors%23your-submissions)

Within each of the folders, you will find a .sh file. You may need to make them executable by running "chmod +x filename.sh" in your terminal to make them executable. We recommend executing these .sh files on a GPU equipped with CUDA for optimal performance. 

1. First run the "data_generate.sh" in the "Data_generate" folder. It will regenerate the data called, "data_16.pt" and "data_16.pt" in the same folder.
2. Copy and paste the files, "data_2.pt" and "data_16.pt" to each of the other four folders (i.e. except "Data_generate")
3. In each of the folders, run the "run_models.sh" files to run the models on the grid as described in the paper. 
4. To regenerate the plots, you can run the code provided in the plot.ipynb file located within each respective folder. You may also run these ipynb files without completing steps 1,2 &3 as we have also provided our resulting tables as .pt files in each of the folders.
