library(data.table)
require(neuroim)

#################################
##### Load and reshape data #####
#################################

uncollapse_data <- function(csv_file_path, nlat = 90, nlon = 144){
    ## Uncollapse csv files from extreme_detection_procedure.py

    # Read the CSV file
    df <- fread(csv_file_path)
    matrix_2d <- as.matrix(df)
    cat("2D matrix dim:", dim(matrix_2d), "\n")

    # Reshape the 2D matrix back to the original 3D shape using list and unlist
    i <- 1
    timeseries_list <- list()
    while(i < nrow(matrix_2d)+1) {
        timeseries_list[[i]] <- matrix(matrix_2d[i,], nrow = nlat, ncol = nlon, byrow = TRUE)
        i <- i + 1
    }

    # Unlist list to produce array with all of the data in order
    matrix_3d <- array(unlist(timeseries_list), dim = c(nlat, nlon, length(timeseries_list)))
    cat("3D matrix dim:", dim(matrix_3d), "\n")
    
    return(matrix_3d)
}

# Get Megan data
megan_top_10 <- uncollapse_data("../../results/megan_top_10.csv")
#megan_bottom_10 <- uncollapse_data("../../results/megan_bottom_10.csv")

# Get OMI data
omi_top_10 <- uncollapse_data("../../results/omi_top_10.csv")
#omi_bottom_10 <- uncollapse_data("../../results/omi_bottom_10.csv")

# Get Model1 data
model1_top_10 <- uncollapse_data("../../results/model1_top_10.csv")
#model_bottom_10 <- uncollapse_data("../../results/model1_bottom_10.csv")

# Get Model2 data
model2_top_10 <- uncollapse_data("../../results/model2_top_10.csv")
#model_bottom_10 <- uncollapse_data("../../results/model1_bottom_10.csv")

##################################
#### 3D cluter identification ####
##################################


run_connComp3D <- function(dataset){
    # replicate your dataset but set 0 to F and 1 to T
    dataset_3D <- array(FALSE, dim(dataset))
    dataset_3D[dataset == 1] <- TRUE

    # run connComp3D
    result_3D_connComp <- connComp3D(dataset_3D)

    # get information on index and size from the resulting list
    result_3D_connComp_index <- result_3D_connComp[[1]]
    result_3D_connComp_size <- result_3D_connComp[[2]]

    cat("Index of biggest extreme:",  max(result_3D_connComp_index, na.rm = TRUE) , "\n")
    cat("Size of biggest extreme:",  max(result_3D_connComp_size, na.rm = TRUE) , "\n")

    return(list(result_3D_connComp_index, result_3D_connComp_size))
}

megan_top_ten_connComp = run_connComp3D(megan_top_10)
omi_top_ten_connComp = run_connComp3D(omi_top_10)
model1_top_ten_connComp = run_connComp3D(model1_top_10)
model2_top_ten_connComp = run_connComp3D(model2_top_10)

# megan_bottom_ten_connComp = run_connComp3D(megan_bottom_10)
# omi_bottom_ten_connComp = run_connComp3D(omi_bottom_10)
# model_bottom_ten_connComp = run_connComp3D(model_bottom_10)


######################
#### Save Results ####
######################
saveRDS(megan_top_ten_connComp, file="megan_top_ten_connComp.RData")
saveRDS(omi_top_ten_connComp, file="omi_top_ten_connComp.RData")
saveRDS(model1_top_ten_connComp, file="model1_top_ten_connComp.RData")
saveRDS(model2_top_ten_connComp, file="model2_top_ten_connComp.RData")