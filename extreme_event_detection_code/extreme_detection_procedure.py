##############################################
#### Import data to run extreme detection ####
##############################################

import netCDF4
import numpy as np
from matplotlib import pyplot as plt


def correct_OMI_data(input_np):
    """
    New OMI data is flipped in both the lon and lat dimentions, this function fixes that
    """
    
    halfway = np.shape(input_np)[3]//2
    first_half = input_np[:,:,:,:halfway]
    second_half = input_np[:,:,:,halfway:]
    long_corrected = np.concatenate((second_half,first_half), axis=3)
    fixed = np.flip(long_corrected, axis=2)

    fixed[fixed==0] = np.nan
    fixed_reshaped = fixed.reshape((13*12, 90, 144))
    return fixed_reshaped

# Specify the path to the data 
INPUT_FILE_OMI = "../../raw_data/bias_corrected_OMI_2005_2017/Isop_Top-Down_OMI_2005-2017.nc"

# Extract the data
ds = netCDF4.Dataset(INPUT_FILE_OMI)
ds.set_auto_mask(True)

# Data is in the shape (year, month, lat, lon)
top_down_raw = ds["Top-Down_Isoprene_Flux"][:]
megan_raw = ds["A_priori_Isoprene_Flux"][:] 

ds.close()

# Correct the data and slect time period to match the model
top_down = correct_OMI_data(top_down_raw)[:144,:,:]
megan = correct_OMI_data(megan_raw)[:144,:,:]

# Plot the data
im = plt.imshow(top_down[0,:, :], origin="lower", extent=[-180, 180, -90, 90])
plt.colorbar(im, fraction=0.022, pad=0.03)
plt.title("isoprene_flux")
plt.show()

# Load model data
INPUT_FILE_MODEL = "../../results/ESM_model_results_2005-2016.nc"

# Extract the data
ds = netCDF4.Dataset(INPUT_FILE_MODEL)
ds.set_auto_mask(True)

J = ds["J"][:]
Isps = ds["IspS"][:]
Jw = ds["Jw"][:]
JmJw = J-Jw

model1_raw = Isps *J
model2_raw = J*Isps + J*Isps*JmJw

ds.close()

# Regrid to fit the OMI data
model1_reshaped = model1_raw.reshape(144, 90, 4, 144, 5)
model1 = model1_reshaped.mean(axis=(2, 4))

model2_reshaped = model2_raw.reshape(144, 90, 4, 144, 5)
model2 = model2_reshaped.mean(axis=(2, 4))

# Mask the data by OMI
model1[np.isnan(top_down)] = np.nan
model2[np.isnan(top_down)] = np.nan

im = plt.imshow(model[0,:, :], origin="lower", extent=[-180, 180, -90, 90])
plt.colorbar(im, fraction=0.022, pad=0.03)
plt.title("isoprene_flux")
plt.show()


##############################################
###### Extreme Event Detection Protocol ######
##############################################

# This code follows the methodology described in (Zscheischler et al., 2013)
# Detection and attribution of large spatiotemporal extreme events in Earth observation data
# https://www.sciencedirect.com/science/article/abs/pii/S1574954113000253?via%3Dihub 

import numpy as np
import statsmodels.api as sm
from scipy.stats import norm


# Define functions for each step of the extreme event detection protocol

def detrend_emission(emission_data):
    """
    1. DETREND EMISSIONS

    Function performs a residual calculation to remove time trends
    Calculates the regression residuals (run over time) for each gridcell

    Args:
    emission_data (np array): initial data where the 0th dimention is time and 1 and 2 dimentions are spatial dimentions

    Returns:
    regression_residuals (np array): regression residual for each gridcell for every time point

    """

    # Make a time series from the length of the data
    time_series = np.arange(emission_data.shape[0])

    # Make an empty matrix that will be filled by the loop with the residuals from the regression
    regression_residuals = np.full(emission_data.shape, np.nan)

    # Calculate residuals from a regression (removing trends)
    for i in range(emission_data.shape[1]):  # loop through each of the gridcells
        for j in range(emission_data.shape[2]):
            if np.sum(~np.isnan(emission_data[:,i, j])) > 2:  # if more than 2 datapoints are NOT NA in gridcell (no residuals with less than 2 datapoints)
                y = emission_data[:,i, j]
                mask = ~np.isnan(y) # removing nans
                X = sm.add_constant(time_series[mask]) #removing nans
                model = sm.OLS(y[mask], X).fit()
                regression_residuals[mask, i, j] = model.resid  # save the residuals for each time point in each gridcell
    return (regression_residuals)


# TEST
# model_residuals = detrend_emission(model)

# model_residuals.shape
# model.shape

# im = plt.imshow(model_residuals[100,:, :], origin="lower", extent=[-180, 180, -90, 90])
# plt.colorbar(im, fraction=0.022, pad=0.03)
# plt.title("model_residuals")
# plt.show()


def remove_MAC(regression_residuals):
    """
    2. REMOVE THE MEAN ANNUAL CYCLE (MAC) VARIATION
    Isolate of the sesonal variation in isoprene emissions and remove from the data

    Args:
    regression_residuals (np array): emission regresiion residuals caluculated using the detrend_emission function

    Returns:
    MAC_resid_anomaly (np array): the anomoly regression residual per month 
    (i.e May 2007 would return the difference between the avearge regression residual in May for a the particular gridcell 
    and the regression residual for May 2007)

    """
    
    # Make an empty array to fill with the average residual per month for each gridcell
    mean_monthly_residuals = np.full((12, regression_residuals.shape[1], regression_residuals.shape[2]), np.nan)

    # Calculate the average residual per month
    for i in range(regression_residuals.shape[1]):  # loop through each of the gridcells
        for j in range(regression_residuals.shape[2]):
            if not np.all(np.isnan(regression_residuals[:, i, j])):  # if at least one datapoint is not NA (avoids NAN in mean calculation)
                for m in range(12):
                    mean_monthly_residuals[m, i, j] = np.nanmean(regression_residuals[np.arange(m, regression_residuals.shape[0] , step = 12), i, j])  # get the mean residual for each month in each gridcell

    # Calculate the anomaly residual for each timepoint with respect to the average residual anomaly for the month
    number_of_years = regression_residuals.shape[0] // 12
    MAC_resid_anomaly = regression_residuals - np.tile(mean_monthly_residuals, (number_of_years, 1, 1))

    return(MAC_resid_anomaly)

# TEST
# model_residuals = detrend_emission(model)
# model_MAC_residuals = remove_MAC(model_residuals)

# model_residuals.shape
# model_MAC_residuals.shape
# model.shape

# im = plt.imshow(model_MAC_residuals[100,:, :], origin="lower", extent=[-180, 180, -90, 90])
# plt.colorbar(im, fraction=0.022, pad=0.03)
# plt.title("model_MAC_residuals")
# plt.show()


def z_score_standardisation(MAC_resid_anomaly):
    """
    3. Z- SCORE STANDARDISATION
    Standardise the MAC_resid_anomoly by the zscore of the gridcell (over the whole time period)

    Args:
    MAC_resid_anomaly (np array): the anomoly regression residual per month as calculated by the remove_MAC function

    Returns:
    standardised_MAC_resid_anomaly (np array): z-score standardised MAC_resid_anomoy 
    """

    # Create empty arrays to fill with means and standard deviation for each gridcell
    MAC_resid_mean = np.full((MAC_resid_anomaly.shape[1], MAC_resid_anomaly.shape[2]), np.nan)
    MAC_resid_sd = np.full((MAC_resid_anomaly.shape[1], MAC_resid_anomaly.shape[2]), np.nan)

    # Create an empty array to store the standardised results
    standardised_MAC_resid_anomaly = np.full(MAC_resid_anomaly.shape, np.nan)

    for i in range(MAC_resid_anomaly.shape[1]):
        for j in range(MAC_resid_anomaly.shape[2]):
            if not np.all(np.isnan(MAC_resid_anomaly[:,i, j])):
                # Maximum-likelihood based sd and mean estimation
                data_nan_mask = MAC_resid_anomaly[:,i, j][~np.isnan(MAC_resid_anomaly[:, i, j])]
                MAC_resid_mean[i, j], MAC_resid_sd[i, j] = norm.fit(data_nan_mask, method="MLE") 
                
                # Standardize the MAC_resid_anomaly by z-score standardization
                if MAC_resid_sd[i, j] > 0:  # i.e. don't continue if SD was calculated on only 1 point
                    # Perform z-score standardization
                    standardised_MAC_resid_anomaly[:,i, j] = (MAC_resid_anomaly[:,i, j] - MAC_resid_mean[i, j]) / MAC_resid_sd[i, j]
    return(standardised_MAC_resid_anomaly)
    
 
# TEST
# model_residuals = detrend_emission(model)
# model_MAC_residuals = remove_MAC(model_residuals)
# model_z_stand_MAC_residuals = z_score_standardisation(model_MAC_residuals)

# model.shape
# model_residuals.shape
# model_MAC_residuals.shape
# model_z_stand_MAC_residuals.shape

# im = plt.imshow(model_z_stand_MAC_residuals[100,:, :], origin="lower", extent=[-180, 180, -90, 90])
# plt.colorbar(im, fraction=0.022, pad=0.03)
# plt.title("model_z_stand_MAC_residuals")
# plt.show()

def get_top_ten_percent_anomalies(standardised_MAC_resid_anomaly):
    """
    4. IDENTIFICATION OF TOP/BOTTOM 10% OF ANOMOLIES
    Identify where and when the top 10% of anomolies occured in the dataset

    Args:
    standardised_MAC_resid_anomaly (np array): z-score standardised MAC_resid_anomoy as calculated by the z_score_standardisation function

    Returns:
    top_ten_hit (np array): contains 1 vs 0 data where 1 is locations/times where the anomoly is in the top 10% of anomolies globally over the time period 
    sum_hit_top_10 (np array): total number of times that gridcell had a top 10% anomoly across the time series

    """

    # Find boundary for top and bottom 10
    masked_standardised_MAC_resid_anomaly = standardised_MAC_resid_anomaly[~np.isnan(standardised_MAC_resid_anomaly)]
    percentile_bounds = np.percentile(masked_standardised_MAC_resid_anomaly, [10, 90])
    
    # Record points where the standardised_MAC_resid_anomaly is in the top 10% of its global value
    top_ten_hit = np.copy(standardised_MAC_resid_anomaly)
    top_ten_hit[~np.isnan(standardised_MAC_resid_anomaly)] = 0 #set everything to 0 that isnt NAN
    top_ten_hit[standardised_MAC_resid_anomaly >= percentile_bounds[1]] = 1 #set everything above or at 90% mark to 1

    # Find how many top 10 hits there is in a gridcell over the time period
    sum_hit_top_10 = np.nansum(top_ten_hit, axis=0)

    return(top_ten_hit, sum_hit_top_10)
    
# TEST
# megan_residuals = detrend_emission(megan)
# megan_MAC_residuals = remove_MAC(megan_residuals)
# megan_z_stand_MAC_residuals = z_score_standardisation(megan_MAC_residuals)
# megan_top10per_anomaly, sum_megan_top10per_anomaly = get_top_ten_percent_anomalies(megan_z_stand_MAC_residuals)

# megan.shape
# megan_residuals.shape
# megan_MAC_residuals.shape
# megan_z_stand_MAC_residuals.shape
# megan_top10per_anomaly.shape
# sum_megan_top10per_anomaly.shape

# im = plt.imshow(megan_top10per_anomaly[100,:, :], origin="lower", extent=[-180, 180, -90, 90])
# plt.colorbar(im, fraction=0.022, pad=0.03)
# plt.title("megan_top10per_anomaly")
# plt.show()

# im = plt.imshow(sum_megan_top10per_anomaly, origin="lower", extent=[-180, 180, -90, 90])
# plt.colorbar(im, fraction=0.022, pad=0.03)
# plt.title("sum_megan_top10per_anomaly")
# plt.show()

def get_bottom_ten_percent_anomalies(standardised_MAC_resid_anomaly):
    """
    4. IDENTIFICATION OF TOP/BOTTOM 10% OF ANOMOLIES
    Identify where and when the bottom 10% of anomolies occured in the dataset

    Args:
    standardised_MAC_resid_anomaly (np array): z-score standardised MAC_resid_anomoy as calculated by the z_score_standardisation function

    Returns:
    bottom_ten_hit (np array): contains 1 vs 0 data where 1 is locations/times where the anomoly is in the bottom 10% of anomolies globally over the time period 
    sum_hit_bottom_10 (np array): total number of times that gridcell had a bottom 10% anomoly across the time series

    """

    # Find boundary for top and bottom 10
    masked_standardised_MAC_resid_anomaly = standardised_MAC_resid_anomaly[~np.isnan(standardised_MAC_resid_anomaly)]
    percentile_bounds = np.percentile(masked_standardised_MAC_resid_anomaly, [10, 90])
    
    # Record points where the standardised_MAC_resid_anomaly is in the bottom 10% of its global value
    bottom_ten_hit = np.copy(standardised_MAC_resid_anomaly)
    bottom_ten_hit[~np.isnan(standardised_MAC_resid_anomaly)] = 0 #set everything to 0 that isnt NAN
    bottom_ten_hit[standardised_MAC_resid_anomaly <= percentile_bounds[0]] = 1 #set everything below or at 10% mark to 1

    # Find how many bottom 10 hits there is in a gridcell over the time period
    sum_hit_bottom_10 = np.nansum(bottom_ten_hit, axis=0)

    return(bottom_ten_hit, sum_hit_bottom_10)
    
# TEST
# megan_residuals = detrend_emission(megan)
# megan_MAC_residuals = remove_MAC(megan_residuals)
# megan_z_stand_MAC_residuals = z_score_standardisation(megan_MAC_residuals)
# megan_bottom10per_anomaly, sum_megan_bottom10per_anomaly = get_bottom_ten_percent_anomalies(megan_z_stand_MAC_residuals)

# megan.shape
# megan_residuals.shape
# megan_MAC_residuals.shape
# megan_z_stand_MAC_residuals.shape
# megan_bottom10per_anomaly.shape
# sum_megan_bottom10per_anomaly.shape

# im = plt.imshow(megan_bottom10per_anomaly[100,:, :], origin="lower", extent=[-180, 180, -90, 90])
# plt.colorbar(im, fraction=0.022, pad=0.03)
# plt.title("megan_bottom10per_anomaly")
# plt.show()

# im = plt.imshow(sum_megan_top10per_anomaly, origin="lower", extent=[-180, 180, -90, 90])
# plt.colorbar(im, fraction=0.022, pad=0.03)
# plt.title("sum_megan_bottom10per_anomaly")
# plt.show()


###################################################
#####      Visialise locations where the      #####
##### most top/botom 10% anomoly events occur #####
###################################################

# from matplotlib.colors import ListedColormap


# # Visualize 10%
# zlimit = [0, 30]
# colours = plt.cm.YlOrRd(np.linspace(0, 1, 9))
# colours[0] = [0.75, 0.75, 0.75, 1]  # grey color
# breaks = np.linspace(zlimit[0], zlimit[1], len(colours) + 1)
# hit10[hit10 > zlimit[1]] = zlimit[1]

# fig, ax = plt.subplots()
# cmap = ListedColormap(colours)
# c = ax.imshow(hit10, cmap=cmap, vmin=zlimit[0], vmax=zlimit[1], origin="lower", extent=[-180, 180, -90, 90])
# plt.colorbar(c, ax=ax, boundaries=breaks, ticks=breaks)
# plt.show()


# # Visualize 90%
# colours = plt.cm.Greens(np.linspace(0, 1, 9))
# colours[0] = [0.75, 0.75, 0.75, 1]  # grey color
# breaks = np.linspace(zlimit[0], zlimit[1], len(colours) + 1)
# hit90[hit90 > zlimit[1]] = zlimit[1]

# fig, ax = plt.subplots()
# cmap = ListedColormap(colours)
# c = ax.imshow(hit90, cmap=cmap, vmin=zlimit[0], vmax=zlimit[1], origin="lower", extent=[-180, 180, -90, 90])
# plt.colorbar(c, ax=ax, boundaries=breaks, ticks=breaks)
# plt.show()



############################################################
#####     Save top/bottom_ten_hit for import into R    #####
##### Use neuroim package in R to identify 3D clusters #####
############################################################
# Detection of spatiotemporal extreme events is done with Neuroim in R but a more transparent python package should be considered 
def reshape_and_save_for_R(xx_ten_hit, path):

    matrix_3d = xx_ten_hit
    original_shape = matrix_3d.shape

    # Collapse along the first dimension
    matrix_2d = matrix_3d.reshape((matrix_3d.shape[0], -1))
    new_shape = matrix_2d.shape

    # Save colapsed matrix
    np.savetxt(path, matrix_2d, delimiter=',')
    
    # Echo information
    print("Original shape of data:  ", original_shape)
    print("New shape:  ", new_shape)
    print(f"2D matrix saved to {path}")


# Run prosedure on MEGAN data
megan_residuals = detrend_emission(megan)
megan_MAC_residuals = remove_MAC(megan_residuals)
megan_z_stand_MAC_residuals = z_score_standardisation(megan_MAC_residuals)
megan_top10per_anomaly, sum_megan_top10per_anomaly = get_top_ten_percent_anomalies(megan_z_stand_MAC_residuals)
#megan_bottom10per_anomaly, sum_megan_bottom10per_anomaly = get_bottom_ten_percent_anomalies(megan_z_stand_MAC_residuals)
reshape_and_save_for_R(megan_top10per_anomaly, '../../results/megan_top_10.csv')
#reshape_and_save_for_R(megan_bottom10per_anomaly, '../../results/megan_bottom_10.csv')

# Run prosedure on OMI top_down data
omi_residuals = detrend_emission(top_down)
omi_MAC_residuals = remove_MAC(omi_residuals)
omi_z_stand_MAC_residuals = z_score_standardisation(omi_MAC_residuals)
omi_top10per_anomaly, sum_omi_top10per_anomaly = get_top_ten_percent_anomalies(omi_z_stand_MAC_residuals)
#omi_bottom10per_anomaly, sum_omi_bottom10per_anomaly = get_bottom_ten_percent_anomalies(omi_z_stand_MAC_residuals)
reshape_and_save_for_R(omi_top10per_anomaly, '../../results/omi_top_10.csv')
#reshape_and_save_for_R(omi_bottom10per_anomaly, '../../results/omi_bottom_10.csv')


# Run prosedure on Model1 data
model1_residuals = detrend_emission(model1)
model1_MAC_residuals = remove_MAC(model1_residuals)
model1_z_stand_MAC_residuals = z_score_standardisation(model1_MAC_residuals)
model1_top10per_anomaly, sum_model1_top10per_anomaly = get_top_ten_percent_anomalies(model1_z_stand_MAC_residuals)
#model1_bottom10per_anomaly, sum_model1_bottom10per_anomaly = get_bottom_ten_percent_anomalies(model1_z_stand_MAC_residuals)
reshape_and_save_for_R(model1_top10per_anomaly, '../../results/model1_top_10.csv')
#reshape_and_save_for_R(model1_bottom10per_anomaly, '../../results/model1_bottom_10.csv')

# Run prosedure on Model2 data
model2_residuals = detrend_emission(model2)
model2_MAC_residuals = remove_MAC(model2_residuals)
model2_z_stand_MAC_residuals = z_score_standardisation(model2_MAC_residuals)
model2_top10per_anomaly, sum_model2_top10per_anomaly = get_top_ten_percent_anomalies(model2_z_stand_MAC_residuals)
#model2_bottom10per_anomaly, sum_model2_bottom10per_anomaly = get_bottom_ten_percent_anomalies(model2_z_stand_MAC_residuals)
reshape_and_save_for_R(model2_top10per_anomaly, '../../results/model2_top_10.csv')
#reshape_and_save_for_R(model2_bottom10per_anomaly, '../../results/model2_bottom_10.csv')
