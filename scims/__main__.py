import argparse
import os
import re
import numpy as np
from scipy.stats import norm
from scipy.stats import gaussian_kde
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def calculate_Rt(idxstats, total_ref, total_map):
    rts = []
    for i in range(len(idxstats)):
        reference_length = idxstats[i][0]
        mapped_reads = idxstats[i][1]
        if total_map == 0 or total_ref == 0:
            rts.append(np.nan)  # Handle divide by zero by appending NaN
        else:
            rts.append(mapped_reads / (total_map / total_ref))
    return np.array(rts)

def read_metadata(metadata_path):
    metadata = pd.read_csv(metadata_path, sep='\t')
    return metadata

def find_sample_id_column(metadata):
    possible_column_names = [
        'sample-id', 'sampleid', 'sample id', 'id', 'featureid', 'feature id', 'feature-id', 'Run', 'SRA'
    ]
    for col in possible_column_names:
        if col in metadata.columns:
            return col
    raise ValueError("No valid sample ID column found in metadata file. Expected one of: " + ", ".join(possible_column_names))

def extract_sample_id(filename):
    match = re.search(r'([A-Za-z0-9\-\_]+)', filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Cannot extract sample ID from filename: {filename}")

def read_master_file(master_file_path):
    with open(master_file_path, 'r') as file:
        idxstats_files = file.readlines()
    idxstats_files = [line.strip() for line in idxstats_files if line.strip()]
    return idxstats_files

def simulate_data(mean, cov, num_samples=1000):
    return np.random.multivariate_normal(mean, cov, num_samples).T  # Shape: (2, N)

def compute_joint_posteriors(Rx, Ry, male_kde, female_kde, P_male, P_female):
    # Prepare the point for evaluation
    point = np.array([Rx, Ry])
    
    # Compute joint likelihoods using KDEs
    male_likelihood = male_kde(point)
    female_likelihood = female_kde(point)
    
    # Compute joint probabilities
    P_male_joint = male_likelihood * P_male
    P_female_joint = female_likelihood * P_female

    # Normalize to get posterior probabilities
    epsilon = 1e-10
    total_joint = P_male_joint + P_female_joint + epsilon
    P_male_posterior = P_male_joint / total_joint
    P_female_posterior = P_female_joint / total_joint

    return P_male_posterior[0], P_female_posterior[0]  # Extract scalar values

def determine_sex_with_joint_posteriors(P_male_posterior, P_female_posterior, threshold):
    # Classification
    if P_male_posterior >= threshold:
        return 'male'
    elif P_female_posterior >= threshold:
        return 'female'
    else:
        return 'uncertain'

def main():
    parser = argparse.ArgumentParser(description='Sex Assignment Script')
    parser.add_argument('--scaffolds', dest="scaffold_ids_file", required=True, type=str, help='File containing scaffold IDs of interest')
    parser.add_argument('--metadata', required=True, help="Path to the QIIME2 metadata file")
    parser.add_argument('--master_file', required=True, help="Path to the master file with idxstats paths")
    parser.add_argument('--homogametic_id', dest="x_id", required=True, type=str, help='Specify scaffold ID for homogametic chromosome (eg. XX = X, ZZ = Z)')
    parser.add_argument('--heterogametic_id', dest="y_id", required=True, type=str, help='Specify scaffold ID for heterogametic chromosome (eg. XY = Y, ZW = W)')
    parser.add_argument('--system', dest="system", required=True, type=str, choices=['XY', 'ZW'], help='Specify the sex determination system (XY or ZW)')
    parser.add_argument('--threshold', dest="threshold", type=float, default=0.95, help='Threshold for determining sex (default: 0.95)')
    parser.add_argument('--output', dest="output_file", required=True, type=str, help='Output file to save the results')
    parser.add_argument('--training_data', dest="training_data", type=str, default="training_data_ry.txt", help='Path to the training data file')
    args = parser.parse_args()

    results = []
    metadata = read_metadata(args.metadata)
    sample_id_col = find_sample_id_column(metadata)
    idxstats_files = read_master_file(args.master_file)

    # Load the training data
    if not os.path.exists(args.training_data):
        raise FileNotFoundError(f"Training data file not found: {args.training_data}")
    training_data = pd.read_csv(args.training_data, sep='\t')

    # Flip the sex labels if the system is ZW
    if args.system == 'ZW':
        training_data['actual_sex'] = training_data['actual_sex'].apply(lambda x: 'female' if x == 'male' else 'male')
        logging.info(f"Training data is adjusted for ZW system")

    male_data = training_data[training_data['actual_sex'] == 'male'][['Rx', 'Ry']].dropna().values.T
    female_data = training_data[training_data['actual_sex'] == 'female'][['Rx', 'Ry']].dropna().values.T

    # Joint KDE for males and females
    kde_male_joint = gaussian_kde(male_data)
    kde_female_joint = gaussian_kde(female_data)

    # Load the pre-trained model and scaler
    results = []

    with open(args.scaffold_ids_file, 'r') as file:
        scaffold_ids = file.read().splitlines()

    for idxstats_file in idxstats_files:
        sample_id = None  # Initialize to avoid UnboundLocalError
        try:
            sample_id = extract_sample_id(os.path.basename(idxstats_file))
            idxstats = pd.read_table(idxstats_file, header=None, index_col=0)
            idxstats = idxstats.loc[scaffold_ids]

            total_ref = idxstats.iloc[:, 0].sum() 
            total_map = idxstats.iloc[:, 1].sum() 

            #Rt_values = calculate_Rt(idxstats[[1, 2]].values, total_ref, total_map)

            if args.system == 'XY':
                x_id = args.x_id
                y_id = args.y_id
                x_index = idxstats.index.get_loc(x_id) if x_id in idxstats.index else None
                y_index = idxstats.index.get_loc(y_id) if y_id in idxstats.index else None

                coverage_X = idxstats.iloc[x_index, 1] / idxstats.iloc[x_index, 0]
                autosomal_coverages = [
                    idxstats.iloc[i, 1] / idxstats.iloc[i, 0] for i in range(len(idxstats))
                    if i != x_index and (y_index is None or i != y_index)
                ]
                Rx = coverage_X / np.mean(autosomal_coverages) if autosomal_coverages else np.nan
                z_value = np.round(norm.ppf(1 - ( 1 - args.threshold)/2), 3)
                #print(f"z_value: {z_value}")

                autosomal_variance = np.var(autosomal_coverages) if autosomal_coverages else 0
                autosomal_sample_size = len(autosomal_coverages)
                SE_Rx = np.sqrt(autosomal_variance) / np.sqrt(autosomal_sample_size) if autosomal_sample_size > 0 else np.nan
                #print(f"SE_Rx: {SE_Rx}, Variance: {autosomal_variance}, Sample Size: {autosomal_sample_size}")

                CI1_Rx = Rx - z_value * SE_Rx
                CI2_Rx = Rx + z_value * SE_Rx
                #print(f"CI1_Rx: {CI1_Rx}, CI2_Rx: {CI2_Rx}")

                x_count = idxstats.loc[x_id].iloc[1]
                y_count = idxstats.loc[y_id].iloc[1]
                tot_y = x_count + y_count
                if tot_y == 0:
                    Ry = np.nan
                    logging.warning(f"Total sex chromosome reads is 0 for sample {sample_id}.")
                else:   
                    Ry = (1.0 * y_count) / tot_y
                SE_y = np.sqrt((Ry * (1 - Ry)) / tot_y) if tot_y > 0 else np.nan
                CI1_y = Ry - z_value * SE_y
                CI2_y = Ry + z_value * SE_y

                # Priors
                P_male = 0.5
                P_female = 0.5

                # Fetch KDEs
                male_kde = kde_male_joint
                female_kde = kde_female_joint

                # Create multivariate normal distributions
                #male_dist = multivariate_normal(mean=male_stats['mean'], cov=male_stats['cov'])
                #female_dist = multivariate_normal(mean=female_stats['mean'], cov=female_stats['cov'])

                P_male_posterior, P_female_posterior = compute_joint_posteriors(Rx, Ry, male_kde, female_kde, P_male, P_female)

                inferred_sex = determine_sex_with_joint_posteriors(P_male_posterior, P_female_posterior, args.threshold)

                results.append({
                    'SCiMS sample ID': sample_id,
                    'SCiMS predicted sex': inferred_sex,
                    'Total reads mapped': total_map,
                    'Reads mapped to X': x_count,
                    'Reads mapped to Y': y_count,
                    'Rx': np.round(Rx, 4),
                    'Rx 95% CI': (float(np.round(CI1_Rx, 3)), float(np.round(CI2_Rx, 3))) if np.isfinite(CI1_Rx) and np.isfinite(CI2_Rx) else (np.nan, np.nan),
                    'Ry': np.round(Ry, 4),
                    'Ry 95% CI': (float(np.round(CI1_y, 3)), float(np.round(CI2_y, 3))) if np.isfinite(CI1_y) and np.isfinite(CI2_y) else (np.nan, np.nan),
                    'Posterior probability of being male ': float(np.round(P_male_posterior, 4)),
                    'Posterior probability of being female ': float(np.round(P_female_posterior, 4)),
                    'Status': 'Success'
                })
            
            else:
                z_id = args.x_id
                w_id = args.y_id
                z_index = idxstats.index.get_loc(z_id) if z_id in idxstats.index else None
                w_index = idxstats.index.get_loc(w_id) if w_id in idxstats.index else None

                coverage_Z = idxstats.iloc[z_index, 1] / idxstats.iloc[z_index, 0]
                autosomal_coverages = [
                    idxstats.iloc[i, 1] / idxstats.iloc[i, 0] for i in range(len(idxstats))
                    if i != z_index and (w_index is None or i != w_index)
                ]
                Rz = coverage_Z / np.mean(autosomal_coverages) if autosomal_coverages else np.nan
                z_value = np.round(norm.ppf(1 - ( 1 - args.threshold)/2), 3)
                #print(f"z_value: {z_value}")

                autosomal_variance = np.var(autosomal_coverages) if autosomal_coverages else 0
                autosomal_sample_size = len(autosomal_coverages)
                SE_Rz = np.sqrt(autosomal_variance) / np.sqrt(autosomal_sample_size) if autosomal_sample_size > 0 else np.nan
                #print(f"SE_Rz: {SE_Rz}, Variance: {autosomal_variance}, Sample Size: {autosomal_sample_size}")

                CI1_Rz = Rz - z_value * SE_Rz
                CI2_Rz = Rz + z_value * SE_Rz
                #print(f"CI1_Rz: {CI1_Rz}, CI2_Rz: {CI2_Rz}")

                z_count = idxstats.loc[z_id].iloc[1]
                w_count = idxstats.loc[w_id].iloc[1]
                tot_w = z_count + w_count
                if tot_w == 0:
                    Rw = np.nan
                    logging.warning(f"Total sex chromosome reads is 0 for sample {sample_id}.")
                else:   
                    Rw = (1.0 * w_count) / tot_w
                    SE_w = np.sqrt((Rw * (1 - Rw)) / tot_w) if tot_w > 0 else np.nan
                    CI1_w = Rw - z_value * SE_w
                    CI2_w = Rw + z_value * SE_w

                # Priors
                P_male = 0.5
                P_female = 0.5

                # Fetch KDEs
                male_kde = kde_male_joint
                female_kde = kde_female_joint

                P_male_posterior, P_female_posterior = compute_joint_posteriors(Rz, Rw, male_kde, female_kde, P_male, P_female)

                inferred_sex = determine_sex_with_joint_posteriors(P_male_posterior, P_female_posterior, args.threshold)
                
                results.append({
                    'SCiMS sample ID': sample_id,
                    'SCiMS predicted sex': inferred_sex,
                    'Total reads mapped': total_map,
                    'Reads mapped to Z': z_count,
                    'Reads mapped to W': w_count,
                    'Rz': np.round(Rz, 4),
                    'Rz 95% CI': (float(np.round(CI1_Rz, 3)), float(np.round(CI2_Rz, 3))) if np.isfinite(CI1_Rz) and np.isfinite(CI2_Rz) else (np.nan, np.nan),
                    'Rw': np.round(Rw, 4),
                    'Rw 95% CI': (float(np.round(CI1_w, 3)), float(np.round(CI2_w, 3))) if np.isfinite(CI1_w) and np.isfinite(CI2_w) else (np.nan, np.nan),
                    'Posterior probability of being male ': float(np.round(P_male_posterior, 4)),
                    'Posterior probability of being female ': float(np.round(P_female_posterior, 4)),
                    'Status': 'Success'
                })

        except Exception as e:
            logging.error(f"Error processing {idxstats_file}: {e}")
            results.append({'SCiMS sample ID': sample_id, 'Status': f"Failed: {e}"})
            
    results_df = pd.DataFrame(results)
    merged_df = pd.merge(metadata, results_df, left_on=sample_id_col, right_on='SCiMS sample ID', how='left')
    merged_df.to_csv(args.output_file, sep='\t', index=False)

    print(merged_df.head())  # Show the first few rows to inspect the contents

if __name__ == "__main__":
    main()
