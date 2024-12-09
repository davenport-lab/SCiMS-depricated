import unittest
from unittest.mock import patch, mock_open
import numpy as np
from scipy.stats import norm
import pandas as pd
from scims.scims import (
    calculate_Rt,
    read_metadata,
    find_sample_id_column,
    extract_sample_id,
    standardize_id,
    match_sample_ids,
    read_master_file,
    calculate_posterior,
    determine_sex_xy,
    determine_sex_zw,
    calculate_confidence_interval,
    logit_transform,
    probit_transform
)

class TestScims(unittest.TestCase):
    
    def test_calculate_Rt_normal_values(self):
        idxstats = np.array([[10, 100], [20, 200], [30, 300]])
        total_ref = 60
        total_map = 600
        expected = np.array([(100/600)/(10/60), (200/600)/(20/60), (300/600)/(30/60)])
        result = calculate_Rt(idxstats, total_ref, total_map)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_calculate_Rt_zero_total_map(self):
        idxstats = np.array([[10, 100], [20, 200]])
        total_ref = 30
        total_map = 0
        expected = np.array([np.nan, np.nan])
        result = calculate_Rt(idxstats, total_ref, total_map)
        np.testing.assert_array_equal(result, expected)
    
    def test_calculate_Rt_zero_idxstat(self):
        idxstats = np.array([[0, 100], [20, 200]])
        total_ref = 20
        total_map = 300
        expected = np.array([np.nan, (200/300)/(20/20)])
        result = calculate_Rt(idxstats, total_ref, total_map)
        np.testing.assert_array_equal(result, expected)
    
    @patch("builtins.open", new_callable=mock_open, read_data="sample_id\tvalue\nid1\t10\nid2\t20")
    def test_read_metadata(self, mock_file):
        metadata_path = "path/to/metadata.tsv"
        expected = pd.DataFrame({
            'sample_id': ['id1', 'id2'],
            'value': [10, 20]
        })
        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = expected
            result = read_metadata(metadata_path)
            mock_read_csv.assert_called_once_with(metadata_path, sep='\t')
            pd.testing.assert_frame_equal(result, expected)
    
    def test_find_sample_id_column_found(self):
        metadata = pd.DataFrame({
            'sample-id': [1, 2],
            'value': [10, 20]
        })
        expected = 'sample-id'
        result = find_sample_id_column(metadata)
        self.assertEqual(result, expected)
    
    def test_find_sample_id_column_not_found(self):
        metadata = pd.DataFrame({
            'id_column': [1, 2],
            'value': [10, 20]
        })
        with self.assertRaises(ValueError):
            find_sample_id_column(metadata)
    
    def test_extract_sample_id_found(self):
        filename = "sample_id123_data.txt"
        known_sample_ids = ["sample_id123", "sample_id456"]
        expected = "sample_id123"
        result = extract_sample_id(filename, known_sample_ids)
        self.assertEqual(result, expected)
    
    def test_extract_sample_id_not_found(self):
        filename = "unknown_sample_data.txt"
        known_sample_ids = ["sample1", "sample2"]
        with self.assertRaises(ValueError):
            extract_sample_id(filename, known_sample_ids)
    
    def test_standardize_id(self):
        sample_id = " Sample-ID_123 "
        expected = "SampleID123"
        result = standardize_id(sample_id)
        self.assertEqual(result, expected)
    
    def test_match_sample_ids_direct_match(self):
        metadata = pd.DataFrame({#
            'sample-id': ['Sample1', 'Sample2', 'Sample3'],#
            'metadata_value': [100, 200, 300]#
        })
        results_df = pd.DataFrame({
            'SCiMS sample ID': ['Sample1', 'Sample2', 'Sample3'],
            'Total reads mapped': [1000, 2000, 3000]
        })
        expected = pd.DataFrame({
            'sample-id': ['Sample1', 'Sample2', 'Sample3'],
            'metadata_value': [100, 200, 300],
            'SCiMS sample ID': ['Sample1', 'Sample2', 'Sample3'],
            'Total reads mapped': [1000, 2000, 3000]
        })
        result = match_sample_ids(metadata, results_df, 'sample-id')
        pd.testing.assert_frame_equal(result, expected)
    
    def test_match_sample_ids_standardized_match(self):
        metadata = pd.DataFrame({
            'sample-id': ['Sample-1', 'Sample_2', 'Sample_3'],
            'metadata_value': [100, 200, 300]
        })
        results_df = pd.DataFrame({
            'SCiMS sample ID': ['Sample1', 'Sample2', 'Sample3'],
            'Total reads mapped': [1000, 2000, 3000]
        })
        expected = pd.DataFrame({
            'sample-id': ['Sample1', 'Sample2', 'Sample3'],
            'metadata_value': [100, 200, 300],
            'SCiMS sample ID': ['Sample1', 'Sample2', 'Sample3'],
            'Total reads mapped': [1000, 2000, 3000]
        })
        result = match_sample_ids(metadata, results_df, 'sample-id')
        pd.testing.assert_frame_equal(result, expected)
    
    @patch("builtins.open", new_callable=mock_open, read_data="path/to/idxstats1\npath/to/idxstats2")
    def test_read_master_file(self, mock_file):
        master_file_path = "path/to/master_file.txt"
        expected = ["path/to/idxstats1", "path/to/idxstats2"]
        result = read_master_file(master_file_path)
        self.assertEqual(result, expected)
    
    def test_calculate_posterior(self):
        prior_male = 0.6
        prior_female = 0.4
        likelihood_male = 0.7
        likelihood_female = 0.3
        posterior_male, posterior_female = calculate_posterior(prior_male, prior_female, likelihood_male, likelihood_female)
        expected_male = (0.6 * 0.7) / (0.6 * 0.7 + 0.4 * 0.3)
        expected_female = 1 - expected_male
        self.assertAlmostEqual(posterior_male, expected_male)
        self.assertAlmostEqual(posterior_female, expected_female)
    
    def test_determine_sex_xy_male(self):
        posterior_male = 0.96
        posterior_female = 0.04
        threshold = 0.95
        result = determine_sex_xy(posterior_male, posterior_female, threshold)
        self.assertEqual(result, 'male')
    
    def test_determine_sex_xy_female(self):
        posterior_male = 0.04
        posterior_female = 0.96
        threshold = 0.95
        result = determine_sex_xy(posterior_male, posterior_female, threshold)
        self.assertEqual(result, 'female')
    
    def test_determine_sex_xy_uncertain(self):
        posterior_male = 0.94
        posterior_female = 0.06
        threshold = 0.95
        result = determine_sex_xy(posterior_male, posterior_female, threshold)
        self.assertEqual(result, 'uncertain')
    
    def test_determine_sex_zw_female(self):
        posterior_female = 0.96
        posterior_male = 0.04
        threshold = 0.95
        result = determine_sex_zw(posterior_female, posterior_male, threshold)
        self.assertEqual(result, 'female')
    
    def test_determine_sex_zw_male(self):
        posterior_female = 0.04
        posterior_male = 0.96
        threshold = 0.95
        result = determine_sex_zw(posterior_female, posterior_male, threshold)
        self.assertEqual(result, 'male')
    
    def test_determine_sex_zw_uncertain(self):
        posterior_female = 0.94
        posterior_male = 0.06
        threshold = 0.95
        result = determine_sex_zw(posterior_female, posterior_male, threshold)
        self.assertEqual(result, 'uncertain')
    
    def test_calculate_confidence_interval(self):
        values = np.array([1, 2, 3, 4, 5])
        mean_val = np.mean(values)
        conf_interval = 1.96 * (np.std(values) / np.sqrt(len(values)))
        expected = (mean_val, mean_val - conf_interval, mean_val + conf_interval)
        result = calculate_confidence_interval(values)
        self.assertAlmostEqual(result[0], expected[0])
        self.assertAlmostEqual(result[1], expected[1])
        self.assertAlmostEqual(result[2], expected[2])
    
    def test_logit_transform(self):
        R = np.array([0.2, 0.5, 0.8])
        expected = np.log(R / (1 - R))
        result = logit_transform(R)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_logit_transform_clipping(self):
        R = np.array([0, 1])
        expected = np.log(np.clip(R, 1e-9, 1 - 1e-9) / (1 - np.clip(R, 1e-9, 1 - 1e-9)))
        result = logit_transform(R)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_probit_transform(self):
        R = np.array([0.2, 0.5, 0.8])
        expected = norm.ppf(np.clip(R, 1e-9, 1 - 1e-9))
        result = probit_transform(R)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_probit_transform_clipping(self):
        R = np.array([0, 1])
        expected = norm.ppf(np.clip(R, 1e-9, 1 - 1e-9))
        result = probit_transform(R)
        np.testing.assert_array_almost_equal(result, expected)

if __name__ == '__main__':
    unittest.main()