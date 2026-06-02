import pandas as pd
import numpy as np
import os

def get_data_dir(data_dir=None):
    if data_dir is None:
        # Default to ../data relative to this file
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    return data_dir

class DataLoader:
    def __init__(self, data_dir=None):
        self.data_dir = get_data_dir(data_dir)
        self.load_all()

    def load_all(self):
        # 1. Macro Data
        macro_df = pd.read_csv(os.path.join(self.data_dir, 'vietnam_macro_2020_2025.csv'))
        self.macro_years = macro_df['year'].values
        self.macro_Y = macro_df['GDP_trillion_VND'].values
        
        # Note: K, L, D, AI, H are not explicitly in the macro CSV with matching unit dimensions,
        # so we default to the legacy values to preserve mathematical integrity of Bai 1
        self.macro_K = np.array([16500, 17800, 19600, 21300, 23500, 25900], dtype=float)
        self.macro_L = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4])
        self.macro_D = np.array([12.0, 12.7, 14.3, 16.5, 18.3, 19.5])
        self.macro_AI = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1])
        self.macro_H = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2])
        
        # 2. Sectors Data
        sectors_df = pd.read_csv(os.path.join(self.data_dir, 'vietnam_sectors_2024.csv'))
        self.sectors_names_vi = sectors_df['sector_name_vi'].values
        self.sectors_names_en = sectors_df['sector_name_en'].values
        self.sectors_employment = sectors_df['labor_million'].values
        self.sectors_automation_risk = sectors_df['automation_risk_pct'].values / 100.0  # converted to decimal

        # Features for Bai 3 (Priority Index)
        # Expected order: growth, productivity, spillover, export, employment, ai_readiness, automation_risk
        # We will map productivity to fdi_attraction for now, as labor_productivity is missing in sector CSV
        self.X_sectors = np.column_stack([
            sectors_df['growth_rate_2024_pct'].values,
            sectors_df['fdi_attraction_billion_USD'].values, 
            sectors_df['spillover_coef_0_1'].values,
            sectors_df['export_billion_USD'].values,
            sectors_df['labor_million'].values,
            sectors_df['ai_readiness_0_100'].values,
            sectors_df['automation_risk_pct'].values
        ])
        
        # 3. Regions Data
        regions_df = pd.read_csv(os.path.join(self.data_dir, 'vietnam_regions_2024.csv'))
        self.regions_names_vi = regions_df['region_name_vi'].values
        self.regions_names_en = regions_df['region_name_en'].values
        self.regions_grdp = regions_df['grdp_trillion_VND'].values
        
        # Features for Bai 6 (TOPSIS)
        # Expected: GRDP_capita, Digital, AI, Labor, R&D, Gini
        self.X_regions = np.column_stack([
            regions_df['grdp_per_capita_million_VND'].values,
            regions_df['digital_index_0_100'].values,
            regions_df['ai_readiness_0_100'].values,
            regions_df['trained_labor_pct'].values,
            regions_df['rd_intensity_pct'].values,
            regions_df['gini_coef'].values
        ])

def get_data(data_dir=None):
    return DataLoader(data_dir)
