"""
Data Processing Module
Handles data cleaning, normalization, and quality assessment for messy Monday.com data
"""

import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
import re
from datetime import datetime


class DataProcessor:
    """Process and clean data from Monday.com boards"""

    def __init__(self):
        self.data_quality_issues = []

    def process_deals_data(self, items: List[Dict]) -> Tuple[pd.DataFrame, Dict]:
        """
        Process deals data and return cleaned DataFrame with quality report
        """
        self.data_quality_issues = []
        records = self._items_to_records(items)
        df = pd.DataFrame(records)

        if df.empty:
            return df, {"total_records": 0, "issues": []}

        # Track original counts for quality report
        total_records = len(df)

        # Clean and normalize columns
        df = self._normalize_deal_values(df)
        df = self._normalize_dates(df)
        df = self._normalize_sectors(df)
        df = self._normalize_deal_status(df)

        # Generate quality report
        quality_report = self._generate_quality_report(df, total_records)

        return df, quality_report

    def process_work_orders_data(self, items: List[Dict]) -> Tuple[pd.DataFrame, Dict]:
        """
        Process work orders data and return cleaned DataFrame with quality report
        """
        self.data_quality_issues = []
        records = self._items_to_records(items)
        df = pd.DataFrame(records)

        if df.empty:
            return df, {"total_records": 0, "issues": []}

        total_records = len(df)

        # Clean and normalize columns
        df = self._normalize_amounts(df)
        df = self._normalize_dates(df)
        df = self._normalize_status_fields(df)

        # Generate quality report
        quality_report = self._generate_quality_report(df, total_records)

        return df, quality_report

    def _items_to_records(self, items: List[Dict]) -> List[Dict]:
        """Convert Monday.com items to flat dictionary records"""
        records = []
        for item in items:
            record = {"Name": item.get("name", "")}
            for col_val in item.get("column_values", []):
                col_title = col_val.get("column", {}).get("title", col_val.get("id", "unknown"))
                text_value = col_val.get("text", "")
                record[col_title] = text_value if text_value else None
            records.append(record)
        return records

    def _normalize_deal_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize deal value column to numeric"""
        value_cols = [col for col in df.columns if 'value' in col.lower() or 'amount' in col.lower()]

        for col in value_cols:
            if col in df.columns:
                original_nulls = df[col].isna().sum()
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace('₹', '').str.strip(), errors='coerce')
                new_nulls = df[col].isna().sum()
                if new_nulls > original_nulls:
                    self.data_quality_issues.append(f"{new_nulls - original_nulls} values in '{col}' couldn't be parsed as numbers")

        return df

    def _normalize_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize amount columns in work orders"""
        amount_cols = [col for col in df.columns if 'amount' in col.lower() or 'value' in col.lower() or 'billed' in col.lower()]

        for col in amount_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace('₹', '').str.strip(), errors='coerce')

        return df

    def _normalize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize date columns"""
        date_cols = [col for col in df.columns if 'date' in col.lower()]

        for col in date_cols:
            if col in df.columns:
                original_nulls = df[col].isna().sum()
                df[col] = pd.to_datetime(df[col], errors='coerce')
                new_nulls = df[col].isna().sum()
                if new_nulls > original_nulls:
                    self.data_quality_issues.append(f"{new_nulls - original_nulls} dates in '{col}' couldn't be parsed")

        return df

    def _normalize_sectors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize sector/service values"""
        sector_col = None
        for col in df.columns:
            if 'sector' in col.lower() or 'service' in col.lower():
                sector_col = col
                break

        if sector_col and sector_col in df.columns:
            # Standardize sector names
            sector_mapping = {
                'mining': 'Mining',
                'powerline': 'Powerline',
                'power line': 'Powerline',
                'renewables': 'Renewables',
                'renewable': 'Renewables',
                'tender': 'Tender',
                'oil': 'Oil & Gas',
                'oil & gas': 'Oil & Gas',
                'oil and gas': 'Oil & Gas',
            }

            df[sector_col] = df[sector_col].astype(str).str.strip().str.lower()
            df[sector_col] = df[sector_col].replace(sector_mapping)
            df[sector_col] = df[sector_col].replace('nan', None).replace('none', None)

        return df

    def _normalize_deal_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize deal status values"""
        status_col = None
        for col in df.columns:
            if col.lower() == 'deal status':
                status_col = col
                break

        if status_col and status_col in df.columns:
            df[status_col] = df[status_col].astype(str).str.strip().str.title()
            df[status_col] = df[status_col].replace('Nan', None).replace('None', None)

        return df

    def _normalize_status_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize various status fields in work orders"""
        status_cols = [col for col in df.columns if 'status' in col.lower()]

        for col in status_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()
                df[col] = df[col].replace('Nan', None).replace('None', None)

        return df

    def _generate_quality_report(self, df: pd.DataFrame, total_records: int) -> Dict:
        """Generate a data quality report"""
        null_counts = df.isnull().sum().to_dict()
        null_percentages = {col: f"{(count/total_records)*100:.1f}%" for col, count in null_counts.items() if count > 0}

        return {
            "total_records": total_records,
            "columns": list(df.columns),
            "null_counts": null_counts,
            "null_percentages": null_percentages,
            "issues": self.data_quality_issues,
            "high_null_columns": [col for col, count in null_counts.items() if count/total_records > 0.3]
        }


def get_data_quality_summary(quality_report: Dict) -> str:
    """Generate a human-readable data quality summary"""
    summary = []

    if quality_report.get("high_null_columns"):
        summary.append(f"⚠️ Columns with >30% missing data: {', '.join(quality_report['high_null_columns'])}")

    for issue in quality_report.get("issues", []):
        summary.append(f"⚠️ {issue}")

    if not summary:
        summary.append("✅ Data quality is good - no major issues detected")

    return "\n".join(summary)
