import streamlit as st
import pandas as pd
import json
import io
import os
import tempfile
from datetime import datetime
import zipfile

from tech_pack_extractor import extract_from_pdf

st.set_page_config(
    page_title="Tech Pack OCR Demo",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main-header {
            text-align: center;
            color: #2E8B57;
            margin-bottom: 1rem;
        }
        
        div.stFileUploader div[role="button"] {
            min-height: 150px !important;
            font-size: 20px !important;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px !important;
        }

        div.stFileUploader div[role="button"]:hover {
            background-color: #2E8B57 !important;
            color: white !important;
        }

        div.stFileUploader span {
            font-size: 18px !important;
        }
        
        .stDataFrame table thead tr th {
            background-color: #2E8B57 !important;
            color: white !important;
        }
        
        .stDataFrame table tbody tr td {
            background-color: #2E8B57 !important;
            color: white !important;
        }
        
        .stDataFrame table {
            background-color: #2E8B57 !important;
        }
        
        .stError {
            background-color: #FF6B6B !important;
            color: white !important;
        }
        
        .stSuccess {
            background-color: #4ECDC4 !important;
            color: white !important;
        }
        
        /* Custom styling for primary buttons */
        .stButton > button {
            background-color: #4169E1 !important;
            color: white !important;
            border: none !important;
        }
        
        .stButton > button:hover {
            background-color: #1E90FF !important;
            color: white !important;
        }
        
        .stButton > button:focus {
            background-color: #4169E1 !important;
            color: white !important;
        }
        
        /* Override Streamlit's default red colors with blue */
        
        /* Primary buttons */
        button[kind="primary"] {
            background-color: #4169E1 !important;
            color: white !important;
            border-color: #4169E1 !important;
        }
        
        button[kind="primary"]:hover {
            background-color: #1E90FF !important;
            border-color: #1E90FF !important;
        }
        
        /* Download buttons */
        .stDownloadButton > button {
            background-color: #4169E1 !important;
            color: white !important;
            border: none !important;
        }
        
        .stDownloadButton > button:hover {
            background-color: #1E90FF !important;
        }
        
        /* Error messages - change from red to blue */
        .stAlert[data-baseweb="notification"] {
            background-color: #E6F3FF !important;
            border-left-color: #4169E1 !important;
        }
        
        /* Success messages */
        .stSuccess {
            background-color: #D4EDDA !important;
            border-left-color: #28A745 !important;
        }
        
        /* Warning messages */
        .stWarning {
            background-color: #FFF3CD !important;
            border-left-color: #FFC107 !important;
        }
        
        /* Info messages */
        .stInfo {
            background-color: #E6F3FF !important;
            border-left-color: #4169E1 !important;
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: #4169E1 !important;
        }
        
        /* Spinner */
        .stSpinner > div {
            border-top-color: #4169E1 !important;
        }
        
        /* Tabs - all states */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        
        /* Tabs - default state - remove all borders */
        .stTabs [data-baseweb="tab-list"] button {
            border-bottom: none !important;
        }
        
        /* Tabs - selected/active tab text color only */
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            color: #4169E1 !important;
        }
        
        /* Override any background color on active tab */
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] > div {
            color: #4169E1 !important;
        }
        
        /* Tabs - hover state */
        .stTabs [data-baseweb="tab-list"] button:hover {
            color: #1E90FF !important;
        }
        
        /* Tabs - focus state */
        .stTabs [data-baseweb="tab-list"] button:focus {
            outline: none !important;
        }
        
        /* Tab indicator/underline - this is the ONLY underline shown */
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: #4169E1 !important;
            height: 2px !important;
        }
        
        /* Tab content panel */
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 1rem;
        }
        
        /* Links */
        a {
            color: #4169E1 !important;
        }
        
        a:hover {
            color: #1E90FF !important;
        }
        
        /* Slider */
        .stSlider [role="slider"] {
            background-color: #4169E1 !important;
        }
        
        /* Checkbox when checked */
        .stCheckbox [data-baseweb="checkbox"] {
            background-color: #4169E1 !important;
            border-color: #4169E1 !important;
        }
        
        /* Radio button when selected */
        .stRadio [role="radio"][aria-checked="true"] {
            background-color: #4169E1 !important;
            border-color: #4169E1 !important;
        }
        
        /* Text input focus */
        .stTextInput > div > div > input:focus {
            border-color: #4169E1 !important;
            box-shadow: 0 0 0 0.2rem rgba(65, 105, 225, 0.25) !important;
        }
        
        /* Number input focus */
        .stNumberInput > div > div > input:focus {
            border-color: #4169E1 !important;
            box-shadow: 0 0 0 0.2rem rgba(65, 105, 225, 0.25) !important;
        }
        
        /* Select box focus */
        .stSelectbox > div > div > div:focus-within {
            border-color: #4169E1 !important;
            box-shadow: 0 0 0 0.2rem rgba(65, 105, 225, 0.25) !important;
        }
    </style>
    """, unsafe_allow_html=True)

def validate_pdf_file(uploaded_file):
    """Validate uploaded PDF file."""
    errors = []
    
    if uploaded_file.type != "application/pdf":
        errors.append("⚠️ File must be a PDF")
    
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > 25:
        errors.append(f"⚠️ File size ({file_size_mb:.1f}MB) exceeds 25MB limit")
    
    return errors

def clean_column_names(headers):
    """Clean and deduplicate column names for pandas DataFrame."""
    cleaned_headers = []
    seen_names = {}
    
    for i, header in enumerate(headers):
        clean_header = str(header).strip() if header is not None else ""
        
        if not clean_header or clean_header.isspace():
            clean_header = f"Column_{i+1}"
        
        original_header = clean_header
        counter = 1
        while clean_header in seen_names:
            clean_header = f"{original_header}_{counter}"
            counter += 1
        
        seen_names[clean_header] = True
        cleaned_headers.append(clean_header)
    
    return cleaned_headers


def format_table_for_display(group, table_type):
    """Format table data for better Streamlit display."""
    if not group.get('merged_data') or len(group['merged_data']) == 0:
        return None, False
    
    headers = group['merged_data'][0]
    rows = group['merged_data'][1:]
    
    has_valid_headers = not any('column_' in str(h).lower() for h in headers)
    clean_headers = clean_column_names(headers)
    
    normalized_rows = []
    for row in rows:
        normalized_row = row[:len(clean_headers)]
        while len(normalized_row) < len(clean_headers):
            normalized_row.append("")
        normalized_rows.append(normalized_row)
    
    if not normalized_rows:
        return None, False
    
    df = pd.DataFrame(normalized_rows, columns=clean_headers)
    return df, has_valid_headers

def display_table_group(group, table_type, group_num):
    """Helper function to display a table group."""
    if group.get('merged_data') and len(group['merged_data']) > 0:
        st.markdown(f"#### {table_type} Table {group_num}")
        pages = [str(t['page']) for t in group.get('tables', [])]
        st.write(f"**Source pages:** {', '.join(pages)}")
        
        df, has_headers = format_table_for_display(group, table_type)
        
        if df is not None and not df.empty:
            if has_headers:
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                if table_type == "Measurements":
                    size_columns = [col for col in df.columns 
                                  if any(size in str(col).upper() for size in ['XS', 'S', 'M', 'L', 'XL', 'XXL'])]
                    if size_columns:
                        st.caption(f"📐 Detected size columns: {', '.join(size_columns)}")
            else:
                st.caption("⚠️ No clear table headers detected - displaying raw data:")
                display_data = [
                    {f"Col_{j+1}": str(cell) for j, cell in enumerate(row)}
                    for row in df.values
                ]
                
                if display_data:
                    display_df = pd.DataFrame(display_data)
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.write(f"**Total {table_type.lower()} items:** {len(df)}")
        else:
            st.info(f"No {table_type.lower()} data rows found")

def create_structured_json(result):
    """Create structured JSON representation of extracted data."""
    structured_data = {
        "extraction_metadata": {
            "timestamp": datetime.now().isoformat(),
            "file_info": {
                "text_length": result.get('text_length', 0),
                "first_page_length": result.get('first_page_length', 0)
            }
        },
        "product_identifiers": {},
        "bom_tables": [],
        "measurement_tables": []
    }
    
    fields = result.get('fields', {})
    for field, value in fields.items():
        structured_data["product_identifiers"][field] = {
            "value": value,
            "found": value != "not found"
        }
    
    def process_table_data(tables, table_type):
        """Helper to process table data for JSON structure."""
        processed_tables = []
        for i, group in enumerate(tables):
            if group['merged_data']:
                headers = group['merged_data'][0] if group['merged_data'] else []
                rows = group['merged_data'][1:] if len(group['merged_data']) > 1 else []
                
                table_structure = {
                    "group_id": i + 1,
                    "source_pages": [t['page'] for t in group['tables']],
                    "table_count": len(group['tables']),
                    "headers": headers,
                    "row_count": len(rows),
                    "data": []
                }
                
                for row_idx, row in enumerate(rows):
                    row_data = {
                        "row_index": row_idx + 1,
                        "cells": {}
                    }
                    for col_idx, header in enumerate(headers):
                        cell_value = row[col_idx] if col_idx < len(row) else ""
                        row_data["cells"][header] = {
                            "value": cell_value,
                            "column_index": col_idx
                        }
                    table_structure["data"].append(row_data)
                
                processed_tables.append(table_structure)
        return processed_tables
    
    structured_data["bom_tables"] = process_table_data(result.get('bom_tables', []), "BOM")
    structured_data["measurement_tables"] = process_table_data(result.get('measurement_tables', []), "Measurement")
    
    return structured_data

def create_excel_export_zip(result, filename_base):
    """Create a ZIP file containing all Excel exports."""
    zip_buffer = io.BytesIO()
    
    def create_excel_file(data, sheet_name):
        """Helper to create Excel file in memory."""
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            data.to_excel(writer, sheet_name=sheet_name, index=False)
        return excel_buffer.getvalue()
    
    def normalize_table_data(group):
        """Helper to normalize table data for Excel export."""
        headers = group['merged_data'][0]
        rows = group['merged_data'][1:]
        
        clean_headers = clean_column_names(headers)
        normalized_rows = []
        for row in rows:
            normalized_row = row[:len(clean_headers)]
            while len(normalized_row) < len(clean_headers):
                normalized_row.append("")
            normalized_rows.append(normalized_row)
        
        return pd.DataFrame(normalized_rows, columns=clean_headers)
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        fields = result.get('fields', {})
        
        # Product Identifiers Excel
        identifiers_data = [
            {'Field': field.replace('_', ' ').title(), 'Value': value}
            for field, value in fields.items()
        ]
        
        if identifiers_data:
            df_identifiers = pd.DataFrame(identifiers_data)
            zip_file.writestr(f"{filename_base}_identifiers.xlsx", 
                            create_excel_file(df_identifiers, 'Product Identifiers'))
        
        # BOM Tables Excel
        for i, group in enumerate(result.get('bom_tables', []), 1):
            if group['merged_data'] and len(group['merged_data']) > 0:
                df_bom = normalize_table_data(group)
                zip_file.writestr(f"{filename_base}_bom_group_{i}.xlsx",
                                create_excel_file(df_bom, f'BOM Table {i}'))
        
        # Measurement Tables Excel
        for i, group in enumerate(result.get('measurement_tables', []), 1):
            if group['merged_data'] and len(group['merged_data']) > 0:
                df_measurements = normalize_table_data(group)
                zip_file.writestr(f"{filename_base}_measurements_group_{i}.xlsx",
                                create_excel_file(df_measurements, f'Measurements Group {i}'))
        
        # Summary Excel
        summary_data = []
        for field, value in fields.items():
            summary_data.append({
                'Category': 'Product Identifier',
                'Field': field.replace('_', ' ').title(),
                'Value': value
            })
        
        for i, group in enumerate(result.get('bom_tables', []), 1):
            summary_data.append({
                'Category': 'BOM',
                'Field': f'BOM Table {i}',
                'Value': f"{len(group['merged_data'])-1 if group['merged_data'] else 0} rows"
            })
        
        for i, group in enumerate(result.get('measurement_tables', []), 1):
            summary_data.append({
                'Category': 'Measurements',
                'Field': f'Measurement Table {i}',
                'Value': f"{len(group['merged_data'])-1 if group['merged_data'] else 0} rows"
            })
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            zip_file.writestr(f"{filename_base}_summary.xlsx",
                            create_excel_file(df_summary, 'Summary'))
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def main():
    st.markdown('<h1 class="main-header">Tech Pack OCR Demo</h1>', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Your PDF Tech Pack")
    
    uploaded_file = st.file_uploader(
        "Drag and drop a PDF file here (max 25MB)",
        type=['pdf'],
        help="Select a PDF garment tech pack"
    )
    
    if uploaded_file is not None:
        validation_errors = validate_pdf_file(uploaded_file)
        
        if validation_errors:
            st.error("⚠️ **File Validation Failed**")
            for error in validation_errors:
                st.write(error)
            return
        
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.success(f"✅ **File accepted:** {uploaded_file.name} ({file_size_mb:.1f}MB)")
        
        if st.button("Extract Data", type="primary", use_container_width=True):
            
            with st.spinner('🔄 Processing PDF... This may take a few moments.'):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                try:
                    result = extract_from_pdf(tmp_file_path)
                    
                    if result:
                        st.success("✅ **Extraction Complete!**")
                        filename_base = uploaded_file.name.replace('.pdf', '').replace(' ', '_')
                        
                        tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "🏷️ Identifiers", "📋 BOM", "📏 Measurements"])
                        
                        with tab1:
                            st.markdown("### 📊 Extraction Summary")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                found_identifiers = len([v for v in result['fields'].values() if v != "not found"])
                                total_identifiers = len(result['fields'])
                                st.metric("Product Identifiers", f"{found_identifiers}/{total_identifiers}")
                            
                            with col2:
                                bom_count = len(result.get('bom_tables', []))
                                st.metric("BOM Tables", bom_count)
                            
                            with col3:
                                measurement_count = len(result.get('measurement_tables', []))
                                st.metric("Measurement Tables", measurement_count)
                        
                        with tab2:
                            st.markdown("### 🏷️ Product Identifiers")
                            
                            fields = result.get('fields', {})
                            identifier_data = [
                                {'Field': field.replace('_', ' ').title(), 'Value': value}
                                for field, value in fields.items()
                            ]
                            
                            if identifier_data:
                                df_identifiers = pd.DataFrame(identifier_data)
                                st.dataframe(df_identifiers, use_container_width=True, hide_index=True)
                            else:
                                st.warning("No identifier data found")
                        
                        with tab3:
                            st.markdown("### 📋 Bill of Materials (BOM)")
                            bom_tables = result.get('bom_tables', [])
                            if bom_tables:
                                for i, group in enumerate(bom_tables, 1):
                                    display_table_group(group, "BOM", i)
                            else:
                                st.warning("No BOM tables found")
                        
                        with tab4:
                            st.markdown("### 📏 Measurements & Fit Specifications")
                            measurement_tables = result.get('measurement_tables', [])
                            if measurement_tables:
                                for i, group in enumerate(measurement_tables, 1):
                                    display_table_group(group, "Measurements", i)
                            else:
                                st.warning("No measurement tables found")
                        
                        st.markdown("---")
                        st.markdown("### 📁 Export Data")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            structured_json = create_structured_json(result)
                            json_str = json.dumps(structured_json, indent=2)
                            
                            st.download_button(
                                label="📄 Download JSON",
                                data=json_str,
                                file_name=f"{filename_base}_structured_data.json",
                                mime="application/json",
                                help="Download structured JSON representation"
                            )
                        
                        with col2:
                            excel_zip_data = create_excel_export_zip(result, filename_base)
                            
                            st.download_button(
                                label="📊 Download Excel (ZIP)",
                                data=excel_zip_data,
                                file_name=f"{filename_base}_excel_export.zip",
                                mime="application/zip",
                                help="Download all tables as Excel files in ZIP"
                            )
                    
                    else:
                        st.error("⚠️ Failed to extract data from PDF. Please check the file format and content.")
                
                except Exception as e:
                    st.error(f"⚠️ Error processing file: {str(e)}")
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)

if __name__ == "__main__":
    main()
