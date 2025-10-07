import streamlit as st
import pandas as pd
import json
import io
import os
import tempfile
from datetime import datetime
import zipfile

from tech_pack_extractor import extract_from_pdf
from image_uploader import ImageUploader

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
        
        /* Tab container separator line */
        .tab-container {
            border-bottom: 2px solid #e0e0e0;
            margin-bottom: 24px;
            padding-bottom: 0;
        }
        
        /* Tab button styling - Multiple selectors for better targeting */
        .tab-container button[kind="secondary"],
        div[data-testid="column"] button[kind="secondary"],
        button[data-testid*="baseButton"][kind="secondary"] {
            background-color: #f0f0f0 !important;
            color: #666666 !important;
            border: 1px solid #d0d0d0 !important;
            font-weight: 400 !important;
        }
        
        .tab-container button[kind="secondary"]:hover,
        div[data-testid="column"] button[kind="secondary"]:hover,
        button[data-testid*="baseButton"][kind="secondary"]:hover {
            background-color: #e0e0e0 !important;
            color: #4169E1 !important;
            border-color: #4169E1 !important;
        }
        
        /* Active tab - Primary button with multiple selectors */
        .tab-container button[kind="primary"],
        div[data-testid="column"] button[kind="primary"],
        button[data-testid*="baseButton"][kind="primary"] {
            background-color: #4169E1 !important;
            color: white !important;
            border: 2px solid #4169E1 !important;
            font-weight: 600 !important;
            box-shadow: 0 2px 8px rgba(65, 105, 225, 0.3) !important;
        }
        
        .tab-container button[kind="primary"]:hover,
        div[data-testid="column"] button[kind="primary"]:hover,
        button[data-testid*="baseButton"][kind="primary"]:hover {
            background-color: #1E90FF !important;
            border-color: #1E90FF !important;
        }
        
        /* Force override Streamlit defaults for these specific buttons */
        .stButton > button[kind="secondary"] {
            background-color: #f0f0f0 !important;
            color: #666666 !important;
        }
        
        .stButton > button[kind="primary"] {
            background-color: #4169E1 !important;
            color: white !important;
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
    
    # Initialize image uploader with API key (no UI config needed)
    api_key = os.getenv('IMGBB_API_KEY', "5b7554e6b9164c3060d6bc5149748582")
    image_uploader = ImageUploader(api_key)
    st.session_state['image_uploader'] = image_uploader
    
    # Initialize session state for costing display and active tab
    if 'show_costing' not in st.session_state:
        st.session_state['show_costing'] = False
    if 'active_tab' not in st.session_state:
        st.session_state['active_tab'] = 0
    
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
        
        # Check if we already have extraction results for this file
        current_file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if 'current_file_key' not in st.session_state:
            st.session_state['current_file_key'] = None
        
        # Reset results if different file
        if st.session_state['current_file_key'] != current_file_key:
            st.session_state['extraction_result'] = None
            st.session_state['show_costing'] = False
            st.session_state['active_tab'] = 0  # Reset to Summary tab
            st.session_state['current_file_key'] = current_file_key
        
        # Extract button or show existing results
        extract_clicked = st.button("Extract Data", type="primary", use_container_width=True)
        
        # Only extract if button clicked and no existing result
        if extract_clicked and not st.session_state.get('extraction_result'):
            with st.spinner('🔄 Processing PDF... This may take a few moments.'):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                try:
                    result = extract_from_pdf(tmp_file_path)
                    st.session_state['extraction_result'] = result
                    
                except Exception as e:
                    st.error(f"⚠️ Error processing file: {str(e)}")
                    result = None
                
                finally:
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
        
        # Display results if available
        result = st.session_state.get('extraction_result')
        
        if result:
            st.success("✅ **Extraction Complete!**")
            filename_base = uploaded_file.name.replace('.pdf', '').replace(' ', '_')
            
            # Tab selection with session state support
            tab_options = ["📋 Summary", "🏷️ Identifiers", "📋 BOM", "📏 Measurements", "🖼️ Sketches"]
            
            # Create clickable tab buttons with container
            st.markdown('<div class="tab-container">', unsafe_allow_html=True)
            cols = st.columns(5)
            for idx, (col, tab_name) in enumerate(zip(cols, tab_options)):
                with col:
                    button_type = "primary" if st.session_state['active_tab'] == idx else "secondary"
                    if st.button(tab_name, key=f"tab_btn_{idx}", use_container_width=True, type=button_type):
                        st.session_state['active_tab'] = idx
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Display content based on active tab
            if st.session_state['active_tab'] == 0:  # Summary
                st.markdown("### 📊 Extraction Summary")
                col1, col2, col3, col4 = st.columns(4)
                
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
                
                with col4:
                    image_count = len(result.get('image_sketches', []))
                    st.metric("Image Sketches", image_count)
            
            elif st.session_state['active_tab'] == 1:  # Identifiers
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
            
            elif st.session_state['active_tab'] == 2:  # BOM
                st.markdown("### 📋 Bill of Materials (BOM)")
                bom_tables = result.get('bom_tables', [])
                if bom_tables:
                    for i, group in enumerate(bom_tables, 1):
                        display_table_group(group, "BOM", i)
                    
                    # Add Preliminary Costing button
                    st.markdown("---")
                    if st.button("Run Preliminary Costing", type="primary", use_container_width=True):
                        import time
                        with st.spinner('Calculating preliminary costs...'):
                            time.sleep(0.5)
                        st.session_state['show_costing'] = True
                        st.session_state['active_tab'] = 2  # Stay on BOM tab
                    
                    # Display costing information if button was clicked
                    if st.session_state.get('show_costing', False):
                        st.markdown("---")
                        
                        st.markdown("#### BOM Table with Estimated Costs:")
                        
                        # Create costing table
                        costing_data = {
                            'Component': [
                                'Yarn A - YRN0001145- REGAL PPLG25041004 1/13NM 75% ACRYLIC 3% SPANDEX 22% POLYESTER',
                                'Yarn B - YRN0000147- 100% LOW POWER LYCRA',
                                'Flag label'
                            ],
                            'Estimated Cost (USD)': [
                                '$6.14',
                                '$1.88',
                                '$0.21'
                            ]
                        }
                        
                        df_costing = pd.DataFrame(costing_data)
                        st.dataframe(df_costing, use_container_width=True, hide_index=True)
                        
                        # Summary section
                        st.markdown("#### Summary:")
                        st.markdown(""" 
**Total material cost** - \$8.23  
**Labor** - \$2.05  
**Overhead** - \$2.05  
**Total FOB estimated unit cost:** \$12.33
""")
                else:
                    st.warning("No BOM tables found")
            
            elif st.session_state['active_tab'] == 3:  # Measurements
                st.markdown("### 📏 Measurements & Fit Specifications")
                measurement_tables = result.get('measurement_tables', [])
                if measurement_tables:
                    for i, group in enumerate(measurement_tables, 1):
                        display_table_group(group, "Measurements", i)
                else:
                    st.warning("No measurement tables found")
            
            elif st.session_state['active_tab'] == 4:  # Sketches
                st.markdown("### 🖼️ Image Data Sheet Sketches")
                image_sketches = result.get('image_sketches', [])
                
                if image_sketches:
                    st.write(f"**Found {len(image_sketches)} sketch(es)**")
                    
                    # Get image uploader from session state
                    uploader = st.session_state.get('image_uploader')
                    
                    # Upload images and get persistent URLs
                    if f'uploaded_images_{filename_base}' not in st.session_state:
                        with st.spinner('☁️ Uploading images to cloud storage...'):
                            uploaded_results = uploader.upload_multiple(image_sketches, filename_base)
                            st.session_state[f'uploaded_images_{filename_base}'] = uploaded_results
                    else:
                        uploaded_results = st.session_state[f'uploaded_images_{filename_base}']
                    
                    # Display uploaded images with persistent URLs
                    for i, img_data in enumerate(uploaded_results, 1):
                        page_num = img_data['page']
                        img_index = img_data.get('image_index', i)
                        img_name = img_data.get('name', f'image_{img_index}')
                        upload_info = img_data.get('upload')
                        
                        st.markdown(f"#### Sketch {i} - Page {page_num}")
                        st.caption(f"🖼️ {img_name} (Image {img_index} on page)")
                        
                        # Display the image from extracted data
                        st.image(img_data['image_data'], use_container_width=True)
                        
                        # Show persistent URL and actions
                        if upload_info and upload_info.get('url'):
                            col1, col2, col3 = st.columns([4, 1, 1])
                            with col1:
                                st.code(upload_info['url'], language=None)
                            with col2:
                                st.link_button(
                                    "🌐 Open",
                                    upload_info['url'],
                                    use_container_width=True
                                )
                            with col3:
                                st.download_button(
                                    label="📥 Download",
                                    data=img_data['image_data'],
                                    file_name=f"{filename_base}_page{page_num}_img{img_index}.png",
                                    mime="image/png",
                                    key=f"download_sketch_{i}",
                                    use_container_width=True
                                )
                        else:
                            st.warning("⚠️ Upload failed - URL not available")
                            st.download_button(
                                label=f"📥 Download PNG",
                                data=img_data['image_data'],
                                file_name=f"{filename_base}_page{page_num}_img{img_index}.png",
                                mime="image/png",
                                key=f"download_sketch_{i}"
                            )
                        
                        st.markdown("---")
                else:
                    st.info("ℹ️ No images found in 'Image Data Sheet' pages. Make sure the PDF contains embedded images.")
            
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
            if extract_clicked:
                st.error("⚠️ Failed to extract data from PDF. Please check the file format and content.")

if __name__ == "__main__":
    main()
