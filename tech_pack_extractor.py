"""
PDF Garment Tech Pack Extractor

Extracts product identifiers, BOM tables, and measurements from garment tech pack PDFs.
"""

import pdfplumber
import re
import os
import pandas as pd
from difflib import SequenceMatcher
from typing import Dict, List, Optional


def similarity(a, b):
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def extract_value_after_keyword(text, keyword_pos, keyword):
    """Extract value that appears after a keyword."""
    after_text = text[keyword_pos + len(keyword):].strip()
    after_text = re.sub(r'^[:\-_\s=|]+', '', after_text)
    
    if not after_text:
        return None
    
    parts = re.split(r'\s{2,}|[,;\n]|(?=\b(?:style|brand|season|designer|date|name|number)\b)', 
                     after_text, flags=re.IGNORECASE)
    
    if parts and parts[0].strip():
        value = parts[0].strip()
        value = re.sub(r'[:\-_\s]+$', '', value)
        return value if len(value) > 1 else None
    
    return None


def extract_fields_from_text(text: str) -> Dict[str, str]:
    """Extract product identifier fields from first two pages text."""
    
    field_patterns = {
        'brand': [
            r'brand[:\s]*(.+)',
            r'company[:\s]*(.+)',
            r'manufacturer[:\s]*(.+)'
        ],
        'description': [
            r'description[:\s]*(.+)',
            r'product\s*description[:\s]*(.+)',
            r'item\s*description[:\s]*(.+)'
        ],
        'style_number': [
            r'style\s*no\.?\s*([A-Z0-9\-_]+)',
            r'style\s*number[:\s]*([A-Z0-9\-_]+)',
            r'sku[:\s]*([A-Z0-9\-_]+)'
        ],
        'season_year': [
            r'season[:\s]*([^,\n\r]+?)(?:\s*$|\s*[,;\n\r]|\s*[a-z]{2,})',
            r'season[/\s]*year[:\s]*([^,\n\r]+?)(?:\s*$|\s*[,;\n\r]|\s*[a-z]{2,})',
            r'collection[:\s]*([^,\n\r]+?)(?:\s*$|\s*[,;\n\r]|\s*[a-z]{2,})',
            r'\b(20\d{2})\b',
            r'\b(spring|summer|fall|autumn|winter)\s*(20\d{2})\b',
            r'\b(spring|summer|fall|autumn|winter)\b(?!\s*(20\d{2}))',
            r'\b(20\d{2})\b(?!\s*(spring|summer|fall|autumn|winter))'
        ],
        'model_id': [
            r'model\s*#[:\s]*([A-Z0-9\-_]+)',
            r'model\s*id[:\s]*([A-Z0-9\-_]+)',
            r'model\s*number[:\s]*([A-Z0-9\-_]+)'
        ],
        'product_line': [
            r'product\s*line[:\s]*(.+)',
            r'line[:\s]*(.+)',
            r'collection[:\s]*(.+)'
        ],
        'date': [
            r'date[:\s]*([^,]+)',
            r'created[:\s]*([^,]+)',
            r'(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})'
        ]
    }
    
    extracted = {}
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or _is_skip_line(line):
            continue
        
        if line.startswith('Brand ') and 'brand' not in extracted:
            extracted['brand'] = line[6:].strip()
        elif line.startswith('Description ') and 'description' not in extracted:
            extracted['description'] = line[12:].strip()
        elif line.startswith('Style No. ') and 'style_number' not in extracted:
            extracted['style_number'] = line[10:].strip()
        elif line.startswith('Season ') and 'season_year' not in extracted:
            extracted['season_year'] = line[7:].strip()
        elif line.startswith('Model #') and 'model_id' not in extracted:
            extracted['model_id'] = line[7:].strip()
        elif line.startswith('Product Line ') and 'product_line' not in extracted:
            extracted['product_line'] = line[13:].strip()
        elif line.startswith('Date ') and 'date' not in extracted:
            extracted['date'] = line[5:].strip()
    
    for field_name, patterns in field_patterns.items():
        if field_name in extracted:
            continue
            
        for pattern in patterns:
            for line in lines:
                if _is_skip_line(line.strip()):
                    continue
                    
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    value = match.group(1).strip() if match.lastindex else match.group(0).strip()
                    
                    if _is_valid_value(value):
                        cleaned_value = _clean_value(value, field_name)
                        if cleaned_value is not None:
                            extracted[field_name] = cleaned_value
                            break
            if field_name in extracted:
                break
    
    _extract_date_and_year(lines, extracted)
    _extract_season_year_smart(text, extracted)
    
    required_fields = ['brand', 'description', 'style_number', 'season_year', 'composition', 'material_finishes', 'yarn_gauge', 'model_id', 'product_line', 'date']
    for field in required_fields:
        if field not in extracted or not extracted[field]:
            extracted[field] = ''
    
    return extracted


def _is_skip_line(line: str) -> bool:
    """Check if line should be skipped during extraction."""
    if not line or line.startswith('Page '):
        return True
    
    skip_patterns = [
        r'page\s+\d+\s+of\s+\d+',
        r'created with backbone',
        r'tolerance.*point of measure',
        r'\d+\.\d+\s*in\b'
    ]
    
    line_lower = line.lower()
    return any(re.search(pattern, line_lower) for pattern in skip_patterns)


def _is_valid_value(value: str) -> bool:
    """Check if extracted value is valid."""
    if not value or len(value.strip()) < 2:
        return False
    
    value_lower = value.strip().lower()
    invalid_values = {'n/a', 'na', 'none', 'null', 'tbd', '...', 'xxx'}
    
    return value_lower not in invalid_values and not all(c in ':-_.,;()[]{}' for c in value)


def _clean_value(value: str, field_type: str) -> str:
    """Clean and format extracted value."""
    value = value.strip()
    value = re.sub(r'^["\'\[\(]+|["\'\]\)]+$', '', value)
    
    if field_type == 'description' and len(value) > 100:
        return None
    
    if field_type == 'date':
        date_part = value.split(',')[0].strip()
        date_match = re.search(r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}', date_part)
        return date_match.group(0) if date_match else date_part
    elif field_type in ['style_number', 'model_id']:
        code_match = re.search(r'[A-Z0-9\-_]{3,}', value.upper())
        return code_match.group(0) if code_match else value
    elif field_type == 'season_year':
        year_match = re.search(r'20\d{2}', value)
        return year_match.group(0) if year_match else value
    
    return value


def _extract_date_and_year(lines: List[str], extracted: Dict[str, str]) -> None:
    """Extract date and derive year for season_year."""
    if 'date' in extracted:
        return
        
    for line in lines:
        date_match = re.search(r'(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})', line)
        if date_match:
            date_value = date_match.group(1)
            date_part = date_value.split(',')[0].strip()
            extracted['date'] = date_part
            year_match = re.search(r'(20\d{2})', date_part)
            if year_match and 'season_year' not in extracted:
                extracted['season_year'] = year_match.group(1)
            break


def _extract_season_year_smart(text: str, extracted: Dict[str, str]) -> None:
    """Extract season/year with intelligent cut-off points."""
    if 'season_year' in extracted:
        return
    
    seasons = ['spring', 'summer', 'fall', 'autumn', 'winter']
    
    season_year_pattern = r'\b(spring|summer|fall|autumn|winter)\s*(20\d{2})\b'
    season_year_match = re.search(season_year_pattern, text, re.IGNORECASE)
    if season_year_match:
        season = season_year_match.group(1).capitalize()
        year = season_year_match.group(2)
        extracted['season_year'] = f"{season} {year}"
        return
    
    for season in seasons:
        season_pattern = r'\b' + season + r'\b(?!\s*(20\d{2}))'
        if re.search(season_pattern, text, re.IGNORECASE):
            extracted['season_year'] = season.capitalize()
            return
    
    patterns = [
        r'season[/\s]*year[:\s]*([^,\n\r]+?)(?:\s*$|\s*[,;\n\r]|\s*[a-z]{2,})',
        r'collection[:\s]*([^,\n\r]+?)(?:\s*$|\s*[,;\n\r]|\s*[a-z]{2,})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            value = re.split(r'[,;\n\r]', value)[0].strip()
            value = re.split(r'\s+(?:style|brand|designer|developer|created|date|size|color)', 
                           value, flags=re.IGNORECASE)[0].strip()
            if value and len(value) <= 20:
                extracted['season_year'] = value
                return
    
    year_match = re.search(r'\b(20\d{2})\b', text)
    if year_match:
        extracted['season_year'] = year_match.group(1)
        return


def _extract_bom_fields(extracted: Dict[str, str], bom_tables: List[Dict], verbose: bool = False) -> None:
    """Extract Composition, Material Finishes, and Yarn Gauge from BOM tables."""
    for group_idx, group in enumerate(bom_tables):
        if not group.get('merged_data') or len(group['merged_data']) < 2:
            continue
            
        headers = group['merged_data'][0]
        rows = group['merged_data'][1:]
        
        main_material_col_idx = None
        main_material_row = None
        
        for i, header in enumerate(headers):
            if header and 'main material' in str(header).lower():
                main_material_col_idx = i
                break
        
        if main_material_col_idx is not None:
            for row_idx, row in enumerate(rows):
                if (main_material_col_idx < len(row) and 
                    row[main_material_col_idx] and 
                    str(row[main_material_col_idx]).strip()):
                    main_material_row = row
                    break
        
        if main_material_row is None:
            for row_idx, row in enumerate(rows):
                non_empty_cells = sum(1 for cell in row if cell and str(cell).strip())
                if non_empty_cells >= 3:
                    main_material_row = row
                    break
        
        if main_material_row is None:
            continue
        
        for i, header in enumerate(headers):
            if i >= len(main_material_row):
                continue
                
            header_lower = str(header).lower().strip()
            cell_value = str(main_material_row[i]).strip()
            
            if not cell_value:
                continue
                
            normalized_header = ' '.join(header_lower.split())
            
            if ('composition' not in extracted or not extracted['composition']):
                if normalized_header == 'material information' or normalized_header == 'product':
                    extracted['composition'] = cell_value
            
            if ('material_finishes' not in extracted or not extracted['material_finishes']):
                finish_keywords = ['material finishes', 'finishes', 'finish', 'treatment', 'coating']
                if any(keyword in header_lower for keyword in finish_keywords):
                    extracted['material_finishes'] = cell_value
            
            if ('yarn_gauge' not in extracted or not extracted['yarn_gauge']):
                gauge_keywords = ['gauge', 'count', 'denier', 'tex']
                if any(keyword in header_lower for keyword in gauge_keywords):
                    gauge_value = cell_value
                    number_of_ends_value = ""
                    for j, end_header in enumerate(headers):
                        end_keywords = ['number of ends', 'ends', 'filaments', 'strands']
                        if any(keyword in str(end_header).lower() for keyword in end_keywords):
                            if j < len(main_material_row) and main_material_row[j]:
                                number_of_ends_value = str(main_material_row[j]).strip()
                            break
                    
                    if number_of_ends_value:
                        extracted['yarn_gauge'] = f"{gauge_value}/{number_of_ends_value}"
                    else:
                        extracted['yarn_gauge'] = gauge_value


def _extract_brand_from_pool(text: str, brand_pool: List[str], extracted: Dict[str, str], verbose: bool = False) -> None:
    """Extract brand from predefined brand pool anywhere in the document."""
    if 'brand' in extracted and extracted['brand']:
        return
    
    # Official brand names mapping
    official_brands = {
        'calvin klein': 'Calvin Klein',
        'ck': 'Calvin Klein',
        'ten thousand': 'Ten Thousand',
        'maurices': 'MAURICES',
        'everlast': 'EVERLAST',
        'greenway': 'GREENWAY'
    }
    
    cleaned_text = re.sub(r'\s+', ' ', text.strip())
    text_lower = cleaned_text.lower()
    
    for brand in brand_pool:
        brand_lower = brand.lower()
        
        if brand_lower in text_lower:
            extracted['brand'] = brand
            return
        
        brand_words = brand_lower.split()
        if len(brand_words) > 1:
            flexible_pattern = r'\b' + r'\s+'.join(re.escape(word) for word in brand_words) + r'\b'
            match = re.search(flexible_pattern, text_lower)
            if match:
                extracted['brand'] = brand
                return
            
            minimal_pattern = r'\b' + r'\s*'.join(re.escape(word) for word in brand_words) + r'\b'
            match = re.search(minimal_pattern, text_lower)
            if match:
                extracted['brand'] = brand
                return
    
    for variation, official_name in official_brands.items():
        if variation in text_lower:
            extracted['brand'] = official_name
            return



def _extract_images_from_page(page, page_num: int, verbose: bool = False) -> List[Dict]:
    """Extract embedded images from a PDF page."""
    extracted_images = []
    
    try:
        if hasattr(page, 'images') and page.images:
            for img_idx, img_info in enumerate(page.images):
                try:
                    img_name = img_info.get('name', f'image_{img_idx+1}')
                    
                    bbox = (
                        img_info['x0'],
                        img_info['top'],
                        img_info['x0'] + img_info['width'],
                        img_info['top'] + img_info['height']
                    )
                    
                    cropped_img = page.within_bbox(bbox).to_image(resolution=300)
                    
                    import io
                    img_byte_arr = io.BytesIO()
                    cropped_img.original.save(img_byte_arr, format='PNG', optimize=False, compress_level=1)
                    img_byte_arr.seek(0)
                    
                    extracted_images.append({
                        'page': page_num,
                        'image_index': img_idx + 1,
                        'image_data': img_byte_arr.getvalue(),
                        'format': 'PNG',
                        'name': img_name
                    })
                
                except Exception:
                    pass
    
    except Exception:
        pass
    
    return extracted_images


def debug_bom_tables(bom_tables: List[Dict]) -> None:
    """Debug function to inspect BOM table structure."""
    print(f"\n🔍 BOM TABLE DEBUG INFO:")
    print(f"   Total BOM groups: {len(bom_tables)}")
    
    for i, group in enumerate(bom_tables, 1):
        print(f"\n   📋 BOM Group {i}:")
        
        if not group.get('merged_data'):
            print(f"     ❌ No merged data")
            continue
            
        headers = group['merged_data'][0]
        rows = group['merged_data'][1:]
        
        print(f"     Headers ({len(headers)}): {headers}")
        print(f"     Rows: {len(rows)}")
        
        # Show first few rows
        for j, row in enumerate(rows[:3]):  # Show first 3 rows
            print(f"       Row {j+1}: {row}")
        
        if len(rows) > 3:
            print(f"       ... and {len(rows) - 3} more rows")
    
    print()


def extract_from_multiple_pdfs(pdf_paths: List[str], verbose: bool = False) -> List[Dict]:
    """Extract data from multiple PDF files."""
    if verbose:
        print(f"📄 Processing {len(pdf_paths)} PDF files...")
    
    results = []
    
    for i, pdf_path in enumerate(pdf_paths, 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing file {i}/{len(pdf_paths)}: {os.path.basename(pdf_path)}")
            print(f"{'='*60}")
        
        try:
            result = extract_from_pdf(pdf_path, verbose=verbose)
            if result:
                result['filename'] = os.path.basename(pdf_path)
                result['file_path'] = pdf_path
                results.append(result)
            else:
                results.append({
                    'filename': os.path.basename(pdf_path),
                    'file_path': pdf_path,
                    'error': 'Failed to extract data'
                })
                
        except Exception as e:
            if verbose:
                print(f"❌ Error processing {os.path.basename(pdf_path)}: {e}")
            results.append({
                'filename': os.path.basename(pdf_path),
                'file_path': pdf_path,
                'error': str(e)
            })
    
    if verbose:
        successful = len([r for r in results if 'error' not in r])
        failed = len([r for r in results if 'error' in r])
        print(f"\n{'='*60}")
        print(f"📊 BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total files: {len(pdf_paths)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        if failed > 0:
            print(f"\n❌ Failed files:")
            for result in results:
                if 'error' in result:
                    print(f"  • {result['filename']}: {result['error']}")
    
    return results


def extract_from_pdf(pdf_path: str, verbose: bool = False) -> Optional[Dict]:
    """Extract product identifiers, BOM tables, and measurements from PDF."""
    if verbose:
        print(f"📄 Processing: {os.path.basename(pdf_path)}")
    
    try:
        result = {
            'full_text': "",
            'first_page_text': "",
            'fields': {},
            'bom_tables': [],
            'measurement_tables': [],
            'image_sketches': [],
            'text_length': 0,
            'first_page_length': 0
        }
        
        bom_tables = []
        measurement_tables = []
        image_sketches = []
        bom_keywords = ['bom', 'bill of materials']
        measurement_keywords = ['measurement', 'fit specifications', 'size specifications', 'points of measure']
        image_keyword = 'image data sheet'
        
        with pdfplumber.open(pdf_path) as pdf:
            if verbose:
                print(f"   Pages: {len(pdf.pages)}")
            
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if not text:
                    continue
                    
                result['full_text'] += f"\n--- Page {page_num} ---\n{text}"
                text_lower = text.lower()
                
                if page_num <= 2:
                    if page_num == 1:
                        result['first_page_text'] = text
                    else:
                        result['first_page_text'] += f"\n--- Page {page_num} ---\n{text}"
                
                if image_keyword in text_lower:
                    page_images = _extract_images_from_page(page, page_num, verbose)
                    if page_images:
                        image_sketches.extend(page_images)
                
                text_lines = text.split('\n')
                first_two_lines = '\n'.join(text_lines[:2]).lower() if len(text_lines) >= 2 else text_lower
                
                is_bom_page = any(keyword in first_two_lines for keyword in bom_keywords)
                is_measurement_page = any(keyword in first_two_lines for keyword in measurement_keywords)
                
                if not is_bom_page and not is_measurement_page:
                    continue
                
                tables = page.extract_tables()
                if not tables:
                    continue
                
                for i, table in enumerate(tables):
                    if not table or len(table) == 0:
                        continue
                        
                    cleaned_table = _clean_table(table)
                    if not cleaned_table:
                        continue
                    
                    table_info = {
                        'page': page_num,
                        'table_num': i + 1,
                        'data': cleaned_table,
                        'headers': cleaned_table[0] if cleaned_table else [],
                        'num_cols': len(cleaned_table[0]) if cleaned_table else 0
                    }
                    
                    if is_bom_page:
                        bom_tables.append(table_info)
                    
                    if is_measurement_page:
                        measurement_tables.append(table_info)
                
        
        result['fields'] = extract_fields_from_text(result['first_page_text'])
        
        brand_pool = ['Calvin Klein', 'Ten Thousand', 'MAURICES', 'EVERLAST', 'GREENWAY']
        _extract_brand_from_pool(result['full_text'], brand_pool, result['fields'], verbose)
        
        result['bom_tables'] = _merge_similar_tables(bom_tables, "BOM", verbose)
        result['measurement_tables'] = _merge_similar_tables(measurement_tables, "Measurement", verbose)
        
        if verbose:
            debug_bom_tables(result['bom_tables'])
        
        _extract_bom_fields(result['fields'], result['bom_tables'], verbose)
        result['image_sketches'] = image_sketches
        result['text_length'] = len(result['full_text'])
        result['first_page_length'] = len(result['first_page_text'])
        
        if verbose:
            _print_extraction_summary(result, bom_tables, measurement_tables, image_sketches)
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"❌ Error: {e}")
        return None


def _clean_table(table: List[List]) -> List[List]:
    """Clean table data by removing empty rows and normalizing content."""
    if not table:
        return []
    
    cleaned = []
    for row in table:
        if row and any(cell and str(cell).strip() for cell in row):
            cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
            if not _is_pagination_row(cleaned_row):
                cleaned.append(cleaned_row)
    
    if not cleaned:
        return []
    
    cleaned = _detect_and_fix_headers(cleaned)
    cleaned = _remove_blank_columns(cleaned)
    
    return cleaned


def _remove_blank_columns(table: List[List]) -> List[List]:
    """Remove columns that are completely blank or contain only empty strings."""
    if not table or not table[0]:
        return table
    
    num_columns = len(table[0])
    blank_columns = []
    
    for col_idx in range(num_columns):
        is_blank = True
        for row in table:
            if col_idx < len(row):
                cell_value = str(row[col_idx]).strip()
                if cell_value:
                    is_blank = False
                    break
        
        if is_blank:
            blank_columns.append(col_idx)
    
    if not blank_columns:
        return table
    
    cleaned_table = []
    for row in table:
        cleaned_row = []
        for col_idx in range(num_columns):
            if col_idx not in blank_columns:
                if col_idx < len(row):
                    cleaned_row.append(row[col_idx])
                else:
                    cleaned_row.append("")
        cleaned_table.append(cleaned_row)
    
    return cleaned_table


def _is_pagination_row(row: List[str]) -> bool:
    """Check if row contains pagination or navigation information."""
    row_text = ' '.join(str(cell).strip() for cell in row if cell).lower()
    
    if not row_text:
        return False
    
    pagination_patterns = [
        r'displaying\s+\d+\s*[-–]\s*\d+\s+of\s+\d+\s+results?',
        r'showing\s+\d+\s*[-–]\s*\d+\s+of\s+\d+',
        r'page\s+\d+\s+of\s+\d+',
        r'results?\s+\d+\s*[-–]\s*\d+\s+of\s+\d+',
        r'items?\s+\d+\s*[-–]\s*\d+\s+of\s+\d+',
        r'records?\s+\d+\s*[-–]\s*\d+\s+of\s+\d+',
        r'entries?\s+\d+\s*[-–]\s*\d+\s+of\s+\d+',
        r'\d+\s*[-–]\s*\d+\s+of\s+\d+\s+(?:items?|results?|records?|entries?)',
        r'(?:first|previous|next|last)\s+(?:\d+\s+)?(?:items?|results?|records?)',
        r'total\s+(?:items?|results?|records?):\s*\d+',
        r'showing\s+(?:all\s+)?\d+\s+(?:items?|results?|records?)'
    ]
    
    for pattern in pagination_patterns:
        if re.search(pattern, row_text):
            return True
    
    if any(nav_word in row_text for nav_word in ['displaying', 'showing', 'results', 'total', 'page']):
        numbers = re.findall(r'\d+', row_text)
        words = row_text.split()
        if len(numbers) >= 2 and len(words) <= 8:
            return True
    
    navigation_phrases = [
        'first', 'previous', 'next', 'last', 'show more', 'load more',
        'view all', 'see more', 'continue', '...', 'more results'
    ]
    
    if row_text.strip() in navigation_phrases:
        return True
    
    if re.match(r'^[\d\s\-–<>«»‹›\[\]().,|]+$', row_text):
        return True
    
    return False


def _detect_and_fix_headers(table: List[List]) -> List[List]:
    """Detect if the table has proper headers and fix missing ones."""
    if len(table) < 2:
        return table
    
    # Assume first row is headers
    headers = table[0]
    data_rows = table[1:]
    
    # Check if headers look like actual headers
    if not _are_valid_headers(headers, data_rows):
        # Try to find real headers in the first few rows
        for i in range(min(3, len(table))):
            potential_headers = table[i]
            remaining_rows = table[i+1:]
            
            if remaining_rows and _are_valid_headers(potential_headers, remaining_rows):
                return [potential_headers] + remaining_rows
        
        # If no valid headers found, generate them
        headers = _generate_headers(headers)
        return [headers] + data_rows
    
    return table


def _are_valid_headers(headers: List[str], data_rows: List[List]) -> bool:
    """Check if the given row looks like valid table headers."""
    if not headers or not data_rows:
        return False
    
    header_scores = []
    
    for i, header in enumerate(headers):
        header = str(header).strip().lower()
        
        if not header:
            header_scores.append(0)
            continue
        
        score = 0
        
        if any(keyword in header for keyword in [
            'name', 'description', 'qty', 'quantity', 'code', 'style', 'color',
            'size', 'material', 'fabric', 'component', 'measure', 'tolerance',
            'point', 'vendor', 'supplier', 'notes', 'comments', 'type', 'category', 'tolerance'
        ]):
            score += 3
        
        if any(word in header for word in [
            'no', 'num', 'id', 'ref', 'spec', 'min', 'max', 'total', 'unit',
            'price', 'cost', 'date', 'status', 'location', 'page'
        ]):
            score += 2
        
        if len(header.split()) <= 4 and header.isalpha():
            score += 1
        
        column_data = []
        for row in data_rows[:5]:
            if i < len(row):
                cell = str(row[i]).strip()
                if cell:
                    column_data.append(cell)
        
        if column_data:
            data_types = _analyze_column_data_types(column_data)
            if header not in column_data and not _looks_like_data(header, data_types):
                score += 2
        
        header_scores.append(score)
    
    avg_score = sum(header_scores) / len(header_scores) if header_scores else 0
    return avg_score >= 1.5


def _analyze_column_data_types(data: List[str]) -> Dict:
    """Analyze the types of data in a column."""
    analysis = {
        'numeric': 0,
        'date': 0,
        'short_text': 0,
        'long_text': 0,
        'code': 0
    }
    
    for item in data:
        item = str(item).strip()
        if not item:
            continue
        
        if re.search(r'\d+\.?\d*\s*(yd|in|cm|mm|pc|piece|each|ea|lbs|kg|oz)', item.lower()):
            analysis['numeric'] += 1
        elif re.search(r'^\d+\.?\d*$', item):
            analysis['numeric'] += 1
        elif re.search(r'\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}', item):
            analysis['date'] += 1
        elif re.search(r'^[A-Z]{1,3}\d+|^\d+[A-Z]+|\w*\d+\w*$', item.upper()):
            analysis['code'] += 1
        elif len(item.split()) <= 3:
            analysis['short_text'] += 1
        else:
            analysis['long_text'] += 1
    
    return analysis


def _looks_like_data(text: str, data_types: Dict) -> bool:
    """Check if text looks like data rather than a header."""
    text = str(text).strip()
    
    if re.search(r'^\d+\.?\d*$', text):
        return True
    if re.search(r'\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}', text):
        return True
    if re.search(r'^[A-Z]{1,3}\d+|^\d+[A-Z]+', text.upper()):
        return True
    
    return False


def _generate_headers(original_headers: List[str]) -> List[str]:
    """Generate meaningful headers when originals are not valid."""
    generated = []
    
    for i, header in enumerate(original_headers):
        header = str(header).strip()
        
        if not header or header.lower() in ['', 'none', 'null']:
            if i == 0:
                generated.append("Item")
            elif i == 1:
                generated.append("Description")
            elif i == 2:
                generated.append("Quantity")
            elif i == 3:
                generated.append("Code")
            else:
                generated.append(f"Column_{i+1}")
        else:
            clean_header = re.sub(r'[^\w\s]', ' ', header)
            clean_header = ' '.join(clean_header.split())
            if clean_header:
                generated.append(clean_header)
            else:
                generated.append(f"Column_{i+1}")
    
    return generated


def _merge_similar_tables(tables: List[Dict], table_type: str, verbose: bool = False) -> List[Dict]:
    """Merge tables with similar structure using two-phase approach: horizontal first, then vertical."""
    if not tables:
        return []
    
    horizontal_groups = []
    
    for table in tables:
        merged = False
        for group in horizontal_groups:
            if _have_same_ids(table, group['tables'][0]):
                group['tables'].append(table)
                merged = True
                break
        
        if not merged:
            horizontal_groups.append({
                'type': table_type,
                'tables': [table],
                'merged_data': None
            })
    
    for group in horizontal_groups:
        group['merged_data'] = _merge_table_data(group['tables'])
    
    vertical_groups = []
    
    for h_group in horizontal_groups:
        merged = False
        for v_group in vertical_groups:
            if _can_merge_vertically(h_group, v_group):
                v_group['tables'].extend(h_group['tables'])
                if h_group['merged_data'] and len(h_group['merged_data']) > 1:
                    v_group['merged_data'].extend(h_group['merged_data'][1:])
                merged = True
                break
        
        if not merged:
            vertical_groups.append({
                'type': table_type,
                'tables': h_group['tables'],
                'merged_data': h_group['merged_data']
            })
    
    valid_groups = []
    for group in vertical_groups:
        if group['merged_data'] and len(group['merged_data']) > 1:
            valid_groups.append(group)
    
    return valid_groups


def _can_merge_vertically(group1: Dict, group2: Dict, threshold: float = 0.7) -> bool:
    """Check if two groups can be merged vertically (similar columns, different IDs)."""
    if not group1.get('merged_data') or not group2.get('merged_data'):
        return False
    
    headers1 = [str(h).lower().strip() for h in group1['merged_data'][0]]
    headers2 = [str(h).lower().strip() for h in group2['merged_data'][0]]
    
    if not headers1 or not headers2:
        return False
    
    common_headers = set(headers1) & set(headers2)
    total_headers = set(headers1) | set(headers2)
    
    if not total_headers:
        return False
    
    similarity = len(common_headers) / len(total_headers)
    
    if similarity >= threshold:
        if group1['tables'] and group2['tables']:
            return not _have_same_ids(group1['tables'][0], group2['tables'][0])
    
    return False


def _have_same_ids(table1: Dict, table2: Dict) -> bool:
    """Check if two tables have the same IDs in their first column."""
    data1 = table1.get('data', [])
    data2 = table2.get('data', [])
    
    if len(data1) < 2 or len(data2) < 2:
        return False
    
    rows1 = data1[1:]
    rows2 = data2[1:]
    
    if not rows1 or not rows2:
        return False
    
    ids1 = set()
    ids2 = set()
    
    for row in rows1:
        if row and len(row) > 0:
            first_cell = str(row[0]).strip()
            if first_cell and first_cell.lower() not in ['', 'none', 'null', 'n/a']:
                ids1.add(first_cell.lower())
    
    for row in rows2:
        if row and len(row) > 0:
            first_cell = str(row[0]).strip()
            if first_cell and first_cell.lower() not in ['', 'none', 'null', 'n/a']:
                ids2.add(first_cell.lower())
    
    if not ids1 or not ids2:
        return False
    
    common_ids = ids1 & ids2
    total_ids = ids1 | ids2
    
    if len(total_ids) > 0:
        overlap_ratio = len(common_ids) / len(total_ids)
        return overlap_ratio >= 0.5
    
    return False


def _merge_table_data(tables: List[Dict]) -> List[List]:
    """Merge data from multiple similar tables."""
    if not tables:
        return []
    
    if len(tables) == 1:
        return tables[0]['data']
    
    first_table = tables[0]
    first_col_identifiers = set()
    for row in first_table['data'][1:]:
        if row and len(row) > 0 and row[0]:
            first_col_identifiers.add(str(row[0]).strip())
    
    all_have_same_identifiers = True
    for table in tables[1:]:
        table_identifiers = set()
        for row in table['data'][1:]:
            if row and len(row) > 0 and row[0]:
                table_identifiers.add(str(row[0]).strip())
        
        if not table_identifiers.issubset(first_col_identifiers) and not first_col_identifiers.issubset(table_identifiers):
            all_have_same_identifiers = False
            break
    
    if all_have_same_identifiers:
        return _merge_tables_by_identifier(tables)
    else:
        merged_data = [first_table['headers']]
        for table in tables:
            data_rows = table['data'][1:] if len(table['data']) > 1 else []
            merged_data.extend(data_rows)
        return merged_data


def _merge_tables_by_identifier(tables: List[Dict]) -> List[List]:
    """Merge tables that have identical identifiers by combining their fields."""
    if not tables:
        return []
    
    identifier_rows = {}
    identifier_order = []
    all_headers = []
    header_set = set()
    
    for table in tables:
        for header in table['headers']:
            if header and str(header).strip():
                clean_header = str(header).strip()
                if clean_header not in header_set:
                    all_headers.append(clean_header)
                    header_set.add(clean_header)
    
    for table in tables:
        headers = [str(h).strip() for h in table['headers']]
        
        for row in table['data'][1:]:
            if not row or len(row) == 0 or not row[0]:
                continue
                
            identifier = str(row[0]).strip()
            
            if identifier not in identifier_rows:
                identifier_rows[identifier] = {}
                identifier_order.append(identifier)
            
            for i, header in enumerate(headers):
                if i < len(row) and row[i]:
                    clean_header = str(header).strip()
                    if clean_header:
                        identifier_rows[identifier][clean_header] = str(row[i]).strip()
    
    merged_data = [all_headers]
    
    for identifier in identifier_order:
        row_data = []
        for header in all_headers:
            value = identifier_rows[identifier].get(header, "")
            row_data.append(value)
        merged_data.append(row_data)
    
    return merged_data


def _print_extraction_summary(result: Dict, raw_bom_tables: List, raw_measurement_tables: List, image_sketches: List = None) -> None:
    """Print extraction summary."""
    fields = result['fields']
    found_count = sum(1 for v in fields.values() if v)
    
    # Count pages for BOM
    bom_pages = set(t['page'] for t in raw_bom_tables) if raw_bom_tables else set()
    measurement_pages = set(t['page'] for t in raw_measurement_tables) if raw_measurement_tables else set()
    image_pages = set(img['page'] for img in image_sketches) if image_sketches else set()
    
    print(f"\n📊 EXTRACTION SUMMARY:")
    print(f"   • Identifiers: {found_count}/10 found from Pages 1-2")
    print(f"   • BOM Tables: {len(raw_bom_tables)} tables from {len(bom_pages)} page(s) {sorted(bom_pages)}" if bom_pages else "   • BOM Tables: None found")
    print(f"   • Measurement Tables: {len(raw_measurement_tables)} tables from {len(measurement_pages)} page(s) {sorted(measurement_pages)}" if measurement_pages else "   • Measurement Tables: None found")
    print(f"   • Image Sketches: {len(image_sketches)} image(s) from {len(image_pages)} page(s) {sorted(image_pages)}" if image_sketches else "   • Image Sketches: None found")


def print_table_to_terminal(table_data, max_rows=10, max_col_width=20):
    """Print table data to terminal in a formatted way."""
    if not table_data or len(table_data) == 0:
        print("    (No data)")
        return
    
    # Get headers and data
    headers = table_data[0]
    rows = table_data[1:max_rows+1] if len(table_data) > 1 else []
    
    # Calculate column widths
    col_widths = []
    for i, header in enumerate(headers):
        # Start with header width
        width = min(len(str(header)), max_col_width)
        
        # Check data rows for max width
        for row in rows:
            if i < len(row):
                cell_width = min(len(str(row[i])), max_col_width)
                width = max(width, cell_width)
        
        col_widths.append(max(width, 8))  # Minimum width of 8
    
    # Print separator
    separator = "    +" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    print(separator)
    
    # Print headers
    header_row = "    |"
    for i, header in enumerate(headers):
        if i < len(col_widths):
            truncated = str(header)[:col_widths[i]]
            header_row += f" {truncated:<{col_widths[i]}} |"
    print(header_row)
    print(separator)
    
    # Print data rows
    for row in rows:
        data_row = "    |"
        for i, cell in enumerate(row):
            if i < len(col_widths):
                truncated = str(cell)[:col_widths[i]]
                data_row += f" {truncated:<{col_widths[i]}} |"
        print(data_row)
    
    print(separator)
    
    # Show total count if truncated
    total_data_rows = len(table_data) - 1
    if total_data_rows > max_rows:
        print(f"    ... showing {max_rows} of {total_data_rows} total rows")
    
    print()


def display_results(result: Dict) -> None:
    """Display extraction results in terminal (for CLI usage)."""
    if not result:
        print("No results to display")
        return
    
    print(f"\n" + "=" * 60)
    print("EXTRACTION RESULTS")
    print("=" * 60)
    
    # Product Identifiers
    print(f"\n🏷️ PRODUCT IDENTIFIERS")
    print("-" * 30)
    
    field_labels = {
        'brand': 'Brand',
        'description': 'Description',
        'style_number': 'Style Number',
        'season_year': 'Season Year',
        'composition': 'Composition',
        'material_finishes': 'Material Finishes',
        'yarn_gauge': 'Yarn Gauge',
        'model_id': 'Model Id',
        'product_line': 'Product Line',
        'date': 'Date'
    }
    
    fields = result['fields']
    found_count = sum(1 for v in fields.values() if v)
    
    for field_key, label in field_labels.items():
        value = fields.get(field_key, '')
        status = "✓" if value else "✗"
        print(f"  {status} {label:15}: {value}")
    
    print(f"\n📊 Found: {found_count}/10 identifiers")
    
    # Tables Summary
    bom_count = len(result.get('bom_tables', []))
    measurement_count = len(result.get('measurement_tables', []))
    
    print(f"\n📋 BOM Tables: {bom_count}")
    print(f"📏 Measurements: {measurement_count}")
    
    if bom_count > 0 or measurement_count > 0:
        print("\n💡 Use Streamlit app for detailed table viewing")


def main():
    """Main function for CLI usage."""
    print("PDF Garment Tech Pack Extractor")
    print("===============================")
    
    # Find PDF files
    pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDF files found in current directory.")
        print("💡 Try running the Streamlit app: streamlit run streamlit_app.py")
        return
    
    print(f"🎯 Found {len(pdf_files)} PDF file(s)")
    
    if len(pdf_files) == 1:
        # Single file processing
        pdf_file = pdf_files[0]
        print(f"📄 Processing: {pdf_file}")
        
        result = extract_from_pdf(pdf_file, verbose=True)
        if result:
            display_results(result)
            print(f"\n💡 For detailed table viewing, run: streamlit run streamlit_app.py")
        else:
            print("❌ Extraction failed")
    else:
        # Batch processing
        print(f"📄 Processing {len(pdf_files)} files in batch...")
        
        results = extract_from_multiple_pdfs(pdf_files, verbose=True)
        
        # Display batch summary
        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]
        
        print(f"\n📊 BATCH RESULTS SUMMARY:")
        print(f"   Total files: {len(pdf_files)}")
        print(f"   Successful: {len(successful)}")
        print(f"   Failed: {len(failed)}")
        
        if successful:
            print(f"\n✅ Successfully processed files:")
            for result in successful:
                found_count = len([v for v in result['fields'].values() if v])
                print(f"   • {result['filename']}: {found_count}/10 identifiers found")
        
        if failed:
            print(f"\n❌ Failed files:")
            for result in failed:
                print(f"   • {result['filename']}: {result['error']}")
        
        print(f"\n💡 For detailed table viewing and batch export, run: streamlit run streamlit_app.py")


# Backward compatibility - keep these functions for Streamlit
def export_to_csv(result: Dict, base_filename: str) -> None:
    """Export results to CSV files (simplified for compatibility)."""
    try:
        fields = result['fields']
        
        # Export identifiers
        identifier_data = {
            'Brand': fields.get('brand', ''),
            'Description': fields.get('description', ''),
            'Style_Number': fields.get('style_number', ''),
            'Season_Year': fields.get('season_year', ''),
            'Composition': fields.get('composition', ''),
            'Material_Finishes': fields.get('material_finishes', ''),
            'Yarn_Gauge': fields.get('yarn_gauge', ''),
            'Model_Id': fields.get('model_id', ''),
            'Product_Line': fields.get('product_line', ''),
            'Date': fields.get('date', ''),
        }
        
        df_identifiers = pd.DataFrame([identifier_data])
        df_identifiers.to_csv(f"{base_filename}_identifiers.csv", index=False)
        
        # Export BOM tables
        for i, group in enumerate(result.get('bom_tables', []), 1):
            if group.get('merged_data'):
                df_bom = pd.DataFrame(group['merged_data'][1:], columns=group['merged_data'][0])
                df_bom.to_csv(f"{base_filename}_bom_group_{i}.csv", index=False)
        
        # Export measurement tables  
        for i, group in enumerate(result.get('measurement_tables', []), 1):
            if group.get('merged_data'):
                df_measurements = pd.DataFrame(group['merged_data'][1:], columns=group['merged_data'][0])
                df_measurements.to_csv(f"{base_filename}_measurements_group_{i}.csv", index=False)
                
    except Exception as e:
        print(f"CSV export error: {e}")


if __name__ == "__main__":
    main()
