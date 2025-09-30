"""
Fuzzy PDF Field Extractor using PDFplumber

This script uses PDFplumber to extract text from PDF files and then performs
fuzzy keyword matching to find specific garment fields. It handles cases where
field names and data may not be separated by colons and can appear in various formats.

Target Fields:
- Style name (keywords: name)
- Style number (keywords: Style No.)
- Brand (keywords: brand)
- Season/year (keywords: season, year)
- Designer/developer name (keywords: designer, developer name, created by, author)
- Date (keywords: date)
"""

import pdfplumber
import re
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher


class FuzzyPDFExtractor:
    """Extract fields from PDF using fuzzy keyword matching."""
    
    def __init__(self):
        """Initialize the fuzzy extractor with field definitions."""
        
        # Define field keywords and their variations
        self.field_keywords = {
            'style_name': {
                'keywords': ['name', 'style name', 'product name', 'item name'],
                'weight': 1.0
            },
            'style_number': {
                'keywords': ['style no', 'style number', 'style code', 'sku', 'product code', 'item number'],
                'weight': 1.0
            },
            'brand': {
                'keywords': ['brand', 'company', 'manufacturer', 'vendor'],
                'weight': 1.0
            },
            'season_year': {
                'keywords': ['season', 'year', 'season/year', 'collection'],
                'weight': 1.0
            },
            'designer_developer': {
                'keywords': ['designer', 'developer name', 'created by', 'author', 'tech designer', 'product developer'],
                'weight': 1.0
            },
            'date': {
                'keywords': ['date', 'created', 'updated', 'revision date'],
                'weight': 1.0
            }
        }
        
        # Common separators and patterns
        self.separators = [':', '-', '=', '|', '\t', '  ', ' ']
        
        # Date patterns for validation
        self.date_patterns = [
            r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}',  # MM/DD/YYYY
            r'\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}',    # YYYY/MM/DD
            r'\d{6}',                                  # MMDDYY
            r'\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{2,4}',
            r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2},?\s+\d{2,4}'
        ]
    
    def similarity(self, a: str, b: str) -> float:
        """Calculate similarity between two strings."""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict:
        """Extract text and tables from PDF using PDFplumber."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_text = ""
                all_lines = []
                tables_data = []
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text
                    text = page.extract_text()
                    if text:
                        all_text += f"\n--- Page {page_num} ---\n{text}"
                        
                        # Split into lines for analysis
                        lines = text.split('\n')
                        for line in lines:
                            if line.strip():
                                all_lines.append({
                                    'page': page_num,
                                    'text': line.strip(),
                                    'words': line.strip().split()
                                })
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for table_num, table in enumerate(tables):
                        tables_data.append({
                            'page': page_num,
                            'table': table_num + 1,
                            'data': table
                        })
                
                return {
                    'full_text': all_text,
                    'lines': all_lines,
                    'tables': tables_data,
                    'total_pages': len(pdf.pages)
                }
                
        except Exception as e:
            print(f"Error extracting from PDF: {e}")
            return {
                'full_text': '',
                'lines': [],
                'tables': [],
                'total_pages': 0,
                'error': str(e)
            }
    
    def find_fuzzy_matches(self, text: str, keywords: List[str], threshold: float = 0.6) -> List[Tuple[str, float, int]]:
        """Find fuzzy matches for keywords in text."""
        matches = []
        text_lower = text.lower()
        words = text_lower.split()
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Exact match
            if keyword_lower in text_lower:
                start_pos = text_lower.find(keyword_lower)
                matches.append((keyword, 1.0, start_pos))
                continue
            
            # Fuzzy match with individual words
            for i, word in enumerate(words):
                similarity = self.similarity(word, keyword_lower)
                if similarity >= threshold:
                    start_pos = text_lower.find(word)
                    matches.append((keyword, similarity, start_pos))
            
            # Fuzzy match with word combinations
            keyword_words = keyword_lower.split()
            if len(keyword_words) > 1:
                for i in range(len(words) - len(keyword_words) + 1):
                    word_combo = ' '.join(words[i:i + len(keyword_words)])
                    similarity = self.similarity(word_combo, keyword_lower)
                    if similarity >= threshold:
                        start_pos = text_lower.find(word_combo)
                        matches.append((keyword, similarity, start_pos))
        
        return matches
    
    def extract_value_near_keyword(self, line: str, keyword_pos: int, keyword: str) -> Optional[str]:
        """Extract value near a found keyword."""
        # Split line into parts
        before_keyword = line[:keyword_pos].strip()
        after_keyword = line[keyword_pos + len(keyword):].strip()
        
        # Try different extraction strategies
        strategies = [
            self._extract_after_separator,
            self._extract_next_words,
            self._extract_before_keyword,
            self._extract_table_cell
        ]
        
        for strategy in strategies:
            value = strategy(before_keyword, after_keyword, line)
            if value and self._is_valid_value(value):
                return value
        
        return None
    
    def _extract_after_separator(self, before: str, after: str, full_line: str) -> Optional[str]:
        """Extract value after common separators."""
        for sep in self.separators:
            if after.startswith(sep):
                value = after[len(sep):].strip()
                if value:
                    # Take first meaningful part
                    parts = re.split(r'\s{2,}|[,;]', value)
                    return parts[0].strip() if parts else None
        return None
    
    def _extract_next_words(self, before: str, after: str, full_line: str) -> Optional[str]:
        """Extract next few words as value."""
        if after:
            words = after.split()
            if words:
                # Take 1-4 words depending on context
                if len(words) >= 3:
                    return ' '.join(words[:3])
                else:
                    return ' '.join(words)
        return None
    
    def _extract_before_keyword(self, before: str, after: str, full_line: str) -> Optional[str]:
        """Extract value that appears before the keyword."""
        if before:
            words = before.split()
            if words:
                # Take last few words
                if len(words) >= 2:
                    return ' '.join(words[-2:])
                else:
                    return words[-1]
        return None
    
    def _extract_table_cell(self, before: str, after: str, full_line: str) -> Optional[str]:
        """Extract value from table-like structure."""
        # Look for patterns like "Key Value" or "Value Key"
        line_words = full_line.split()
        if len(line_words) >= 2:
            # Try different combinations
            combinations = [
                ' '.join(line_words[-2:]),  # Last two words
                ' '.join(line_words[:2]),   # First two words
                line_words[-1],             # Last word
                line_words[0]               # First word
            ]
            
            for combo in combinations:
                if self._is_valid_value(combo):
                    return combo
        
        return None
    
    def _is_valid_value(self, value: str) -> bool:
        """Check if extracted value is valid."""
        if not value or len(value.strip()) < 2:
            return False
        
        value = value.strip().lower()
        
        # Invalid values
        invalid_values = {
            'n/a', 'na', 'none', 'null', 'tbd', 'tbc', '...', 'xxx', 
            'field', 'value', 'data', 'info', 'details'
        }
        
        if value in invalid_values:
            return False
        
        # Too short or just punctuation
        if len(value) < 2 or value.isspace() or all(c in ':-_.,;' for c in value):
            return False
        
        return True
    
    def extract_from_tables(self, tables_data: List[Dict]) -> Dict[str, str]:
        """Extract fields from table structures."""
        extracted = {}
        
        for table_info in tables_data:
            table = table_info['data']
            if not table:
                continue
            
            # Process each row in the table
            for row in table:
                if not row or len(row) < 2:
                    continue
                
                # Convert row to text for analysis
                row_text = ' '.join([str(cell) if cell else '' for cell in row])
                
                # Check each field
                for field_name, field_info in self.field_keywords.items():
                    if field_name in extracted:
                        continue
                    
                    # Find matches for this field
                    matches = self.find_fuzzy_matches(row_text, field_info['keywords'])
                    
                    if matches:
                        # Get the best match
                        best_match = max(matches, key=lambda x: x[1])
                        keyword, similarity, pos = best_match
                        
                        # Extract value
                        value = self.extract_value_near_keyword(row_text, pos, keyword)
                        if value:
                            extracted[field_name] = value
        
        return extracted
    
    def extract_from_lines(self, lines: List[Dict]) -> Dict[str, str]:
        """Extract fields from text lines."""
        extracted = {}
        
        for line_info in lines:
            line_text = line_info['text']
            
            # Check each field
            for field_name, field_info in self.field_keywords.items():
                if field_name in extracted:
                    continue
                
                # Find matches for this field
                matches = self.find_fuzzy_matches(line_text, field_info['keywords'])
                
                if matches:
                    # Get the best match
                    best_match = max(matches, key=lambda x: x[1])
                    keyword, similarity, pos = best_match
                    
                    # Extract value
                    value = self.extract_value_near_keyword(line_text, pos, keyword)
                    if value:
                        extracted[field_name] = value
        
        return extracted
    
    def post_process_values(self, extracted: Dict[str, str]) -> Dict[str, str]:
        """Post-process extracted values for better quality."""
        processed = {}
        
        for field_name, value in extracted.items():
            if not value:
                processed[field_name] = ""
                continue
            
            # Clean up value
            cleaned_value = value.strip()
            
            # Remove common prefixes/suffixes
            cleaned_value = re.sub(r'^[:\-_\s]+', '', cleaned_value)
            cleaned_value = re.sub(r'[:\-_\s]+$', '', cleaned_value)
            
            # Field-specific processing
            if field_name == 'date':
                cleaned_value = self._process_date_value(cleaned_value)
            elif field_name == 'style_number':
                cleaned_value = self._process_style_number(cleaned_value)
            elif field_name == 'season_year':
                cleaned_value = self._process_season_year(cleaned_value)
            
            processed[field_name] = cleaned_value
        
        return processed
    
    def _process_date_value(self, value: str) -> str:
        """Process date values."""
        # Look for date patterns in the value
        for pattern in self.date_patterns:
            match = re.search(pattern, value, re.IGNORECASE)
            if match:
                return match.group(0)
        return value
    
    def _process_style_number(self, value: str) -> str:
        """Process style number values."""
        # Extract alphanumeric codes
        match = re.search(r'[A-Z0-9\-_]{3,}', value.upper())
        if match:
            return match.group(0)
        return value
    
    def _process_season_year(self, value: str) -> str:
        """Process season/year values."""
        # Look for year patterns
        year_match = re.search(r'20\d{2}', value)
        season_match = re.search(r'(spring|summer|fall|autumn|winter|ss|fw|aw)', value, re.IGNORECASE)
        
        if year_match and season_match:
            return f"{season_match.group(0)} {year_match.group(0)}"
        elif year_match:
            return year_match.group(0)
        elif season_match:
            return season_match.group(0)
        
        return value
    
    def process_pdf(self, pdf_path: str) -> Tuple[Dict, Dict]:
        """Main processing function."""
        print(f"🔍 Processing PDF: {os.path.basename(pdf_path)}")
        
        # Extract text and tables
        pdf_data = self.extract_text_from_pdf(pdf_path)
        
        if pdf_data.get('error'):
            return {
                'file_name': os.path.basename(pdf_path),
                'error': pdf_data['error'],
                'extraction_method': 'fuzzy_pdfplumber'
            }, pdf_data
        
        print(f"  📄 Pages: {pdf_data['total_pages']}")
        print(f"  📝 Lines: {len(pdf_data['lines'])}")
        print(f"  📊 Tables: {len(pdf_data['tables'])}")
        
        # Extract from tables first (higher priority)
        table_extracted = self.extract_from_tables(pdf_data['tables'])
        print(f"  🔍 Table extraction: {len(table_extracted)} fields")
        
        # Extract from lines for missing fields
        line_extracted = self.extract_from_lines(pdf_data['lines'])
        print(f"  🔍 Line extraction: {len(line_extracted)} fields")
        
        # Combine results (tables take priority)
        combined = {}
        for field_name in self.field_keywords.keys():
            if table_extracted.get(field_name):
                combined[field_name] = table_extracted[field_name]
            elif line_extracted.get(field_name):
                combined[field_name] = line_extracted[field_name]
            else:
                combined[field_name] = "not found"
        
        # Post-process values
        final_extracted = self.post_process_values(combined)
        
        # Add metadata
        result = {
            **final_extracted,
            'file_name': os.path.basename(pdf_path),
            'extraction_method': 'fuzzy_pdfplumber',
            'pages_processed': pdf_data['total_pages'],
            'tables_found': len(pdf_data['tables']),
            'lines_processed': len(pdf_data['lines']),
            'extraction_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Show results
        found_fields = [field for field in self.field_keywords.keys() if result.get(field)]
        print(f"  🎯 Found {len(found_fields)}/6 fields: {', '.join(found_fields)}")
        
        return result, pdf_data
    
    def display_results(self, data: Dict):
        """Display extraction results."""
        print("\n" + "=" * 70)
        print(f"FUZZY EXTRACTION RESULTS - {data.get('file_name', 'Unknown')}")
        print("=" * 70)
        
        if data.get('error'):
            print(f"❌ Error: {data['error']}")
            return
        
        field_labels = {
            'style_name': 'Style Name',
            'style_number': 'Style Number',
            'brand': 'Brand',
            'season_year': 'Season/Year',
            'designer_developer': 'Designer/Developer',
            'date': 'Date'
        }
        
        print("\n🔑 EXTRACTED FIELDS:")
        print("-" * 40)
        found_count = 0
        for field_name in self.field_keywords.keys():
            value = data.get(field_name, "not found")
            label = field_labels.get(field_name, field_name.replace('_', ' ').title())
            status = "✓" if value and value != "not found" else "✗"
            print(f"  {status} {label:18}: {value}")
            if value and value != "not found":
                found_count += 1
        
        print(f"\n📊 SUMMARY:")
        print("-" * 20)
        print(f"  Success Rate: {found_count}/6 ({found_count/6*100:.1f}%)")
        print(f"  Pages: {data.get('pages_processed', 0)}")
        print(f"  Tables: {data.get('tables_found', 0)}")
        print(f"  Lines: {data.get('lines_processed', 0)}")
        print(f"  Method: {data.get('extraction_method', 'unknown')}")
    
    def export_to_csv(self, data: Dict, output_path: str):
        """Export results to CSV."""
        try:
            # Clean data for export
            export_data = {k: v for k, v in data.items() if not k.startswith('_')}
            df = pd.DataFrame([export_data])
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"📊 Results exported to: {output_path}")
        except Exception as e:
            print(f"❌ Export error: {e}")
    
    def export_extracted_text(self, pdf_data: Dict, output_path: str):
        """Export extracted text to a file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("EXTRACTED TEXT FROM PDF\n")
                f.write("=" * 50 + "\n\n")
                
                # Write full text
                f.write("FULL TEXT:\n")
                f.write("-" * 20 + "\n")
                f.write(pdf_data.get('full_text', ''))
                f.write("\n\n")
                
                # Write table information
                if pdf_data.get('tables'):
                    f.write("TABLES FOUND:\n")
                    f.write("-" * 20 + "\n")
                    for i, table_info in enumerate(pdf_data['tables'], 1):
                        f.write(f"\nTable {i} (Page {table_info['page']}):\n")
                        for row in table_info['data']:
                            if row:
                                row_text = " | ".join([str(cell) if cell else '' for cell in row])
                                f.write(f"  {row_text}\n")
                    f.write("\n")
                
                # Write line information
                f.write("TEXT LINES:\n")
                f.write("-" * 20 + "\n")
                for line_info in pdf_data.get('lines', []):
                    f.write(f"Page {line_info['page']}: {line_info['text']}\n")
            
            print(f"📝 Extracted text saved to: {output_path}")
        except Exception as e:
            print(f"❌ Text export error: {e}")


def main():
    """Main function."""
    print("Fuzzy PDF Field Extractor")
    print("=========================")
    print("Uses fuzzy keyword matching to extract fields from PDF tables and text")
    print()
    
    # Look for PDF files
    pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in current directory.")
        return
    
    pdf_file = pdf_files[0]
    print(f"📄 Found PDF file: {pdf_file}")
    
    try:
        # Initialize extractor
        extractor = FuzzyPDFExtractor()
        
        # Process PDF
        result, pdf_data = extractor.process_pdf(pdf_file)
        
        # Display results
        extractor.display_results(result)
        
        # Export results
        if not result.get('error'):
            base_name = os.path.splitext(pdf_file)[0]
            
            # Export extracted text
            text_file = f"{base_name}_extracted_text.txt"
            extractor.export_extracted_text(pdf_data, text_file)
            
            # Export results to CSV
            csv_file = f"{base_name}_fuzzy_results.csv"
            extractor.export_to_csv(result, csv_file)
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
