#!/usr/bin/env python3
"""
Manual restoration script for mega document.
"""

import os
import re

def restore_files():
    """Restore files by finding each file section manually."""
    
    with open('THRESHOLD_DROPS_BUG_MEGA_DOCUMENT.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by file headers
    file_sections = re.split(r'## 📄 \*\*FILE \d+ of \d+\*\*:', content)
    
    files_restored = 0
    
    for i, section in enumerate(file_sections[1:], 1):  # Skip first empty section
        try:
            # Extract file path
            path_match = re.search(r'- \*\*Path\*\*: `(.*?)`', section)
            if not path_match:
                continue
                
            file_path = path_match.group(1)
            
            # Extract content between ```text and ```
            content_match = re.search(r'```text\n(.*?)\n```', section, re.DOTALL)
            if not content_match:
                continue
                
            file_content = content_match.group(1)
            
            # Create directory if needed
            dir_path = os.path.dirname(file_path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                print(f"Created directory: {dir_path}")
            
            # Write file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
            
            print(f"Restored: {file_path}")
            files_restored += 1
            
        except Exception as e:
            print(f"Error processing file {i}: {e}")
    
    print(f"\nRestoration complete! {files_restored} files restored.")
    return files_restored

if __name__ == "__main__":
    restore_files()
