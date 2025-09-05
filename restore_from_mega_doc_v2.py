#!/usr/bin/env python3
"""
Script to restore all source files from the mega document.
"""

import os
import re
import sys

def restore_from_mega_doc(mega_doc_path):
    """Restore all source files from the mega document."""
    
    if not os.path.exists(mega_doc_path):
        print(f"Error: Mega document not found: {mega_doc_path}")
        return False
    
    print(f"Reading mega document: {mega_doc_path}")
    
    with open(mega_doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match file sections: ## 📄 **FILE X of Y**: path
    file_header_pattern = r'## 📄 \*\*FILE \d+ of \d+\*\*: (.*?)\n\n- \*\*Path\*\*: `(.*?)`\n\n- \*\*Size\*\*: \d+ lines\n\n```cpp\n(.*?)\n```'
    
    files_restored = 0
    
    for match in re.finditer(file_header_pattern, content, re.DOTALL):
        file_name = match.group(1)
        file_path = match.group(2)
        file_content = match.group(3)
        
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")
        
        # Write the file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
            print(f"Restored: {file_path}")
            files_restored += 1
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
    
    print(f"\nRestoration complete! {files_restored} files restored.")
    return True

if __name__ == "__main__":
    mega_doc = "THRESHOLD_DROPS_BUG_MEGA_DOCUMENT.md"
    restore_from_mega_doc(mega_doc)
