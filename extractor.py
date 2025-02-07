import os
import re

# Directory to search for Python files
project_dir = '.'

# Regex patterns for 'import' and 'from' statements
import_pattern = re.compile(r'^\s*import (\S+)')
from_pattern = re.compile(r'^\s*from (\S+)')

# Set to store unique libraries
libraries = set()

# Walk through the project directory
for root, dirs, files in os.walk(project_dir):
    for file in files:
        if file.endswith('.py'):
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Check for import and from statements
                    match = import_pattern.match(line) or from_pattern.match(line)
                    if match:
                        libraries.add(match.group(1).split('.')[0])

# Write libraries to requirements.txt
with open('requirements.txt', 'w') as req_file:
    for library in sorted(libraries):
        req_file.write(f"{library}\n")

print("Extracted libraries saved to requirements.txt")