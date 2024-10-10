#!/bin/bash

# Loop through each line in requirements.txt
while IFS= read -r requirement; do
    # Check if the requirement specifies a file
    if [[ "$requirement" == *"@"* ]]; then
        # Try to install the package from the file location
        pip install "$requirement" || {
            # If it fails, extract the package name and install the latest version
            package_name="${requirement%%@*}"
            echo "Failed to install $requirement, installing latest version of $package_name..."
            pip install "${package_name// /}"  # Install the latest version
        }
    else
        # If it does not specify a file, install as is (exact version)
        pip install "$requirement"
    fi
done < requirements.txt