#!/bin/bash

# Define the output file for the metadata.sha1 fields
OUTPUT_FILE="s2orc_pdfs_v2.txt"

# Clear the output file if it already exists
> "$OUTPUT_FILE"

# Create a temporary directory for partial outputs
temp_output_dir=$(mktemp -d)

# Ensure the temporary directory is cleaned up on exit or error
trap 'rm -rf "$temp_output_dir"' EXIT

# Export the temporary output directory variable for use in xargs
export temp_output_dir

echo "temp dir $temp_output_dir"

# Find all .gz files recursively from the current directory
find 'split=train' -type f -name "*.gz" | \
    xargs -P 30 -I{} bash -c '
        gz_file="$1"
        partial_output="$temp_output_dir/$(basename "$gz_file").txt"

        # Stream uncompressed data directly into jq and format the output
        gunzip -c "$gz_file" | jq -r '"'"'
            select(.metadata.sha1 != null) |
            "s3://ai2-s2-pdfs/" + (.metadata.sha1[:4]) + "/" + (.metadata.sha1[4:]) + ".pdf"
        '"'"' >> "$partial_output"
    ' _ {}

# Concatenate all partial outputs into the final output file
cat "$temp_output_dir"/*.txt >> "$OUTPUT_FILE"

echo "All metadata.sha1 fields have been extracted to $OUTPUT_FILE."
