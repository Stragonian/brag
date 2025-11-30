#!/bin/bash
# embed.sh : A bash script for processing the chuncked data files into
# vector files for use by the query-cuda.sh file.
#
# Created by Jerry B Nettrouer II <j2@inpito.org> https://www.inpito.org/projects.php
#
#   This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>
#
# NOTE:  I came up with this crazy idea of using Bash & C & the ollama REST_API
# or "BRAG" as an alternative to the python RAGs to learn how RAG basics work.
#
# This idea and project is still very experimental and I don't know how far I
# plan on going with it - it may just depend on how well it works.
#
# While still an amnionic idea and project, I'm attempting to accomplish much
# of the same work that I've seen done to create a RAG using python, but
# instead my hope is to avoid using python, and try to keep the entire BRAG
# pipeline and project in Bash, C, and interacting with the Ollama REST_API
# as much as possible, and to keep the scripts and C programs about as simple
# as I can.
#
# NOTICE: This project is in the development stage and I make no guarantees
# about its abilities or performance.
#
# REQUIRED: jq, curl, Ollama and LLMs that run on Ollama are required to use
# this BRAG pipeline.  Make sure to use the same embedding LLM that was used in
# the embedding stage of your BRAG as you plan to use within your query to
# process top_K entities of the pipeline, otherwise, your results might not be
# all that good.
#
# WARNING: If your vector embedding was done using qwen3-embedding:8b then use that
# LLM to process your top_K in the query-cuda.sh file.  Likewise, if you used
# the mxbai-embed-large:latest to create your vector embedding .json file then use
# mxbai-embed-large:latest to calculate top_K in the query-cuda.sh file.
#
# Make sure ollama is up and running then execute ...
# usage: ./embed.sh chunk.txt embed.json

# variable for input file.
INPUT=$1

# name of the output vector json file.
OUTPUT=$2

# Ollama curl command string for processing
API_URL="http://localhost:11434/api/embeddings"

# module to use for creating vector json file.
MODEL="qwen3-embedding:8b"

# Start the embedded vector file
echo "[" > "$OUTPUT"

# loop through the input file by line
while IFS= read -r CHUNK; do

    # Skip empty lines
    [[ -z "$CHUNK" ]] && continue

    # Prepare JSON payload with correct key "prompt"
    PAYLOAD=$( jq -n --arg model "$MODEL" --arg prompt "$CHUNK" '{model: $model, prompt: $prompt}')

    # Call Ollama embedding API
    RESPONSE=$(curl -s -H "Content-Type: application/json" -d "$PAYLOAD" "$API_URL")
    EMBEDDING=$(echo "$RESPONSE" | jq '.embedding')

    # Append to output JSON as multiple lines
    # jq -n --arg text "$CHUNK" --argjson emb "$EMBEDDING" '{text: $text, embedding: $emb}' >> "$OUTPUT"

    # Append to output JSON as a single line
    DATA=$( jq -nc --arg text "$CHUNK" --argjson emb "$EMBEDDING" '{text: $text, embedding: $emb}' )

    # Append to output JSON as a single line and add a comma at the end of the vector entry
    echo "$DATA," >> "$OUTPUT"

    # Free the data for the next line
    unset DATA

done < $INPUT

# Remove final trailing comma
sed -i '${s/,[[:space:]]*$//}' "$OUTPUT"

# Append the closing of the vector json file.
echo "]" >> "$OUTPUT"

# Announce the script is finished and saved to output file
echo "All embeddings saved to $OUTPUT"
