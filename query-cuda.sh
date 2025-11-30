#!/bin/bash
# query-cuda.sh : A bash script for processing the top-K of a .json file
# and answering a query question based off those top-K .json entries.
#
# By Jerry B Nettrouer II <j2@inpito.org> https://www.inpito.org/projects.php
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
# as a RAG as an alternative to the python RAGs to learn how RAG's handle data
# without all the python dependencies.
#
# This idea and project is still very experimental and I don't know how far I
# plan on going with it - it may just depend on how well it works.
#
# While still an amnionic idea and project, I'm attempting to accomplish much
# of the same work that I've seen done to create a RAG using python, but instead
# my hope is to avoid using python as much as possible, and try to keep the
# entire BRAG pipeline and project in Bash, C, and interacting with the Ollama
# REST_API, and keep the scripts and C programs about as simple as I possibly can.
#
# NOTICE: This project is in the development stage and I make no guarantees
# about its abilities or performance.
#
# REQUIRED: jq, curl, cudatoolkit, Ollama and LLMs that run on Ollama are required
# to use this BRAG pipeline.  Make sure to use the same embedding LLM that was
# used in the embedding stage of your BRAG as you plan to use within your query
# to process top-K entities of the pipeline, otherwise, your results might not
# be all that good.
#
# This script is currently set to use the qwen3-embedding:8b for processing
# top-K and qwen3:8b for handling the query.  If you would like to use
# different LLMs then replace qwen3-embedding:8b in QUERY_EMBED and
# replace qwen3:8b in the RESPONSE.  Otherwise, you will need to pull both
# the qwen3-embedding:8b and qwen3:8b to use this script while running
# ollama.
#
# WARNING: If your vector embedding was done using qwen3-embedding:8b then use
# that LLM to process your top-K.  Likewise, if you used mxbai-embed-large
# to create your vector embedding .json file then use mxbai-embed-large to
# calculate top-K.
#
# Make sure ollama is up and running then execute ...
# Usage: ./query-cuda.sh embed-file.json "Your query here"

EMBED=$1
QUERY="Answer **only** using the text provided. $2"
TOP_K=10

# Get query embedding
QUERY_EMBED=$(curl -s http://localhost:11434/api/embeddings -d "{
  \"model\": \"qwen3-embedding:8b\",
  \"prompt\": \"$QUERY\"
}" | jq -c '.embedding')

# Iterate over each chunk and compute cosine similarity
jq -c '.[]' "$EMBED" | while read -r ROW; do
    ID=$(echo "$ROW" | jq -r '.id')
    TEXT=$(echo "$ROW" | jq -r '.text')
    EMB=$(echo "$ROW" | jq -c '.embedding')

    # Write vectors to files for cosine computation
    echo "$EMB" | jq -r '.[]' > embed.tmp
    echo "$QUERY_EMBED" | jq -r '.[]' > query.tmp

    # Compute cosine similarity via C/CUDA program
    #SCORE=$(./cosine -e embed.tmp -q query.tmp)        # CPU mode
    SCORE=$(./cosine -e embed.tmp -q query.tmp -n)    # GPU mode

    echo -e "$SCORE\t$TEXT"
done | sort -gr | head -n "$TOP_K" > topk.txt

# Feed top-k context into ollama model
CONTEXT=$(awk -F'\t' '{print $2}' topk.txt | tr '\n' ' ')

echo
echo "=== Retrieved top-k Context ==="
awk -F'\t' '{print $2}' topk.txt
echo "==============================="
echo

RESPONSE=$(curl -s http://localhost:11434/api/generate -d "{
  \"model\": \"qwen3:8b\",
  \"prompt\": \"$CONTEXT\n\nQuestion: $QUERY\nAnswer:\"
}")

echo "$RESPONSE" | jq -r '.response' > response.txt
awk 'BEGIN{RS=""; ORS="\n\n"} {gsub(/\n/, ""); print}' response.txt > response_clean.txt
cat response_clean.txt

# Clean up
rm -f embed.tmp query.tmp topk.txt
