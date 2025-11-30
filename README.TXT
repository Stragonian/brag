Please support the Indiana Non-Profit Information Technology Organization, Inc. Find more information at https://www.inpito.org/index.php

Greetings, I'm Jerry B Nettrouer II, the creator of this "BRAG" ( Bash + RAG ) pipeline project.  I thought I'd use this README.TXT file as a sample file that could be used throughout the process to demonstrate each step in this RAG ( Retrieval-Augmented Generation ) written to be used within Linux, Bourne Again Shell and helped along with a few simple C programs.

I came up with this crazy idea of using Bash & C & the ollama REST_API as a RAG alternative to the python RAGs to learn how RAG's handle data without all the python dependencies.  This idea and project is still very experimental and I don't know how far I plan on going with it - it may just depend on how well it works.

While still an amnionic idea and project, I'm attempting to accomplish much of the same work that I've seen done to create a RAG using python, but instead my hope is to avoid using python as much as possible, and try to keep the entire BRAG pipeline and project in Bash, C, and interacting with the Ollama REST_API, and keep the scripts and C programs about as simple as I possibly can.

NOTICE: This project is very much in the development and possible crazy idea stage, so I make no guarantees about its abilities or its performance or its life cycle.  I don't know how long I will experiment with this idea.  I mostly decided to create a BRAG pipeline to learn how RAG basics work to create a data pipeline.

REQUIRED: jq, curl, cudatoolkit, Ollama and LLMs that run on Ollama are required to use this BRAG pipeline.  Make sure to use the same embedding LLM that was used in the embedding stage of your BRAG as you plan to use within your query to process top_K entities of the pipeline, otherwise, your results might not be all that good.

For example, if you use qwen3-embedding:8b in the embedding stage of the BRAG pipeline, then it's a good idea to use qwen3-embedding:8b to process the top_K within the query portion of the processing within the pipeline.

Pipeline Walk Through:
I'll start by compiling chunks a small helper program I created written in C.

gcc -o chunks chunks.c

Next I'll use this README.TXT and turning it in a chunked file. Usage of chunks is ./chunks -i input.txt -o chunks.txt [-s]  optional -s shows the process of chunking to the display screen while still outputing the chunked data to the file.

./chunks -i README.TXT -o readchunks.txt -s

This will result in a text file of chunked text, breaking each line at around 128 characters long.  Chunks also preps the file for being properly processed by the embed.sh.

NOTE: Make sure ollama serve is up and running, and that qwen3-embedding:8b has been installed.  Or, you can edit the embed.sh script with your own choice of embedding LLM.

Next, we execute the embed.sh file to turn the readchunks.txt into an embedded vector .json file, so it can be used by LLMs for processing.

./embed.sh readchunks.txt readme-qwen3-embedding-8b.json

I often use the embedding LLM's name as a major part of the .json file's name to keep which embedding LLM that I used completely obvious.

Once the embed.sh script has finished generating vectors within the readme-qwen3-embedding-8b.json file, it will announce that the script is finished processing and saved the vectors to the file readme-qwen3-embedding-8b.json.

Now that the readme-qwen3-embedding-8b.json is done, it's almost time to execute the query-cuda.sh to process the top-K entries from the readme-qwen3-embedding-8b.json file as needed and perform the query upon that data.

However, the cosine helper program is required to handle top-K entry computations.  I used the cudatoolkit (10.2.89) nvcc compiler from slackbuilds.org to compile the cosine.cu to create the cosine helper program.

nvcc -o cosine cosine.cu

NOTE: Make sure ollama serve is up and running, and that qwen3-embedding:8b and qwen3:8b have been installed.  Or, you can edit the query-cuda.sh script with your own choice of embedding and generation LLM.

Once cosine is compiled the query-cuda.sh bash script is ready to answer questions about your text documents ...

./query-cuda.sh readme-qwen3-embedding-8b.json "Who is the creator of the BRAG pipeline?"

Give the script a little time to process, and it should hopefully provide the correct answer, otherwise it's back to the drawing board.
