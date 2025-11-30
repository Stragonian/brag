/*
chunks.c : a small C program for chunking text files into specific paramaters, and
altering special characters for the embedding stage of the BRAG pipeline.

Created by Jerry B Nettrouer II <j2@inpito.org> https://www.inpito.org/projects.php

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>

 NOTE:  I came up with this crazy idea of using Bash & C & the ollama REST_API
 as a RAG as an alternative to the python RAGs to learn how RAG's handle data
 without all the python dependencies.

 This idea and project is still very experimental and I don't know how far I
 plan on going with it - it may just depend on how well it works.

 While still an amnionic idea and project, I'm attempting to accomplish much
 of the same work that I've seen done to create a RAG using python, but instead
 my hope is to avoid using python as much as possible, and try to keep the
 entire BRAG pipeline and project in Bash, C, and interacting with the Ollama
 REST_API, and keep the scripts and C programs about as simple as I possibly can.

 NOTICE: This project is in the development stage and I make no guarantees
 about its abilities or performance.

 REQUIRED: jq, curl, cudatoolkit, Ollama and LLMs that run on Ollama are required
 to use this BRAG pipeline.  Make sure to use the same embedding LLM that was
 used in the embedding stage of your BRAG as you plan to use within your query
 to process top-K entities of the pipeline, otherwise, your results might not
 be all that good.

gcc is required to compile this C file into an executable binary.

usage: ./chunks -i input.txt -o output.txt [-s]
-s show each line chunked on the terminal while chunking.

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>

// Change the number 128 below for smaller or larger chunk sizes.
#define TARGET_LEN 128

void escape_special_chars(char *dest, const char *src) {
    while (*src) {
        if (*src == '"') {
            *dest++ = '\\';
            *dest++ = '"';
        } else if (*src == '\\') {
            *dest++ = '\\';
            *dest++ = '\\';
        } else {
            *dest++ = *src;
        }
        src++;
    }
    *dest = '\0';
}

int find_split_point(const char *text, int target) {
    int len = strlen(text);
    if (len <= target) return len;

    int lower = target, upper = target;
    while (lower > 0 && !isspace((unsigned char)text[lower])) lower--;
    while (upper < len && !isspace((unsigned char)text[upper])) upper++;

    if ((target - lower) <= (upper - target) && lower > 0)
        return lower;
    else if (upper < len)
        return upper;
    else
        return len;
}

int main(int argc, char *argv[]) {
    char *input_file = NULL;
    char *output_file = NULL;
    int print_screen = 0;

    int opt;
    while ((opt = getopt(argc, argv, "i:o:s")) != -1) {
        switch (opt) {
            case 'i': input_file = optarg; break;
            case 'o': output_file = optarg; break;
            case 's': print_screen = 1; break;
            default:
                fprintf(stderr, "Usage: %s -i input.txt -o chunks.txt [-s]\n", argv[0]);
                return 1;
        }
    }

    if (!input_file || !output_file) {
        fprintf(stderr, "Usage: %s -i input.txt -o chunks.txt [-s]\n", argv[0]);
        return 1;
    }

    FILE *in = fopen(input_file, "r");
    if (!in) { perror("Error opening input file"); return 1; }

    fseek(in, 0, SEEK_END);
    long size = ftell(in);
    rewind(in);

    char *buffer = malloc(size + 1);
    if (!buffer) { fprintf(stderr, "Memory allocation failed\n"); fclose(in); return 1; }
    fread(buffer, 1, size, in);
    buffer[size] = '\0';
    fclose(in);

    FILE *out = fopen(output_file, "w");
    if (!out) { perror("Error opening output file"); free(buffer); return 1; }

    const char *p = buffer;
    char temp[1024];

    while (*p) {
       int split = find_split_point(p, TARGET_LEN);
       if (split <= 0) split = strlen(p);

       // Remove trailing whitespace
       while (split > 0 && isspace((unsigned char)p[split - 1])) split--;

       strncpy(temp, p, split);
       temp[split] = '\0';

       // Replace internal newlines/tabs with a space
       for (int i = 0; temp[i]; i++) {
          if (temp[i] == '\n' || temp[i] == '\r' || temp[i] == '\t')
             temp[i] = ' ';
       }

       // Trim leading spaces
       char *start = temp;
       while (*start && isspace((unsigned char)*start)) start++;

       // Skip empty lines
       if (*start == '\0') {
          p += split;
          while (isspace((unsigned char)*p)) p++;
          continue;
       }

       char escaped[2048];
       escape_special_chars(escaped, start);

       // Write to file
       fprintf(out, "%s\n", escaped);

       // Optional: print to screen
       if (print_screen) printf("%s\n", escaped);

       p += split;
       while (isspace((unsigned char)*p)) p++;
    }

    fclose(out);
    free(buffer);
    return 0;
}
