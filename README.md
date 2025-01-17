# How Robust is Your Parser? An Adversarial Evaluation of Parsers
Shantanu Thorat

## About
This repository contains the project code. 

## Environment

Create a Python 3.10.15 environment and then install the requirements:
```shell
pip -r requirements.txt
```

## Shell Scripts
The `.sh` files are primarily for constituency parsing. For dependency parsing results, you have to change the file paths manually in the source code. 

To run your own experiments, modify the shell scripts for the respective parser.

## Data

`json/` contains sentence data with gold dependency labels and constituency parses in JSON format. 
`gold_standard` contains CSV data for dependency relations and .txt data for constituency parsing. 

Note that the `plots/` folder has to be recreated to run some data visualization scripts.

As an example, we provide a subset of the data from the GUM corpus. 