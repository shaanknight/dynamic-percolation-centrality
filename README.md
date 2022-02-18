# dynamic-percolation-centrality
Programs corresponding to the work "Efficient Parallel Algorithms for Dynamic Percolation Centrality"

In this work, we study the update of the percolation centrality measure in a dynamic graph. We present parallel algorithms to handle both the addition and deletion of edges to the graph and changes to the percolation value of vertices. We leverage structural properties of the graphs to improve performance.

This repo contains the implementations for handling both the update variants on CPU and GPU respectively.

## Dependencies

- python 3.7+
- CUDA  Version  11.3
- g++ version 7.3.0 compiler
- OpenMP version 4.5

## Run

The helper script `run.py` can be used to execute the programs on a Linux Environment. 
```
usage: run.py [-h] [-a ALGO] [-d DATASET] [-q QUERY ] [-o OUTFILE] [-g] [-r]

Compute percolation centrality on dynamic graphs with edge and vertex percolation value updates

optional arguments:
  -h, --help            show this help message and exit
  -a ALGO, --algorithm ALGO
                        dynperc - dynamic algorithm for vertex percolation update, statperc - static algorithm for vertex percolation update dynedge -
                        dynamic algorithm for edge update statedge - static algorithm for edge update
  -d DATASET, --dataset DATASET
                        The dataset to run on. Dataset must be present in
                        ./datasets subdirectory. Look at existing datasets for
                        the input format
  -o OUTFILE, --outfile OUTFILE
                        Output file name
  -q QUERY, --query QUERY
                        The query file containing the updates to be handled. File must be present in ./queries/percolation-update-queries or
                        ./queries/edge-update-queries subdirectory. Look at existing query files for the input and naming format
  -g, --gpu             Run experiment of GPU (default: multicore CPU)
  -r, --recompile       Recompile executables
```

For example, to run an edge update on the dataset `PGPgiantcompo.in` using Dynamic algorithm on CPU,
```
python run.py --algorithm dynedge --dataset PGPgiantcompo.in --query PGPgiantcompo_queries
```

To run a vertex percolation value update on the dataset `PGPgiantcompo.in` using Dynamic algorithm on GPU and store the output in `my_outfile.txt`,
```
python run.py --algorithm dynperc --dataset PGPgiantcompo.in --query PGPgiantcompo_queries --gpu -o my_outfile.txt 
```
