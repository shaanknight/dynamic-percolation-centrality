from subprocess import run
import argparse
from sys import exit
from os.path import isfile, exists
from os import mkdir

parser = argparse.ArgumentParser(description='Compute percolation centrality on dynamic graphs with edge and vertex percolation value updates')
parser.add_argument('-a', '--algorithm', dest='algo',
                    default=['dynperc'], nargs=1,
                    help='dynperc - dynamic algorithm for vertex percolation update, \
                    	  statperc - static algorithm for vertex percolation update \
                    	  dynedge - dynamic algorithm for edge update \
                    	  statedge - static algorithm for edge update')

parser.add_argument('-d', '--dataset', dest='dataset',
                    default=['none'], nargs=1,
                    help='The dataset to run on. Dataset must be present in ./datasets subdirectory.\
                    	  Look at existing datasets for the input format')

parser.add_argument('-q', '--query', dest='query',
                    default=['none'], nargs=1,
                    help='The query file containing the updates to be handled.\
                     	  File must be present in ./queries/percolation-update-queries or ./queries/edge-update-queries subdirectory.\
                    	  Look at existing query files for the input and naming format')

parser.add_argument('-o', '--outfile', dest='outfile',
                    default=['out.txt'], nargs=1,
                    help='Output file name')

parser.add_argument('-g', '--gpu', dest='gpuflag', action='store_true',
                    help='Run experiment of GPU (default: multicore CPU)')

parser.add_argument('-r', '--recompile', dest='compileflag', action='store_true',
                    help='Recompile executables')



args = parser.parse_args()
algo = args.algo[0]
dataset = args.dataset[0]
query = args.query[0]
outfile = args.outfile[0]	
gpuflag = args.gpuflag
compileflag = args.compileflag


if algo not in ['dynperc', 'statperc', 'dynedge', 'statedge']:
	exit('Error: Invalid algorithm.')

if not isfile("./datasets/"+dataset):
	exit('Error: Dataset not found.')

if (algo == "dynperc" or algo == "statperc") and not isfile("./queries/percolation-update-queries/"+query):
	exit('Error: Invalid Query Specification.')

if (algo == "dynedge" or algo == "statedge") and not isfile("./queries/edge-update-queries/"+query):
	exit('Error: Invalid Query Specification.')

command_map = {
	('dynperc',False) : 'g++ -O3 -fopenmp -static-libstdc++ ./src/CPU/dynamic-percolation-update.cpp -o ./exec/CPU/dpu',
	('statperc',False) : 'g++ -O3 -fopenmp -static-libstdc++ ./src/CPU/static-percolation-update.cpp -o ./exec/CPU/spu',
	('dynedge',False) : 'g++ -O3 -fopenmp -static-libstdc++ ./src/CPU/dynamic-edge-update.cpp -o ./exec/CPU/deu',
	('statedge',False) : 'g++ -O3 -fopenmp -static-libstdc++ ./src/CPU/static-edge-update.cpp -o ./exec/CPU/seu',
	('dynperc',True) : 'nvcc ./src/GPU/dynamic-percolation-update.cu -arch=sm_70 -std=c++11 -Xcompiler -fopenmp -O3 -o ./exec/GPU/dpu',
	('statperc',True) : 'nvcc ./src/GPU/static-percolation-update.cu -arch=sm_70 -std=c++11 -Xcompiler -fopenmp -O3 -o ./exec/GPU/spu',
	('dynedge',True) : 'nvcc ./src/GPU/dynamic-edge-update.cu -arch=sm_70 -std=c++11 -Xcompiler -fopenmp -O3 -o ./exec/GPU/deu',
	('statedge',True) : 'nvcc ./src/GPU/static-edge-update.cu -arch=sm_70 -std=c++11 -Xcompiler -fopenmp -O3 -o ./exec/GPU/seu'
}

exec_map = {
	('dynperc',False) : './exec/CPU/dpu',
	('statperc',False) : './exec/CPU/spu',
	('dynedge',False) : './exec/CPU/deu',
	('statedge',False) : './exec/CPU/seu',
	('dynperc',True) : './exec/GPU/dpu',
	('statperc',True) : './exec/GPU/spu',
	('dynedge',True) : './exec/GPU/deu',
	('statedge',True) : './exec/GPU/seu'
}

if not exists('./exec'):
	mkdir('./exec')

if not exists('./exec/CPU'):
	mkdir('./exec/CPU')

if not exists('./exec/GPU'):
	mkdir('./exec/GPU')

if not exists('./output'):
	mkdir('./output')


if not isfile(exec_map[(algo,gpuflag)]) or compileflag:
	print("Didn't find executable or recompile flag set.")
	print("Compiling ...")
	run(command_map[(algo,gpuflag)], shell=True, check=True)
	print("Finished Compiling")
else:
	print("Found existing executables. Skipping compilation.")

print("Running Algorithm {} on dataset {} on {}".format(algo, dataset, ['CPU','GPU'][int(gpuflag)]))
if algo == "dynperc" or algo == "statperc":
	run(exec_map[(algo,gpuflag)]+ " ./datasets/{} ./queries/percolation-update-queries/{} ./output/{}".format(dataset, query, outfile), shell=True, check=True)
if algo == "dynedge" or algo == "statedge":
	run(exec_map[(algo,gpuflag)]+ " ./datasets/{} ./queries/edge-update-queries/{} ./output/{}".format(dataset, query, outfile), shell=True, check=True)
print("Successfully completed execution. Output can be found at ./output/{}".format(outfile))
