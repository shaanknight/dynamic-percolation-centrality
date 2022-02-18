#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>
#include<vector>
#include<set>
#include<time.h>
#include<iostream>
#include<vector>
#include<stack>
#include<set>
#include<queue>
#include<list>
#include<iomanip>
#include<algorithm>
#include<chrono>
#include<fstream>
#include<omp.h>
using namespace std;

// compile : nvcc <file_name>.cu -arch=sm_70 -std=c++11 -Xcompiler -fopenmp -O3 -o computePC-static-percUpdate

using namespace std;

#define NUM_THREADS 32
#define NUM_BLOCKS 1024

typedef struct
{
	int child;
	int parent;
} node;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void brandes(int V, int E, int *dColumn, int *dRow, int *Distance, int *Queue,
double *Paths, double *dDelta, node *Parents, double *dCentrality, int *crr, double *pc)
{
	__shared__ int arr[2];
	int *QLen = arr, *parIndex = arr+1;
	
	int rootIndex = blockIdx.x + 1;
	int *Q = Queue + (blockIdx.x)*(V+1);
	double *dPaths = Paths + (blockIdx.x)*(V+1);
	double *delta = dDelta + (blockIdx.x)*(V+1);
	int *done = crr + (blockIdx.x)*(V+1);
	int *dDistance = Distance + (blockIdx.x)*(V+1);
	node *dParent = Parents + (blockIdx.x)*(E+1);

	while(rootIndex <= V)
	{
		for(int i=threadIdx.x; i<=V; i+=NUM_THREADS) 
		{
			dPaths[i] = 0;
			dDistance[i] = -1;
			done[i] = 0;
			delta[i] = 0.0;
		}

		if(threadIdx.x==0)
		{
			*QLen = *parIndex = 1;
			int root = rootIndex;
			Q[0] = root;
			dPaths[root] = 1.0;
			dDistance[root] = 0;
			dParent[0] = {root, 0};
		}
		__syncthreads();

		int oldQLen = 0;
		while(oldQLen < *QLen)
		{
			int id = threadIdx.x;
			int source = Q[oldQLen++];
			int degree = dRow[source+1] - dRow[source];

			while(id < degree)
			{
				int neighbour = dColumn[dRow[source]+id];
				if(dDistance[neighbour] == -1)
				{
					dDistance[neighbour] = dDistance[source]+1;
					Q[atomicAdd(QLen, 1)] = neighbour;
				}
				if(dDistance[neighbour] == dDistance[source]+1)
				{
					dPaths[neighbour] += dPaths[source];
					dParent[atomicAdd(parIndex, 1)] = {neighbour,source};
				}
				id += NUM_THREADS;
			}
			__syncthreads();
		}

		if(threadIdx.x==0)
		{
			arr[0] = dDistance[Q[*QLen-1]];
			*parIndex -= 1;
		}
		__syncthreads();

		int id = *parIndex - threadIdx.x, *reach = arr;
		
		while(*reach > 0)
		{
			if(id > 0 && dDistance[dParent[id].child] == *reach)
			{
				node n = dParent[id];

				atomicAdd(&delta[n.parent], (dPaths[n.parent]/dPaths[n.child])*(max(0.0,pc[rootIndex]-pc[n.child])+(delta[n.child])));

				if(atomicExch(&done[n.child], 1) == 0)
				{
					atomicAdd(&dCentrality[n.child], delta[n.child]);
				}

				bool flag = dDistance[dParent[id-1].child] == *reach-1;
				if(threadIdx.x==NUM_THREADS-1 || flag)
				{
					*parIndex = id-1;
					*reach -= (flag);
				}
			}
			__syncthreads();
			id = *parIndex - threadIdx.x;
		}
		rootIndex += NUM_BLOCKS;
	}
}

vector<double> x, updated_x;
vector<vector<int> > graph;
vector<int> query_node;

pair<int,vector<double> > compute_constants()
{
	int N = (int)x.size()-1;
	vector<pair<double,int> > perc(N+1);
	vector<double> contrib(N+1,0.0);
	for(int i=1;i<=N;++i)
		perc[i] = {x[i],i};
	sort(perc.begin(),perc.end());
	long double carry = 0,sum_x = 0;
	for(int i=1;i<=N;++i)
	{
		contrib[perc[i].second] = (long double)(i-1)*perc[i].first-carry;
		carry += perc[i].first;
		sum_x += contrib[perc[i].second];
	}
	carry = 0;
	for(int i=N;i>=1;i--)
	{
		contrib[perc[i].second] += carry-(long double)(N-i)*perc[i].first;
		carry += perc[i].first;
	}
	return make_pair(sum_x,contrib);
}

int main( int argc, char **argv )
{
	ios::sync_with_stdio(false);
	cin.tie(0);
	cout.tie(0);

   string input = argv[1];
	string queries = argv[2];
	string output = argv[3];
	ifstream fin(input);
	ofstream fout(output);

   int V,E;

   fin >> V >> E;

	graph.resize(V+1);
	for(int i=0; i<E; ++i)
	{
		int u, v;
		fin >> u >> v;
		if(u == v) continue;
		graph[u].push_back(v);
		graph[v].push_back(u);
	}
	x.push_back(0);
	for(int i=0;i<V;++i)
	{
		double prc = (1.0/V)*(rand()%V);
		x.push_back(prc);
	}

	auto t1 = std::chrono::high_resolution_clock::now();

	auto res = compute_constants();
	double sum_x = res.first;
	vector<double> contrib = res.second;

	int *hColumn = new int[2*E];
	int *hRow	 = new int[V+2];
	double *perc  = new double[V+2];
	double *updated_perc = new double[V+2];

	for(int i=1;i<=V;++i)
		perc[i] = x[i];
	perc[0] = perc[V+1] = 1.0;

	for(int index=0, i=1; i<=V; ++i) 
	{
		for(int j=0;j<(int)graph[i].size();++j)
		{
			int n = graph[i][j]; 
			hColumn[index++] = n;
		}
	}

	long count = 0;
	for(int i=0; i<=V;)
	{
		for(int j=0;j<(int)graph.size();++j)
		{
			vector<int> v = graph[i];
			hRow[i++] = count;
			count += v.size();
		}
	}
	hRow[V+1] = count;

	double *delta, *Paths, *dCentrality, *pc;
	node *Parents;
	int *dColumn, *dRow, *Distance, *Queue, *crr;

	cudaMalloc((void**)&dRow,    		sizeof(int)*(V+2));
	cudaMalloc((void**)&dCentrality,	sizeof(double)*(V+2));
	cudaMalloc((void**)&pc,				sizeof(double)*(V+2));
	cudaMalloc((void**)&dColumn, 		sizeof(int)*(2*E));
	cudaMalloc((void**)&crr,    		sizeof(int)*(V+2)*NUM_BLOCKS);
	cudaMalloc((void**)&Queue,    	sizeof(int)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&Distance,		sizeof(int)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&Paths,			sizeof(double)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&delta,			sizeof(double)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&Parents,		sizeof(node)*(E+1)*NUM_BLOCKS);

	cudaMemcpy(dRow, hRow, sizeof(int)*(V+2),cudaMemcpyHostToDevice);
	cudaMemcpy(dColumn, hColumn, sizeof(int)*(2*E), cudaMemcpyHostToDevice);
	cudaMemcpy(pc, perc, sizeof(double)*(V+2),cudaMemcpyHostToDevice);
	gpuErrchk( cudaPeekAtLastError() );

	brandes <<<NUM_BLOCKS, NUM_THREADS, 32>>> (V, E, dColumn, dRow, Distance, Queue, Paths, delta,
				Parents, dCentrality, crr, pc);

	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );
	cudaDeviceSynchronize();

	double *Centrality = new double[V+1];
	cudaMemcpy(Centrality, dCentrality, sizeof(double)*(V+1), cudaMemcpyDeviceToHost);

	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
	cerr << "Initial Static Computation time : " << duration << " mu.s." <<endl;
	duration = 0;
	auto duration_actual = duration; 

	ifstream qin(queries);
	int batch_size;
	while(qin >> batch_size)
	{
		updated_x = x;
		query_node.clear();
		int node;
		double val;
		for(int i=1;i<=batch_size;++i)
		{
			qin >> node >> val;
			if(x[node] != val)
			{
				query_node.push_back(node);
				updated_x[node] = val;
			}
		}
		x = updated_x;
		for(int i=1;i<=V;++i)
			perc[i] = x[i];
		perc[0] = perc[V+1] = 1.0;

		auto t3 = std::chrono::high_resolution_clock::now();
		cudaMemcpy(pc, perc, sizeof(double)*(V+2),cudaMemcpyHostToDevice);
		double *fCentrality;
		cudaMalloc((void**)&fCentrality,	sizeof(double)*(V+2));
		gpuErrchk( cudaPeekAtLastError() );

		brandes <<<NUM_BLOCKS, NUM_THREADS, 32>>> (V, E, dColumn, dRow, Distance, Queue, Paths, delta,
					Parents, fCentrality, crr, pc);

		cudaDeviceSynchronize();
		gpuErrchk( cudaPeekAtLastError() );
		cudaDeviceSynchronize();

		cudaMemcpy(Centrality, fCentrality, sizeof(double)*(V+1), cudaMemcpyDeviceToHost);

		auto t4 = std::chrono::high_resolution_clock::now();
		duration_actual += std::chrono::duration_cast<std::chrono::microseconds>( t4 - t3 ).count();

		for(int i=1;i<=V;++i)
			fout << Centrality[i]/(sum_x-contrib[i]) << " ";
		fout << "\n";
	}
	cerr << "Total time for updates : " << duration_actual << " mu.s." << endl;

	double *fCentrality;
	cudaMalloc((void**)&fCentrality,	sizeof(double)*(V+2));
	gpuErrchk( cudaPeekAtLastError() );

	brandes <<<NUM_BLOCKS, NUM_THREADS, 32>>> (V, E, dColumn, dRow, Distance, Queue, Paths, delta,
				Parents, fCentrality, crr, pc);

	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );
	cudaDeviceSynchronize();

	double *corrCentrality = new double[V+1];
	cudaMemcpy(corrCentrality, fCentrality, sizeof(double)*(V+1), cudaMemcpyDeviceToHost);

	double max_diff = 0;
	for(int i=1;i<=V;++i)
		max_diff = max(max_diff,abs(corrCentrality[i]-Centrality[i])/(sum_x-contrib[i]));
	cerr << "Max difference in PC point-wise : " << max_diff << "\n";

	return 0;
}