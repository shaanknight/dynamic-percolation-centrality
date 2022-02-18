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

// compile : nvcc <file_name>.cu -arch=sm_70 -std=c++11 -Xcompiler -fopenmp -O3 -o computePC-dynamic-percUpdate

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
double *Paths, double *dDelta, node *Parents, double *dCentrality, int *crr, double *perc_state, 
int *reach_vec, int *starters, int *org)
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
		for(int i=threadIdx.x; i<=V; i+=NUM_THREADS) dPaths[i] = 0;
		for(int i=threadIdx.x; i<=V; i+=NUM_THREADS) dDistance[i] = -1;
		for(int i=threadIdx.x; i<=V; i+=NUM_THREADS) done[i] = 0;	
		for(int i=threadIdx.x; i<=V; i+=NUM_THREADS) delta[i] = 0;

		if(threadIdx.x==0)
		{
			*QLen = *parIndex = 1;
			int root = rootIndex;
			Q[0] = root;
			dPaths[root] = 1.0f;
			dDistance[root] = 0;
			dParent[0] = {root, 0};
		}
		__syncthreads();

		int oldQLen = 0;
		while(oldQLen < *QLen)
		{
			int	source = Q[oldQLen++];
			int degree = dRow[source+1] - dRow[source];
			if(source != rootIndex)
			{
				for(int i = starters[source]+threadIdx.x;i < starters[source+1];i+=NUM_THREADS)
				{
					if(reach_vec[i] == org[source])
						continue;
					for(int j = starters[rootIndex];j < starters[rootIndex+1];++j)
					{
						if(reach_vec[i] > reach_vec[j])
							atomicAdd(&delta[source], abs(perc_state[reach_vec[j]]-perc_state[reach_vec[i]]));
						// atomicAdd(&delta[source], max(0.0,perc_state[reach_vec[j]]-perc_state[reach_vec[i]]));
					}
				}
				// __syncthreads();
				// for(int i = starters[rootIndex]+threadIdx.x;i < starters[rootIndex+1];i+=NUM_THREADS)
				// 	atomicAdd(&delta[source], -1*max(0.0,perc_state[reach_vec[i]]-perc_state[source]));
			}
			__syncthreads();

			int id = threadIdx.x;
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
				double add = 0.0;
				for(int j = starters[rootIndex];j < starters[rootIndex+1];++j)
				{
					if(org[n.child] > reach_vec[j])
						add += abs(perc_state[reach_vec[j]]-perc_state[n.child]);
					// add += max(0.0,perc_state[reach_vec[j]]-perc_state[n.child]);
				}
				
				atomicAdd(&delta[n.parent], (dPaths[n.parent]/dPaths[n.child])*(add+(delta[n.child])));
				
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


__global__ void update_brandes(int V, int E, int *dColumn, int *dRow, int *Distance, int *Queue,
double *Paths, double *dDelta, node *Parents, double *dCentrality, int *crr, double *pc, 
int *reach_vec, int *starters, int *reach_buffer, int *org,
int query_size, int batch_size, int *dQueries, double *updated_pc, double *subDelta)
{
	__shared__ int arr[3];
	int *QLen = arr, *parIndex = arr+1, *affected = arr+2;
	
	int rootPointer = blockIdx.x;
	int *Q = Queue + (blockIdx.x)*(V+1);
	double *dPaths = Paths + (blockIdx.x)*(V+1);
	double *new_delta = dDelta + (blockIdx.x)*(V+1);
	double *old_delta = subDelta + (blockIdx.x)*(V+1);
	int *done = crr + (blockIdx.x)*(V+1);
	int *dDistance = Distance + (blockIdx.x)*(V+1);
	int *reach_affected = reach_buffer + (blockIdx.x)*(V+1);
	node *dParent = Parents + (blockIdx.x)*(E+1);

	for(int i=threadIdx.x; i<=V; i+=NUM_THREADS) 
	{
		dPaths[i] = 0;
		dDistance[i] = -1;
		done[i] = 1;
		old_delta[i] = 0.0;
		new_delta[i] = 0.0;
	}

	while(rootPointer < batch_size)
	{
		int rootIndex = dQueries[rootPointer];
		if(threadIdx.x==0)
		{
			*affected = 0;
		}
		__syncthreads();
		
		for(int i = threadIdx.x;i < query_size;i+=NUM_THREADS)
		{
			int lo = starters[rootIndex];
			int hi = starters[rootIndex+1]-1; 
			while (lo <= hi) {
			    int mid = lo + (hi - lo) / 2;
			    if(dQueries[i] == reach_vec[mid])
			    {
			    	reach_affected[atomicAdd(affected,1)] = dQueries[i];
			    	break;
			    }
			    if (dQueries[i] < reach_vec[mid]) {
			        hi = mid - 1;
			    }
			    else {
			        lo = mid + 1;
			    }
			}
		}
		
		__syncthreads();
		if(*affected == 0)
		{
			rootPointer += NUM_BLOCKS;
			continue;
		}

		if(threadIdx.x==0)
		{
			*QLen = *parIndex = 1;
			Q[0] = rootIndex;
			dPaths[rootIndex] = 1.0;
			dDistance[rootIndex] = 0;
			done[rootIndex] = 0;
			dParent[0] = {rootIndex, 0};
		}
		__syncthreads();

		int oldQLen = 0;
		while(oldQLen < *QLen)
		{
			int source = Q[oldQLen++];
			int degree = dRow[source+1] - dRow[source];
			old_delta[source] = 0.0;
			new_delta[source] = 0.0;

			if(source != rootIndex)
			{
				if(starters[source+1]-starters[source] >= *affected)
				{
					for(int i = starters[source]+threadIdx.x;i < starters[source+1];i+=NUM_THREADS)
					{
						if(reach_vec[i] == org[source])
							continue;
						bool is_non_query_node = (pc[reach_vec[i]] == updated_pc[reach_vec[i]]);
						for(int j = 0;j < *affected;++j)
						{
							if(is_non_query_node || reach_vec[i] > reach_affected[j])
							{
								atomicAdd(&new_delta[source], abs(updated_pc[reach_vec[i]]-updated_pc[reach_affected[j]]));
								atomicAdd(&old_delta[source], abs(pc[reach_vec[i]]-pc[reach_affected[j]]));
							}
						}
					}
				}
				else
				{
					for(int j = threadIdx.x;j < *affected;j+=NUM_THREADS)
					{
						for(int i = starters[source];i < starters[source+1];i++)
						{
							if(reach_vec[i] == org[source])
								continue;
							bool is_non_query_node = (pc[reach_vec[i]] == updated_pc[reach_vec[i]]);
							if(is_non_query_node || reach_vec[i] > reach_affected[j])
							{
								atomicAdd(&new_delta[source], abs(updated_pc[reach_vec[i]]-updated_pc[reach_affected[j]]));
								atomicAdd(&old_delta[source], abs(pc[reach_vec[i]]-pc[reach_affected[j]]));
							}
						}
					}
				}

				__syncthreads();
			}
			__syncthreads();

			int id = threadIdx.x;
			while(id < degree)
			{
				int neighbour = dColumn[dRow[source]+id];
				if(done[neighbour] == 1)
				{
					done[neighbour] = 0;
					dDistance[neighbour] = dDistance[source]+1;
					dPaths[neighbour] = 0.0;
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
				bool is_non_query_node = (pc[n.child] == updated_pc[n.child]);
				double new_add = 0.0, old_add = 0.0;
				for(int j = 0;j < *affected;++j)
				{
					if(is_non_query_node || org[n.child] > reach_affected[j])
					{
						new_add += abs(updated_pc[reach_affected[j]]-updated_pc[n.child]);
						old_add += abs(pc[reach_affected[j]]-pc[n.child]);
					}
				}
				atomicAdd(&new_delta[n.parent], (dPaths[n.parent]/dPaths[n.child])*(new_add + new_delta[n.child]));
				atomicAdd(&old_delta[n.parent], (dPaths[n.parent]/dPaths[n.child])*(old_add + old_delta[n.child]));

				if(atomicExch(&done[n.child], 1) == 0)
					atomicAdd(&dCentrality[n.child], new_delta[n.child]-old_delta[n.child]);

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
		done[rootIndex] = 1;
		rootPointer += NUM_BLOCKS;
	}
}

int n,m;
int vertices;
int timer;
vector<double> x,updated_x,pc,global_pc;
vector<vector<int> > reach;
vector<vector<int> > reachv;
vector<vector<int> > copies;
vector<pair<int,int> > st;
vector<int> vis,vis1,low,entry, cur_comp;
vector<vector<int> > g, tmp_g;
vector<bool> in_otherbccs;
vector<pair<double,int> > perc;
vector<int> rep;
vector<int> query_node;

void dfs(int u, int par){
	timer++;
	entry[u] = timer;
	vis[u] = 2;
	low[u] = entry[u];
	vector<int> children = {u};
	
	for(int v:g[u])
	{
		if(vis[v]< 2)
		{
			st.push_back({u,v});
			dfs(v, u);
			low[u] = min(low[u], low[v]);

			if(low[v] >= entry[u])
			{
				++vertices;
				rep.push_back(u);
				copies[u].push_back(vertices);
				x[vertices] = x[u];
				vector<int> unique_vertices;
				while(st.back() != make_pair(u,v)){
					int p = st.back().first;
					int q = st.back().second;
					st.pop_back();
					if(p == u)
						p = vertices;
					if(q == u)
						q = vertices;
					unique_vertices.push_back(p);
					unique_vertices.push_back(q);
					tmp_g[p].push_back(q);
					tmp_g[q].push_back(p);
				}
				st.pop_back();
				tmp_g[vertices].push_back(v);
				tmp_g[v].push_back(vertices);
				unique_vertices.push_back(v);
				unique_vertices.push_back(vertices);
				sort(unique_vertices.begin(), unique_vertices.end());
				unique_vertices.erase(unique(unique_vertices.begin(), unique_vertices.end()),unique_vertices.end());

				for(int uv:unique_vertices){
					if(uv != vertices)
					{
						for(auto v:reachv[uv])
						{
							children.push_back(v);
							in_otherbccs[v] = 1;
						}
					}
				}

				for(int i:cur_comp)
				{
					if(!in_otherbccs[i])
						reach[vertices].push_back(i);
					else
						in_otherbccs[i] = 0;
				};
				sort(reach[vertices].begin(),reach[vertices].end());
			}
		}
		else if(v != par && entry[v] < entry[u])
		{
			st.push_back({u,v});
			low[u] = min(low[u], entry[v]);
		}
	}
	reachv[u] = children;
	reach[u].clear();
	sort(reachv[u].begin(), reachv[u].end());
	reachv[u].erase(unique(reachv[u].begin(), reachv[u].end()),reachv[u].end());
	for(int i:reachv[u]) 
		reach[u].push_back(i);
	
	vis[u] = 3;
}

void prelim_dfs(int u){
	vis[u] = 1;
	cur_comp.push_back(u);
	for(int v:g[u]){
		if(!vis[v]){
			prelim_dfs(v);
		}
	}
}

pair<int,vector<double> > compute_constants()
{
	int N = (int)x.size()-1;
	vector<pair<double,int> > perc(N+1);
	vector<double> contrib(N+1,0.0);
	for(int i=1;i<=N;++i)
		perc[i] = {x[i],i};
	sort(perc.begin(),perc.end());
	double carry = 0,sum_x = 0;
	for(int i=1;i<=N;++i)
	{
		contrib[perc[i].second] = (double)(i-1)*perc[i].first-carry;
		carry += perc[i].first;
		sum_x += contrib[perc[i].second];
	}
	carry = 0;
	for(int i=N;i>=1;i--)
	{
		contrib[perc[i].second] += carry-(double)(N-i)*perc[i].first;
		carry += perc[i].first;
	}
	return make_pair(sum_x,contrib);
}

int main(int argc, char **argv)
{
	ios::sync_with_stdio(false);
	cin.tie(0);
	cout.tie(0);

	string input = argv[1];
	string queries = argv[2];

	ifstream fin(input);

	fin >> n >> m;
	vertices = n;
	for(int i=0;i<=n;++i)
		rep.push_back(i);
	g.resize(n+1);
	copies.resize(n+1);
	for(int i=1;i<=n;++i)
		copies[i].push_back(i);
	tmp_g.resize(2*m+1);
	x.resize(n+1);
	reach.resize(2*m+1);
	reachv.resize(n+1);
	vis.resize(n+1);
	low.resize(n+1);
	entry.resize(n+1);
	in_otherbccs.resize(n+1,0);
	perc.resize(n+1);
	global_pc.resize(n+1,0);
	pc.resize(n+1,0);
	
	timer = 0;
	x[0] = 0;
	for(int i=1;i<=n;i++){
		x[i] = (1.0/(double)(n))*(rand()%n);
	}
	for(int i=0;i<m;i++){
		int u,v;
		fin >> u >> v;
		if(u != v)
		{
			g[u].push_back(v);
			g[v].push_back(u);
		}
	}
	
   auto t1 = std::chrono::high_resolution_clock::now();

   auto res = compute_constants();
	double sum_x = res.first;
	vector<double> contrib = res.second;
	x.resize(2*m+1);

	for(int i=1;i<=n;i++){
		if(!vis[i]){
			cur_comp.clear();
			prelim_dfs(i);
			for(auto v:cur_comp)
			{
				if((int)g[v].size() != 1)
				{
					dfs(v,0);
					break;
				}
			}
		}
	}

	tmp_g.resize(vertices+1);
	
	int V, E = 0;
	V = vertices;
	int cnt_reach_vec = 0;
	for(int i=1;i<=V;++i)
	{
		cnt_reach_vec += (int)(reach[i].size());
		E += (int)(tmp_g[i].size());
	}
	E = E/2;

	int *hColumn = new int[2*E];
	int *hRow	 = new int[V+2];
	int *cOrg 	 = new int[V+2];
	double *perc  = new double[V+2];
	double *updated_perc = new double[V+2];

	for(int i=1;i<=V;++i)
		perc[i] = x[i];
	perc[0] = perc[V+1] = 1.0;
	for(int i=1;i<=V;++i)
		cOrg[i] = rep[i];
	cOrg[0] = cOrg[V+1] = 0;

	for(int index=0, i=1; i<=V; ++i) 
	{
		for(int j=0;j<(int)tmp_g[i].size();++j)
		{
			int n = tmp_g[i][j]; 
			hColumn[index++] = n;
		}
	}
	
	long count = 0;
	for(int i=0; i<=V;)
	{
		for(int j=0;j<(int)tmp_g.size();++j)
		{
			vector<int> v = tmp_g[i];
			hRow[i++] = count;
			count += v.size();
		}
	}
	
	hRow[V+1] = count;
	int *intm_reach = new int[cnt_reach_vec];
	int *pointers = new int[V+2];

	int index = 0;
	for(int i=1;i<=V;++i)
	{
		pointers[i] = index;
		for(int j=0;j<(int)reach[i].size();++j)
		{
			intm_reach[index] = reach[i][j];
			index++;
		}
	}
	pointers[V+1] = index;
	
	double *delta, *Paths, *dCentrality, *perc_state;
	node *Parents;
	int *dColumn, *dRow, *Distance, *Queue, *crr, *starters, *reach_vec, *reach_buffer, *org;

	cudaMalloc((void**)&dRow,    		 sizeof(int)*(V+2));
	cudaMalloc((void**)&dCentrality,	 sizeof(double)*(V+2));
	cudaMalloc((void**)&perc_state,		 sizeof(double)*(V+2));
	cudaMalloc((void**)&dColumn, 		 sizeof(int)*(2*E));
	cudaMalloc((void**)&crr,    		 sizeof(int)*(V+2)*NUM_BLOCKS);
	cudaMalloc((void**)&Queue,    		 sizeof(int)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&Distance,		 sizeof(int)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&reach_buffer,	 sizeof(int)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&Paths,			 sizeof(double)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&delta,			 sizeof(double)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&Parents,		 sizeof(node)*(E+1)*NUM_BLOCKS);
	cudaMalloc((void**)&reach_vec,		 sizeof(int)*(cnt_reach_vec));
	cudaMalloc((void**)&starters, 		 sizeof(int)*(V+2));
	cudaMalloc((void**)&org, 		 	 sizeof(int)*(V+2));

	cudaMemcpy(dRow, hRow, sizeof(int)*(V+2),cudaMemcpyHostToDevice);
	cudaMemcpy(dColumn, hColumn, sizeof(int)*(2*E), cudaMemcpyHostToDevice);
	cudaMemcpy(perc_state, perc, sizeof(double)*(V+2),cudaMemcpyHostToDevice);
	cudaMemcpy(reach_vec, intm_reach, sizeof(int)*(cnt_reach_vec),cudaMemcpyHostToDevice);
	cudaMemcpy(starters, pointers, sizeof(int)*(V+2),cudaMemcpyHostToDevice);
	cudaMemcpy(org, cOrg, sizeof(int)*(V+2),cudaMemcpyHostToDevice);
	gpuErrchk( cudaPeekAtLastError() );

	brandes <<<NUM_BLOCKS, NUM_THREADS, 32>>> (V, E, dColumn, dRow, Distance, Queue, Paths, delta,
				Parents, dCentrality, crr, perc_state, reach_vec, starters, org);

	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );
	cudaDeviceSynchronize();
	double *Centrality = new double[V+2];
	cudaMemcpy(Centrality, dCentrality, sizeof(double)*(V+2), cudaMemcpyDeviceToHost);

	for(int i=1;i<=V;++i)
		global_pc[rep[i]] += Centrality[i];
	for(int i=1;i<=n;++i)
		global_pc[i] /= (sum_x - contrib[i]);
	
	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
	cerr << "Initial Static Computation time : " << duration << " mu.s." << endl;

	duration = 0;
	auto duration_dynamic = duration;
	
	double *updated_pc,*subDelta;
	int *dQueries;
	int query_nodes[V];

	cudaMalloc((void**)&updated_pc,	sizeof(double)*(V+2));
	cudaMalloc((void**)&subDelta, sizeof(double)*(V+1)*NUM_BLOCKS);

	ifstream qin(queries);
	int batch_size;
	while(qin >> batch_size)
	{
		updated_x = x;
		int node;
		double val;
		int query_size = 0;
		for(int i=1;i<=batch_size;++i)
		{
			qin >> node >> val;
			if(x[node] != val)
			{
				for(auto repr:copies[node])
				{
					updated_x[repr] = val;
				}
				query_nodes[query_size++] = node;
			}
		}
		for(int i=1;i<=V;++i)
			updated_perc[i] = updated_x[i];
		updated_perc[0] = updated_perc[V+1] = 1.0;
		batch_size = query_size;

		auto t3 = std::chrono::high_resolution_clock::now();

		for(int i=1;i<=V;++i)
		{
			if((int)(reach[i].size()) > 1)
			{
				if(rep[i] == i && x[i] != updated_x[i])
					continue;
				if(query_size <= 20)
					query_nodes[batch_size++] = i;
				else
				{
					for(auto v:reach[i])
					{
						if(x[v] != updated_x[v])
						{
							query_nodes[batch_size++] = i;
							break;
						}
					}
				}
			}
		}
		
		cudaMalloc((void**)&dQueries,sizeof(int)*(batch_size));

		cudaMemcpy(dQueries, query_nodes, sizeof(int)*(batch_size), cudaMemcpyHostToDevice);
		cudaMemcpy(updated_pc, updated_perc, sizeof(double)*(V+2),cudaMemcpyHostToDevice);
		cudaMemcpy(perc_state, perc, sizeof(double)*(V+2),cudaMemcpyHostToDevice);
		gpuErrchk( cudaPeekAtLastError() );

		update_brandes <<<NUM_BLOCKS, NUM_THREADS, 32>>> (V, E, dColumn, dRow, Distance, Queue, Paths, delta,
					Parents, dCentrality, crr, perc_state, reach_vec, starters, reach_buffer, org, query_size, batch_size, dQueries, updated_pc, subDelta);

		cudaDeviceSynchronize();
		gpuErrchk( cudaPeekAtLastError() );
		cudaDeviceSynchronize();

		auto t4 = std::chrono::high_resolution_clock::now();
		duration_dynamic += std::chrono::duration_cast<std::chrono::microseconds>( t4 - t3 ).count();

		x = updated_x;
		for(int i=1;i<=V;++i)
			perc[i] = x[i];
		perc[0] = perc[V+1] = 1.0;
	}
	cerr << "Total time for updates : " << duration_dynamic << " mu.s." << endl;

	cudaMemcpy(Centrality, dCentrality, sizeof(double)*(V+1), cudaMemcpyDeviceToHost);

	double *fCentrality;
	cudaMalloc((void**)&fCentrality,	sizeof(double)*(V+2));
	gpuErrchk( cudaPeekAtLastError() );

	brandes <<<NUM_BLOCKS, NUM_THREADS, 32>>> (V, E, dColumn, dRow, Distance, Queue, Paths, delta,
				Parents, fCentrality, crr, updated_pc, reach_vec, starters, org);

	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );
	cudaDeviceSynchronize();

	double *corrCentrality = new double[V+1];
	cudaMemcpy(corrCentrality, fCentrality, sizeof(double)*(V+1), cudaMemcpyDeviceToHost);

	fill(global_pc.begin(),global_pc.end(),0.0);
	fill(pc.begin(),pc.end(),0.0);

	for(int i=1;i<=V;++i)
		global_pc[rep[i]] += Centrality[i];
	for(int i=1;i<=n;++i)
		global_pc[i] /= (sum_x - contrib[i]);
	for(int i=1;i<=V;++i)
		pc[rep[i]] += corrCentrality[i];
	for(int i=1;i<=n;++i)
		pc[i] /= (sum_x - contrib[i]);

	double max_diff = 0;
	for(int i=1;i<=n;++i)
		max_diff = max(max_diff,abs(global_pc[i]-pc[i]));
	cerr << "Max difference in PC point-wise : " << max_diff << "\n";

	return 0;
}
