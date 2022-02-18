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

// compile : nvcc <file_name>.cu -arch=sm_70 -std=c++11 -Xcompiler -fopenmp -O3 -o computePC-dynamic-edgeUpdate

#define NUM_THREADS 32
#define NUM_BLOCKS 1024
const int BUFFER_SIZE = 5;

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
double *reach_suf, int *starters)
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
				for(int i = starters[source]+threadIdx.x;i < starters[source+1]-1;i+=NUM_THREADS)
				{
					int l = starters[rootIndex];
					int s = l;
					int r = starters[rootIndex+1]-1;
					int h = r;
					int mid = l;
					double xi = reach_suf[i]-reach_suf[i+1];
					while(l<=r)
					{
						mid = l+(r-l)/2;
						if(reach_suf[mid]-reach_suf[mid+1] >= xi && (mid == s || reach_suf[mid-1]-reach_suf[mid] < xi))
							break;
						else if(reach_suf[mid]-reach_suf[mid+1] < xi)
							l = mid+1;
						else
							r = mid-1; 
					}
					atomicAdd(&delta[source], reach_suf[mid]-xi*(h-mid));
					// for(int j = starters[source];j < starters[source+1];++j)
					// {
					// 	atomicAdd(&delta[source], max(0.0,reach_suf[i]-reach_suf[i+1]-reach_suf[j]+reach_suf[j+1]));
					// }
				}
				__syncthreads();
				for(int i = starters[rootIndex+1]-2-threadIdx.x;i >= starters[rootIndex] && reach_suf[i]-reach_suf[i+1] >= perc_state[source];i-=NUM_THREADS)
					atomicAdd(&delta[source], perc_state[source]-reach_suf[i]+reach_suf[i+1]);
			}
			__syncthreads();

			for(int id=threadIdx.x;id<degree;id+=NUM_THREADS)
			{
				int neighbour = dColumn[dRow[source]+id];
				if(neighbour == 0)
					break;
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

				int l = starters[rootIndex];
				int s = l;
				int r = starters[rootIndex+1]-1;
				int h = r;
				int mid = h;
				double xi = perc_state[n.child];
				if(reach_suf[h-1] >= xi)
				{
					while(l<=r)
					{
						mid = l+(r-l)/2;
						if(reach_suf[mid]-reach_suf[mid+1] >= xi && (mid == s || reach_suf[mid-1]-reach_suf[mid] < xi))
							break;
						else if(reach_suf[mid]-reach_suf[mid+1] < xi)
							l = mid+1;
						else
							r = mid-1; 
					}
				}
				
				atomicAdd(&delta[n.parent], (dPaths[n.parent]/dPaths[n.child])*(reach_suf[mid]-xi*(h-mid)+(delta[n.child])));
				
				if(atomicExch(&done[n.child], 1) == 0)
					atomicAdd(&dCentrality[n.child], delta[n.child]);

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
double *Paths, double *dDelta, node *Parents, double *dCentrality, int *crr, double *perc_state, 
double *reach_suf, int *starters, int batch_size, int *dQueries, double factor)
{
	__shared__ int arr[2];
	int *QLen = arr, *parIndex = arr+1;
	
	int rootPointer = blockIdx.x;
	int *Q = Queue + (blockIdx.x)*(V+1);
	double *dPaths = Paths + (blockIdx.x)*(V+1);
	double *delta = dDelta + (blockIdx.x)*(V+1);
	int *done = crr + (blockIdx.x)*(V+1);
	int *dDistance = Distance + (blockIdx.x)*(V+1);
	node *dParent = Parents + (blockIdx.x)*(E+1);

	while(rootPointer < batch_size)
	{
		int rootIndex = dQueries[rootPointer];
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
			int source = Q[oldQLen++];
			int degree = dRow[source+1] - dRow[source];
			if(source != rootIndex)
			{
				for(int i = starters[source]+threadIdx.x;i < starters[source+1]-1;i+=NUM_THREADS)
				{
					int l = starters[rootIndex];
					int s = l;
					int r = starters[rootIndex+1]-1;
					int h = r;
					int mid = l;
					double xi = reach_suf[i]-reach_suf[i+1];
					while(l<=r)
					{
						mid = l+(r-l)/2;
						if(reach_suf[mid]-reach_suf[mid+1] >= xi && (mid == s || reach_suf[mid-1]-reach_suf[mid] < xi))
							break;
						else if(reach_suf[mid]-reach_suf[mid+1] < xi)
							l = mid+1;
						else
							r = mid-1; 
					}
					atomicAdd(&delta[source], reach_suf[mid]-xi*(h-mid));
				}
				__syncthreads();
				for(int i = starters[rootIndex+1]-2-threadIdx.x;i >= starters[rootIndex] && reach_suf[i]-reach_suf[i+1] >= perc_state[source];i-=NUM_THREADS)
					atomicAdd(&delta[source], perc_state[source]-reach_suf[i]+reach_suf[i+1]);
			}
			__syncthreads();

			for(int id=threadIdx.x;id<degree;id+=NUM_THREADS)
			{
				int neighbour = dColumn[dRow[source]+id];
				if(neighbour == 0)
					break;
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

				int l = starters[rootIndex];
				int s = l;
				int r = starters[rootIndex+1]-1;
				int h = r;
				int mid = l;
				double xi = perc_state[n.child];
				while(l<=r)
				{
					mid = l+(r-l)/2;
					if(reach_suf[mid]-reach_suf[mid+1] >= xi && (mid == s || reach_suf[mid-1]-reach_suf[mid] < xi))
						break;
					else if(reach_suf[mid]-reach_suf[mid+1] < xi)
						l = mid+1;
					else
						r = mid-1; 
				}
				
				atomicAdd(&delta[n.parent], (dPaths[n.parent]/dPaths[n.child])*(reach_suf[mid]-xi*(h-mid)+(delta[n.child])));
				
				if(atomicExch(&done[n.child], 1) == 0)
					atomicAdd(&dCentrality[n.child], factor*delta[n.child]);

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
		rootPointer += NUM_BLOCKS;
	}
}

__global__ void check_affected(int V, int E, int *dColumn, int *dRow, int *Distance, int *Queue,
int *dQueries, int *dAffected)
{
	__shared__ int arr[1];
	int *QLen = arr;
	
	int rootPointer = blockIdx.x;
	int *Q = Queue + (blockIdx.x)*(V+1);
	int *nodeDistances = Distance + (2*blockIdx.x)*(V+1);
	for(int j=0;j<=1;++j)
	{
		int rootIndex = dQueries[2*rootPointer+j];
		int *dDistance = nodeDistances + j*(V+1);
		for(int i=threadIdx.x; i<=V; i+=NUM_THREADS) 
			dDistance[i] = -1;

		if(threadIdx.x==0)
		{
			*QLen = 1;
			int root = rootIndex;
			Q[0] = root;
			dDistance[root] = 0;
		}
		__syncthreads();

		int oldQLen = 0;
		while(oldQLen < *QLen)
		{
			int source = Q[oldQLen++];
			int degree = dRow[source+1] - dRow[source];

			for(int id=threadIdx.x;id<degree;id+=NUM_THREADS)
			{
				int neighbour = dColumn[dRow[source]+id];
				if(neighbour == 0)
					break;
				if(dDistance[neighbour] == -1)
				{
					dDistance[neighbour] = dDistance[source]+1;
					Q[atomicAdd(QLen, 1)] = neighbour;
				}
			}
			__syncthreads();
		}
	}
	__syncthreads();
	for(int i=threadIdx.x; i<=V; i+=NUM_THREADS) 
		if(nodeDistances[i] != nodeDistances[i+V+1])
			atomicAdd(&dAffected[i],1);
}

int n,m;
int vertices;
int timer;
int bcc_cnt;
int max_bcc,max_bcc_edges;
vector<double> x,updated_x,pc,global_pc;
vector<vector<double> > reach;
vector<vector<int> > reachv;
vector<vector<int> > copies;
vector<int> bcc_id;
vector<pair<int,int> > st;
vector<int> vis,vis1,low,entry, cur_comp;
vector<vector<int> > g, tmp_g;
vector<bool> in_otherbccs;
vector<pair<long double,int> > perc;
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

				max_bcc = max(max_bcc,(int)(unique_vertices.size()));
				++bcc_cnt;
				int edges = 0;
				for(int uv:unique_vertices){
					bcc_id[uv] = bcc_cnt;
					edges += (int)(tmp_g[uv].size());
					if(uv != vertices)
					{
						for(auto v:reachv[uv])
						{
							children.push_back(v);
							in_otherbccs[v] = 1;
						}
					}
				}
				max_bcc_edges = max(max_bcc_edges,edges);
				for(int i:cur_comp)
				{
					if(!in_otherbccs[i])
						reach[vertices].push_back(x[i]);
					else
						in_otherbccs[i] = 0;
				};
				sort(reach[vertices].begin(),reach[vertices].end());
				for(int i=(int)(reach[vertices].size())-2;i>=0;i--)
					reach[vertices][i] += reach[vertices][i+1];
				reach[vertices].push_back(0);
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
		reach[u].push_back(x[i]);
	sort(reach[u].begin(),reach[u].end());
	for(int i=(int)(reach[u].size())-2;i>=0;i--)
		reach[u][i] += reach[u][i+1];
	reach[u].push_back(0);
	
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

int calculate_diameter(vector<vector<int> > &adj)
{
	int diam = 0;
	for(int i=1;i<=10;++i)
	{
		int N = (int)adj.size()-1;
		int u = rand()%N + 1;
		int dmax = 0;

		for(int j=1;j<=20;++j)
		{
			queue<int> q;
			q.push(u);
			vector<int> dist(N+1,-1);
			dist[u] = 0;

			while(!q.empty())
			{
				u = q.front();
				q.pop();

				for(auto v:adj[u])
				{
					if(dist[v] < 0)
					{
						dist[v] = dist[u]+1;
						q.push(v);
					}
				}
			}
			if(dist[u] >= dmax)
				dmax = dist[u];
			else
				break;
		}
		diam = max(diam,dmax);
	}
	return diam;
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
	copies.resize(n+1);
	for(int i=1;i<=n;++i)
		copies[i].push_back(i);
	bcc_id.resize(2*m+1,0);
	
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
	for(int i=1;i<=n;++i)
	{
		sort(g[i].begin(),g[i].end());
		g[i].erase(unique(g[i].begin(), g[i].end()),g[i].end());
	}
	// int diameter = calculate_diameter(g);
	auto t1 = std::chrono::high_resolution_clock::now();

	auto res = compute_constants();
	double sum_x = res.first;
	vector<double> contrib = res.second;
	x.resize(2*m+1);
	for(int i=1;i<=n;i++){
		if(!vis[i]){
			cur_comp.clear();
			prelim_dfs(i);
			dfs(i,0);
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
	// cout << "," << bcc_cnt;
	// double ratio = (1.0*max_bcc+1.0*max_bcc_edges)/(1.0*n+2.0*m);
	// cerr << diameter << "," << max_bcc << "," << max_bcc_edges << "," << ratio << "," << cnt_reach_vec << endl;
	// cout << diameter << "," << max_bcc << "," << max_bcc_edges << "," << ratio << "," << cnt_reach_vec << ",";
	// return 0;
	int CAP = 2*E+V*BUFFER_SIZE;
	int *hColumn = new int[CAP];
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
		for(int b=0;b<BUFFER_SIZE;++b)
			hColumn[index++] = 0;
	}
	
	long count = 0;
	for(int i=1; i<=V;)
	{
		for(int j=1;j<(int)tmp_g.size();++j)
		{
			vector<int> v = tmp_g[i];
			hRow[i++] = count;
			count += (int)(v.size())+BUFFER_SIZE;
		}
	}
	
	hRow[V+1] = count;
	double *intm_reach = new double[cnt_reach_vec+2];
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
	
	double *delta, *Paths, *dCentrality, *perc_state, *reach_suf;
	node *Parents;
	int *dColumn, *dRow, *Distance, *Queue, *crr, *starters, *org;

	cudaMalloc((void**)&dRow,    		 sizeof(int)*(V+2));
	cudaMalloc((void**)&dCentrality,	 sizeof(double)*(V+2));
	cudaMalloc((void**)&perc_state,	 sizeof(double)*(V+2));
	cudaMalloc((void**)&dColumn, 		 sizeof(int)*(CAP));
	cudaMalloc((void**)&crr,    		 sizeof(int)*(V+2)*NUM_BLOCKS);
	cudaMalloc((void**)&Queue,    	 sizeof(int)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&Distance,		 sizeof(int)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&Paths,			 sizeof(double)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&delta,			 sizeof(double)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&Parents,		 sizeof(node)*(E+1+V*BUFFER_SIZE)*NUM_BLOCKS);
	cudaMalloc((void**)&reach_suf,	 sizeof(double)*(cnt_reach_vec+2));
	cudaMalloc((void**)&starters, 	 sizeof(int)*(V+2));
	cudaMalloc((void**)&org, 		 	 sizeof(int)*(V+2));

	cudaMemcpy(dRow, hRow, sizeof(int)*(V+2),cudaMemcpyHostToDevice);
	cudaMemcpy(dColumn, hColumn, sizeof(int)*(CAP), cudaMemcpyHostToDevice);
	cudaMemcpy(perc_state, perc, sizeof(double)*(V+2),cudaMemcpyHostToDevice);
	cudaMemcpy(reach_suf, intm_reach, sizeof(double)*(cnt_reach_vec+2),cudaMemcpyHostToDevice);
	cudaMemcpy(starters, pointers, sizeof(int)*(V+2),cudaMemcpyHostToDevice);
	cudaMemcpy(org, cOrg, sizeof(int)*(V+2),cudaMemcpyHostToDevice);
	gpuErrchk( cudaPeekAtLastError() );

	brandes <<<NUM_BLOCKS, NUM_THREADS, 32>>> (V, E, dColumn, dRow, Distance, Queue, Paths, delta,
				Parents, dCentrality, crr, perc_state, reach_suf, starters);

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
	
	int *dQueries;
	int *dAffected;

	ifstream qin(queries);
	int batch_size;
	while(qin >> batch_size)
	{
		int queries[2*batch_size];
		for(int i=1;i<=batch_size;++i)
		{
			int u,v;
			qin >> u >> v;
			int p = 0,q = 0;
			for(auto c_u:copies[u])
			{
				for(auto c_v:copies[v])
				{
					if(bcc_id[c_u] == bcc_id[c_v])
					{
						p = c_u;
						q = c_v;
						break;
					}
				}
				if(p > 0)
					break;
			}
			if(p == 0)
			{
				cerr << "Caution : Need to recompute BCCs" << endl;
				return 0;
			}
			queries[2*i-2] = p;
			queries[2*i-1] = q;
		}

		auto t3 = std::chrono::high_resolution_clock::now();

		cudaMalloc((void**)&dQueries,sizeof(int)*(2*batch_size));
		cudaMalloc((void**)&dAffected,sizeof(int)*(V+1));
		cudaMemcpy(dQueries, queries, sizeof(int)*(2*batch_size), cudaMemcpyHostToDevice);

		gpuErrchk( cudaPeekAtLastError() );

		check_affected <<<batch_size, NUM_THREADS, 32>>> (V, E, dColumn, dRow, Distance, Queue, dQueries, dAffected);

		cudaDeviceSynchronize();
		gpuErrchk( cudaPeekAtLastError() );
		cudaDeviceSynchronize();

		int *affected = new int[V+1];
		cudaMemcpy(affected, dAffected, sizeof(int)*(V+1), cudaMemcpyDeviceToHost);

		int affected_nodes = 0;
		for(int i=1;i<=V;++i)
			if(affected[i]) affected_nodes++;

		int k = 0;
		int batch[affected_nodes];
		for(int i=1;i<=V;++i)
			if(affected[i]) batch[k++] = i;

		cudaMalloc((void**)&dQueries,sizeof(int)*(affected_nodes));

		cudaMemcpy(dQueries, batch, sizeof(int)*(affected_nodes), cudaMemcpyHostToDevice);
		gpuErrchk( cudaPeekAtLastError() );

		update_brandes <<<NUM_BLOCKS, NUM_THREADS, 32>>> (V, E, dColumn, dRow, Distance, Queue, Paths, delta,
				Parents, dCentrality, crr, perc_state, reach_suf, starters, affected_nodes, dQueries, -1.0);

		cudaDeviceSynchronize();
		gpuErrchk( cudaPeekAtLastError() );
		cudaDeviceSynchronize();

		for(int i=1;i<=batch_size;++i)
		{
			int u,v;
			u = queries[2*i-2];
			v = queries[2*i-1];
			int sz_u = tmp_g[u].size();
			int sz_v = tmp_g[v].size();
			hColumn[hRow[u]+sz_u] = v;
			hColumn[hRow[v]+sz_v] = u;
			tmp_g[u].push_back(v);
			tmp_g[v].push_back(u);
		}

		E += batch_size;
		cudaMemcpy(dColumn, hColumn, sizeof(int)*(CAP), cudaMemcpyHostToDevice);

		gpuErrchk( cudaPeekAtLastError() );

		update_brandes <<<NUM_BLOCKS, NUM_THREADS, 32>>> (V, E, dColumn, dRow, Distance, Queue, Paths, delta,
				Parents, dCentrality, crr, perc_state, reach_suf, starters, affected_nodes, dQueries, 1.0);

		cudaDeviceSynchronize();
		gpuErrchk( cudaPeekAtLastError() );
		cudaDeviceSynchronize();

		auto t4 = std::chrono::high_resolution_clock::now();
		duration_dynamic += std::chrono::duration_cast<std::chrono::microseconds>( t4 - t3 ).count();
	}
	cerr << "Total time for updates : " << duration_dynamic << " mu.s." << endl;
	
	cudaMemcpy(Centrality, dCentrality, sizeof(double)*(V+1), cudaMemcpyDeviceToHost);

	double *fCentrality;
	cudaMalloc((void**)&fCentrality,	sizeof(double)*(V+2));
	gpuErrchk( cudaPeekAtLastError() );

	brandes <<<NUM_BLOCKS, NUM_THREADS, 32>>> (V, E, dColumn, dRow, Distance, Queue, Paths, delta,
				Parents, fCentrality, crr, perc_state, reach_suf, starters);

	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );
	cudaDeviceSynchronize();

	double *corrCentrality = new double[V+1];
	cudaMemcpy(corrCentrality, fCentrality, sizeof(double)*(V+1), cudaMemcpyDeviceToHost);

	double max_diff = 0;
	for(int i=1;i<=V;++i)
		max_diff = max(max_diff,abs(corrCentrality[i]-Centrality[i])/(sum_x-contrib[i]));
	cerr << "Max difference in PC point-wise : " << max_diff << "\n";
}
