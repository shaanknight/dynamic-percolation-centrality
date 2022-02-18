#include<bits/stdc++.h>
#include<omp.h>
#include<chrono>
using namespace std;

// compile : g++ -O3 -fopenmp -static-libstdc++ <file_name>.cpp -o computePC-static-edgeUpdate
int numthreads = 48;
const int INF = 1e9;

int N,M;
vector<vector<int> > adj;
vector<int> query_node;
vector<double> percolation, x, updated_x, contrib, global_pc;
vector<pair<double,int> > perc;
vector<double> test_percolation;

void brandes(int src,vector<double> &x, vector<vector<int> > &adj,double *ptr, bool sub = false)
{	
	int N = (int)x.size()-1;
	queue<int> q;
	stack<int> st;
	vector<int> dist(N+1,-1);
	vector<double> sig(N+1,0.0),delta(N+1,0.0);
	vector<vector<int> > pr(N+1);

	int u = src;
	q.push(u);
	dist[u] = 0;
	sig[u] = 1.0;

	while(!q.empty())
	{
		u = q.front();
		q.pop();
		st.push(u);

		for(auto v:adj[u])
		{
			if(dist[v] < 0)
			{
				dist[v] = dist[u]+1;
				q.push(v);
			}
			if(dist[v] == dist[u]+1)
			{
				pr[v].push_back(u);
				sig[v] = sig[u]+sig[v];
			}
		}
	}

	while(!(st.empty()))
	{
		u = st.top();
		st.pop();
		for(auto p:pr[u])
		{
			double g = sig[p]/sig[u];
			g = g*(max(x[src]-x[u],(double)(0.0))+delta[u]);
			delta[p] = delta[p]+g;
		}
		if(u != src){
			if(sub) {
				ptr[u] -= delta[u];
			} 
			else {
				ptr[u] += delta[u];
			}

		}
		pr[u].clear();
		delta[u] = 0;
		sig[u] = 0;
		dist[u] = -1;
	}
}

void get_dist_array(int src, vector<vector<int> > &adj, vector<int> &dist) {
	queue<int> q;
	dist[src] = 0;
	q.push(src);
	while(!q.empty()){
		int u = q.front();
		q.pop();
		for(int v: adj[u]){
			if(dist[v] != -1) continue;
			dist[v] = dist[u] + 1;
			q.push(v);
		}
	}
}

void get_affected_vertices(vector<vector<int> > &adj, vector<pair<int,int> > &edge_batch, vector<int> &affected) {
	int *mtr = &affected[0];
	#pragma omp parallel for reduction (+:mtr[:N+1]) 
	for(int j=0;j<(int)edge_batch.size();++j){
		vector<int> epDistance1(N+1,-1), epDistance2(N+1,-1);
		auto e = edge_batch[j];
		int u = e.first;
		int v = e.second;
		get_dist_array(u, adj, epDistance1);
		get_dist_array(v, adj, epDistance2);
		for(int i=1; i<=N; i++) {
			if(epDistance1[i] != epDistance2[i]) {
				mtr[i]++;
			}
		}
	}
}

int main( int argc, char **argv ) {
	ios::sync_with_stdio(false);
	cin.tie(0);
	cout.tie(0);

    string input = argv[1];
	string queries = argv[2];
	string output = argv[3];
	omp_set_num_threads(numthreads);
	ifstream fin(input);
	ofstream fout(output);

	fin >> N >> M; 
	int u,v;
	adj.resize(N+1);
	x.push_back(0);
	for(int i=0;i<N;++i)
	{
		double prc = 1.0/(double)(i+1);
		x.push_back(prc);
	}
	for(int i=0;i<M;++i)
	{
		fin >> u >> v;
		adj[u].push_back(v);
		adj[v].push_back(u);
	}
	auto t1 = std::chrono::high_resolution_clock::now();

	perc.resize(N+1);
	contrib.resize(N+1);
	updated_x.resize(N+1);
	global_pc.resize(N+1);
	percolation.resize(N+1,0);
	test_percolation.resize(N+1,0);

	for(int i=1;i<=N;++i)
		perc[i] = {x[i],i};
	sort(perc.begin(),perc.end());
	double carry = 0, sum_x = 0;
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
	
	double *ptr = &percolation[0];
	#pragma omp parallel for reduction (+:ptr[:N+1]) 
	for(int i=1;i<=N;++i)
		brandes(i,x,adj,ptr);
	for(int i=1;i<=N;++i)
		global_pc[i] = percolation[i]/(sum_x-contrib[i]);

	auto t2 = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
	cerr << "Initial Static Computation time : " << duration << " mu.s." << endl;
	duration = 0;
	auto duration_actual = duration;

	ifstream qin(queries);
	int batch_size;
	while(qin >> batch_size)
	{
		query_node.resize(1);
		int u,v;
		vector<pair<int,int> > query_edge;
		for(int i=1;i<=batch_size;++i)
		{
			qin >> u >> v;
			query_edge.push_back(make_pair(u,v));
		}
		auto t3 = std::chrono::high_resolution_clock::now();
		for(auto &e: query_edge){
			adj[e.first].push_back(e.second);
			adj[e.second].push_back(e.first);
		}
		fill(percolation.begin(),percolation.end(),0);
		ptr = &percolation[0];
		#pragma omp parallel for reduction (+:ptr[:N+1]) 
		for(int i=1;i<=N;++i)
			brandes(i,x,adj,ptr);
		auto t4 = std::chrono::high_resolution_clock::now();
		duration_actual += std::chrono::duration_cast<std::chrono::microseconds>( t4 - t3 ).count();
		for(int i=1;i<=N;++i)
			fout << percolation[i]/(sum_x-contrib[i]) << " ";
		fout << "\n";
	}
	cerr << "Total time for updates : " << duration_actual << " mu.s." <<endl;
	fill(test_percolation.begin(),test_percolation.end(),0);
	ptr = &test_percolation[0];
	#pragma omp parallel for reduction (+:ptr[:N+1]) 
	for(int i=1;i<=N;++i)
		brandes(i,x,adj,ptr);
	double max_diff = 0;
	for(int i=1;i<=N;++i)
		max_diff = max(max_diff,abs(percolation[i]-test_percolation[i])/(sum_x-contrib[i]));
	cerr << "Max difference in PC point-wise : " << max_diff << "\n";

	return 0;
}