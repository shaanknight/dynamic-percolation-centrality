#include<bits/stdc++.h>
#include<omp.h>
#include<chrono>
using namespace std;

// compile : g++ -O3 -fopenmp -static-libstdc++ <file_name>.cpp -o computePC-dynamic-edgeUpdate
int numthreads = 48;
const int INF = 1e9;

void brandes(int src,vector<double> x, vector<vector<int> > &adj,double *ptr)
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
			double g;
			g = sig[p]/sig[u];
			g = g*(max(x[src]-x[u],(double)(0.0))+delta[u]);
			delta[p] = delta[p]+g;
		}
		if(u != src)
			ptr[u] += delta[u];
		pr[u].clear();
		delta[u] = 0;
		sig[u] = 0;
		dist[u] = -1;
	}
}

void bcc_brandes(int src,vector<double> x,vector<vector<int> > &adj,vector<vector<double> > &reach,double *ptr,double factor)
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

		if(src != u)
		{
			for(int i=0;i<(int)(reach[u].size())-1;++i)
			{
				double xi = reach[u][i]-reach[u][i+1];
				int l = 0;
				int s = l;
				int r = (int)(reach[src].size())-1;
				int h = r;
				int mid = l;
				while(l<=r)
				{
					mid = l+(r-l)/2;
					if(reach[src][mid]-reach[src][mid+1] >= xi && (mid == s || reach[src][mid-1]-reach[src][mid] < xi))
						break;
					else if(reach[src][mid]-reach[src][mid+1] < xi)
						l = mid+1;
					else
						r = mid-1; 
				}
				delta[u] += reach[src][mid]-xi*(h-mid);
			}
			for(int i=(int)(reach[src].size())-2;i>=0 && reach[src][i]-reach[src][i+1] >= x[u];i--)
				delta[u] += x[u]-reach[src][i]+reach[src][i+1];
		}

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
			double g;
			g = sig[p]/sig[u];
			double xi = x[u];
			int l = 0;
			int s = l;
			int r = (int)(reach[src].size())-1;
			int h = r;
			int mid = l;
			while(l<=r)
			{
				mid = l+(r-l)/2;
				if(reach[src][mid]-reach[src][mid+1] >= xi && (mid == s || reach[src][mid-1]-reach[src][mid] < xi))
					break;
				else if(reach[src][mid]-reach[src][mid+1] < xi)
					l = mid+1;
				else
					r = mid-1; 
			}
			g = g*(reach[src][mid]-xi*(h-mid)+delta[u]);
			delta[p] = delta[p]+g;
		}
		if(u != src)
			ptr[u] += delta[u]*factor;
		pr[u].clear();
		delta[u] = 0;
		sig[u] = 0;
		dist[u] = -1;
	}
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
	int N = (int)(adj.size())-1;
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

int main(int argc, char **argv)
{
	ios::sync_with_stdio(false);
	cin.tie(0);
	cout.tie(0);

	string input = argv[1];
   string queries = argv[2];
   omp_set_num_threads(numthreads);

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
	int V = vertices;

	vector<double> pCentrality(V+1,0.0),ac(V+1,0.0);
	double *ptr = &pCentrality[0];
	#pragma omp parallel for reduction (+:ptr[:V+1]) 
	for(int i=1;i<=V;++i)
		bcc_brandes(i,x,tmp_g,reach,ptr,1.0);
	for(int i=1;i<=V;++i)
		ac[rep[i]] += pCentrality[i];
	for(int i=1;i<=n;++i)
		ac[i] /= (sum_x - contrib[i]);

	auto t2 = std::chrono::high_resolution_clock::now();

	ptr = &pc[0];
	#pragma omp parallel for reduction (+:ptr[:n+1]) 
	for(int i=1;i<=n;++i)
		brandes(i,x,g,ptr);
	for(int i=1;i<=n;++i)
		pc[i] /= (sum_x - contrib[i]);

	double max_diff = 0;
	for(int i=1;i<=n;++i)
		max_diff = max(max_diff,abs(ac[i]-pc[i]));
	
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
	cerr << "Initial Static Computation time : " << duration << " mu.s." << endl;
	// cerr << "Max difference in PC point-wise : " << max_diff << "\n";

	duration = 0;
	auto duration_dynamic = duration;

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
			query_edge.push_back(make_pair(p,q));
			g[u].push_back(v);
			g[v].push_back(u);
		}

		auto t3 = std::chrono::high_resolution_clock::now();

		vector<int> affected(V+1);
		get_affected_vertices(tmp_g, query_edge, affected);
		for(int i=1;i<=V;i++){
			if(affected[i]) {
				query_node.push_back(i);
				// cerr<<i << " is affected"<<endl;
			}
		}
		batch_size = (int)query_node.size()-1;
		// cerr<<"# affected nodes = "<<batch_size<<endl;

		ptr = &pCentrality[0];
		#pragma omp parallel for reduction (+:ptr[:V+1]) 
		for(int i=1;i<=batch_size;++i){
			bcc_brandes(query_node[i],x,tmp_g,reach,ptr,-1.0);
		}
		for(auto &e: query_edge){
			tmp_g[e.first].push_back(e.second);
			tmp_g[e.second].push_back(e.first);
		}
		#pragma omp parallel for reduction (+:ptr[:V+1]) 
		for(int i=1;i<=batch_size;++i){
			bcc_brandes(query_node[i],x,tmp_g,reach,ptr,1.0);
		}
		auto t4 = std::chrono::high_resolution_clock::now();
		duration_dynamic += std::chrono::duration_cast<std::chrono::microseconds>( t4 - t3 ).count();
	}
	cerr << "Total time for updates : " << duration_dynamic << " mu.s." <<endl;

	fill(ac.begin(),ac.end(),0);
	for(int i=1;i<=V;++i)
		ac[rep[i]] += pCentrality[i];
	for(int i=1;i<=n;++i)
		ac[i] /= (sum_x - contrib[i]);
	fill(pc.begin(),pc.end(),0);
	ptr = &pc[0];
	#pragma omp parallel for reduction (+:ptr[:n+1]) 
	for(int i=1;i<=n;++i)
		brandes(i,x,g,ptr);
	for(int i=1;i<=n;++i)
		pc[i] /= (sum_x - contrib[i]);

	max_diff = 0;
	for(int i=1;i<=n;++i)
		max_diff = max(max_diff,abs(ac[i]-pc[i]));
	cerr << "Max difference in PC point-wise : " << max_diff << "\n";

	return 0;
}
