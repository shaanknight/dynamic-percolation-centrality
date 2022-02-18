#include<bits/stdc++.h>
#include<omp.h>
#include<chrono>
using namespace std;

int N,M;
int timer;
vector<pair<int,int> > st;
vector<int> vis,low,entry;

vector<vector<int> > adj;
vector<vector<int> > bccno;

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
int bcccnt = 0;
void dfs(int u, int par){
	// //cerr<<u<<endl;
	timer++;
	entry[u] = timer;
	vis[u] = 2;
	low[u] = entry[u];
	for(int v:adj[u]){
		if(vis[v]< 2){
			st.push_back({u,v});
			dfs(v, u);
			low[u] = min(low[u], low[v]);
			if(low[v] >= entry[u]){
				bcccnt += 1;
				// this is an articulation point and everything I discovered is a BCC
				//cerr<<u<<" is an articulation point for "<<v<<endl;
				while(st.back() != make_pair(u,v)){
					int p = st.back().first;
					int q = st.back().second;
					st.pop_back();
					//cerr<<"popping "<<p<<" "<<q<<" from the stack"<<endl;
					bccno[p].push_back(bcccnt);
					bccno[q].push_back(bcccnt);
				}
				st.pop_back();
				bccno[v].push_back(bcccnt);
				bccno[u].push_back(bcccnt);
			}
		}
		else if(v != par && entry[v] < entry[u]){ //back edge
			st.push_back({u,v});
			low[u] = min(low[u], entry[v]);
		}
	}
	vis[u] = 3;
}

bool same_bcc(int u, int v){
	for(int i:bccno[u]){
		if(find(bccno[v].begin(), bccno[v].end(), i) != bccno[v].end()){
			return true;
		}
	}
	return false;
}

int main( int argc, char **argv ) {
    string input = argv[1];
    int query_size = atoi(argv[2]);
    int batch_size = 1;

    ifstream fin(input);
	fin >> N >> M; 
	int u,v;
	adj.resize(N+1);
	bccno.resize(N+1);
	set<pair<int,int> > exist_edges;
	for(int i=0;i<M;++i)
	{
		fin >> u >> v;
		adj[u].push_back(v);
		adj[v].push_back(u);
	}
	vis.resize(N+1);
	low.resize(N+1);
	entry.resize(N+1);
	for(int i=1;i<=N;i++){
		if(!vis[i]){
			dfs(i,0);
		}
	}
	for(int i=1;i<=N;i++){
		sort(bccno[i].begin(), bccno[i].end());
		bccno[i].erase(unique(bccno[i].begin(), bccno[i].end()), bccno[i].end());
	}
	set<pair<int,int> > done;
	for(int t=0;t<query_size;++t)
	{
		cout<<batch_size<<endl;
		for(int i=0;i<batch_size;i++){
			int u = uniform_int_distribution<int>(1,N)(rng);
			int v = uniform_int_distribution<int>(1,N)(rng);
			if(u == v || done.count({u,v}) || !same_bcc(u,v)){
				//cout << "failed " << u << " " << v << endl;
				i--;
				continue;
			}
			cout<<u<<" "<<v<<endl;
			done.insert({u,v});
			done.insert({v,u});
		}
	}
}