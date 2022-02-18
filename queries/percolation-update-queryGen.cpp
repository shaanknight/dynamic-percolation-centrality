#include<bits/stdc++.h>
#include<omp.h>
#include<chrono>
using namespace std;

int main( int argc, char **argv ) {
	ios::sync_with_stdio(false);
	cin.tie(0);
	cout.tie(0);

	int N,Q,batch_size;
    string input = argv[1];
    Q = atoi(argv[2]);
    batch_size = 1;

    ifstream fin(input);
    fin >> N;
    int arr[N];
	for(int i = 0; i < N; ++i)
	    arr[i] = i+1;
    
    for(int i=1;i<=Q;++i)
    {
		random_shuffle(arr, arr+N);
		cout << batch_size << "\n";
		for(int j=0;j<batch_size;++j)
			cout << arr[j] << " " << 0.001*(rand()%1000) << "\n";
    }
    fin.close();
}