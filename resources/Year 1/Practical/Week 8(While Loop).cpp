#include <iostream>
using namespace std;


int main(){
	int sum = 0;
	int i = 1;
	
	while (i * 7 < 1000){
		sum = sum + (i * 7);
		i = i + 1;
	}
	
	cout<<"The Sum is "<<sum;
	
	return 0;
}
