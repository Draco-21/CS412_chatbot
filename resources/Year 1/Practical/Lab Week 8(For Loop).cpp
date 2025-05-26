#include <iostream>
using namespace std;


int main(){
	
	int sum = 0;
	
	for (int i = 7; i < 1000; i = i +7){
			sum = sum + i;
		}
		
	cout<<"The Sum is "<<sum;
	return 0;
}
