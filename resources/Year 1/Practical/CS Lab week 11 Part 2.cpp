#include <iostream>
using namespace std;

int main()
{
	int num [10] = {1,2,3,4,5,6,7,8,9,10};
	int product = 1;
	for(int i = 0; i < 10; i++){
		product = product * num[i];
		
	}
	
	cout<<"The product is "<<product<<endl;
	return 0;
}
