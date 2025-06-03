#include <iostream>
using namespace std;

int main()
{
	const int PROPORTION = 100;
	int num [PROPORTION];
	int size = 0;
	int input;
	const int MULTIPLIER = 2;
	
	cout<<"Please enter a number. Press any letter to exit :"<<endl;
	while (cin >> input)
	{
		if(size < 100){
			num[size]=input;
			size++;
		}
		else{
			break;
		}
		
	}
	cin.clear();
	cin.ignore();
	
	for(int i = 0; i < size; i++){
		num[i] = num[i] * MULTIPLIER;
		cout<<num[i]<<endl;
	}
	
	return 0;
}
