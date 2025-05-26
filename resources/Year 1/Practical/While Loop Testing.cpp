#include <iostream>
#include <iomanip>
using namespace std;

int main(){
	int n=0;
	int p=0;
	int v;
	
	while( v >= 0  && v < 0){
		cout<<"Enter a Value between -100 and 100 only. Enter Q to Quit: "<<endl;
		cin>>v;
		
		if(v > 100 && v < -100){
			cout<<"Enter in the input range"<<endl;
		}
		
		if(v>=-100 && v<=0){
			n=n+1;
		}
		else if(v>0 && v<=100){
			p=p+1;
		}
		
		else {
			if(cin.fail()){
			cin.clear();
			cin.ignore();
			cout<<"Do Not Enter An Alphabet"<<endl;
			}
		}
		
	
}
	
	cout<<"Total positive numbers entered:"<<p<<endl;
	cout<<"Total negative numbers entered:"<<n<<endl;
	
	
	
	return 0;
}
