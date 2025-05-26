#include<iostream>
#include <iomanip>
using namespace std;

int main(){
	int km;
	char drunk;
	const int s1 = 60;
	const int s2 = 65;
	const int s3 = 70;
	
	cout<<"Enter Speed:";
	cin>>km;
	cout<<"Is the offender drunk? y for Yes and n for No"<<endl;
	cin>>drunk;
	
	if(drunk =='y'){
		
		if(km>s1 && km<s2){
		
			cout<<"Be careful And Go take a shower"<<endl;
		}
	
		else if(km==s2){
			
			cout<<"$7 fine for each km/hr over 60km/hr and please go take a shower"<<endl;
		}
		else if(km<=s3){
			cout<<"$7 fine for each km/hr over 60km/hr and please go take a shower"<<endl;
		}
		else if(km > s3){
			cout<<"$7 fine for each km/hr over 60km/hr and including 70km/hr & Pay $15 fine for each km/hr(for being) and spend the day/night"<<endl;
		}
				
		else{
			
			cout<<"INVALID INPUT"<<endl;
		}
	}
	
	else if(drunk =='n'){
			if(km>s1 && km<s2){
				cout<<"Warning Youre slightly over the speed limit"<<endl;
			}
			
			else if (km==s2){
				cout<<"Pay $5 fine for each km/hr over 60km/hr"<<endl;
			}
			else if (km<=s3){
				cout<<"Pay $5 fine for each km/hr over 60km/hr"<<endl;
			}
			else if(km>s3){
				cout<<"Pay $5 fine for each km/hr over 60km/hr upto and including 70km/hr"<<endl;
			}
				
			else{
				cout<<"INVALID INPUT"<<endl;
			}
					
	}
	else{
		cout<<"INVALID INPUT"<<endl;
	}
	
	return 0;
}
