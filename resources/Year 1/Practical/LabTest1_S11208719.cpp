#include <iostream> /*Header File*/       //Name: Yash Pratik Prasad
using namespace std; /*Standard library*/  //Student ID: S11208719

int main(){
	int option; //option is the choice of game
	const int car = 10; //amount of car racing
	const double mr = 11.50; //amount of motorcycle racing
	const int vr = 16; //amount of virtual reality
	const int vrg = 3; //amount of virtual reality glasses
	int carp=0; //calculated price for car racing
	int mrp=0; //calculated price for motorcycle racing
	int vrp=0; //calculated price for virtual reality
	int num=0; //nuumber of players
	
	cout<<"\t\t\tGames offered\t\tCost Per Person\n";
	cout<<"\t\t\t1. Car Racing\t\t\t$10.00\n";
	cout<<"\t\t\t2. Motorcycle Racing\t\t$11.50\n";
	cout<<"\t\t\t3. Virtual Reality - 3D\t\t$16.00\n\n";
	cout<<"\t\t\tEnter choice:"; cin>>option; cout<<"\n"; //Enter choice of game
	cout<<"\t\t\tNumber of players:"; cin>>num; cout<<"\n\n"; //Enter number of players
	
	if(option == 1){
		carp = car*num;
		cout<<"\t\t\tTotal Payment is $"<<carp<<endl; //calculate the price if user chooses car racing
	}
	else if (option == 2){
		mrp = mr * num;
		cout<<"\t\t\tTotal Payment is $"<<mrp<<endl; //calculate the price if user chooses motorcycle racing
	}
	else if (option == 3){
		vrp = (vr*num) + (num*vrg);
		cout<<"\t\t\tTotal Payment is $"<<vrp<<endl; //calculate the price if user chooses virtual reality
	}
	else
		cout<<"\t\t\tINVALID INPUT"<<endl; //User enters any number other than 1, 2, 3
		
	while (cin.fail()){
		cin.clear();
		string not_int;
		cin>>not_int;
		cout<<"\t\t\tPlease Retry And Enter A Number From Above \n"; //Avoid using letters
		
	}
	
	return 0;
}
