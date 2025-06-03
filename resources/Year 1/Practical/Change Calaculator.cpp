/*CS111 week 4 Lab
*/
#include <iostream>
using namespace std;

int  main()
{
	double const price=10.50;
	int x; 
	int cash;
	 int tenb=10;
	 int fiveb=5;
	 int twoc=2;
	 int onec=1;
	 int fiftyc=50;
	 int twenc=20;
	 int tenc=10;
	 int fivec=5;
	 int conversion=100;
	double z;
	double y;
	double change;
	cout<<"\t\t\t\tChange Calculator Program for Yummy Pizza & Bakery Shop\n\n"<<endl;
	cout<<"\t\t\t\tYummy Pizza sells only medium sized pizza for $10.50 only\n\n";
	cout<<"\t\t\t\t\tIts the best deal in  town! So come along!!!\n\n\n\n";
	cout<<"\t\t\t\t**********************************************************\n\n"<<endl;
	cout<<"Enter Quantity of Pizzas:";
	cin>>x;
	z=x*price;
	cout<<"Total Price is $"<<z<<endl;
	cout<<"Enter Cash Received:";
	cin>>cash;
	y=cash-z;
	cout<<"Change is $"<<y<<endl;
	change =y*conversion;
	tenb = change/1000;
	cout<<tenb; cout<<" 10 dollars note"<<endl;
	fiveb = change/100/50;
	cout<<fiveb; cout<<" 5 dollars note"<<endl;
	twoc = change/100/50;
	cout<<twoc; cout<<" 2 dollars coin"<<endl;
	onec = change/100/50;
	cout<<onec; cout<<" 1 dollars coin"<<endl;
	 = change/100/50;
	cout<<fiveb; cout<<" 5 dollars note"<<endl;
	
	
	
	
	
	return 0;
}
