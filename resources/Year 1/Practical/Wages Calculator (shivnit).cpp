#include <iostream>
using namespace std;

int main(){
	string id;
	string name;
	const double rate = 11.25;
	double hrs;
	const int fee = 50;
	double gpay = 0;
	double deduc = 0;
	double final_pay = 0;
	double retire;
	
	
	cout<<"Please enter your ID:";
	cin >> id; cout<<endl;
	
	cout<<"Please enter your Full Name:";
	cin >> name; cout <<endl;
	
	cout<<"Please enter hours worked:";
	cin >> hrs; cout << endl;
	
	gpay = rate * hrs; 
	
	retire = 0.1 * gpay;
	
	deduc = fee + retire; 
	
	final_pay = gpay - deduc;
	
	cout<<"Name: "<<name<<endl;
	cout<<"ID: "<<id<<endl;
	cout<<"Gross Pay: "<<gpay<<endl;
	cout<<"Deductions: "<<deduc<<endl;
	cout<<"Take Home Pay: "<<final_pay<<endl;
	
	return 0;
}
