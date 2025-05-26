#include <iostream>
using namespace std;

int main(){
	
	char category;
	int R_hours = 0;
	int X_hours = 0;
	int hours = 0;
	double X_Pay = 0;
	double gross_wage = 0;	
	double net_wage = 0;
	const int M_Rate = 35;
	const double M_Pay = 10.60;
	const double M_Extra_Pay = 15.90;
	const int F_Rate = 20;
	const double F_Pay = 8.30;
	const double X_F_Pay = 12.45;


	cout<<"\t\t\t\t\t***************************************\n";
	cout<<"\t\t\t\t\t*Wage Calculator for shoe Lace Company*\n";	
	cout<<"\t\t\t\t\t***************************************\n\n";


	start:

	cout<<"\tEnter Category: 'M' for Management, 'F' for Floor Worker. ";
	cin>>category;

	 hour:
 	
	 cout<<"\tEnter Hours Worked. ";
	 cin>>hours;	
		if (category == 'M'){
  
  			if(hours <= 40){

    		gross_wage = M_Pay * hours;
	    	net_wage = gross_wage * M_Rate/100;
	    	net_wage = gross_wage - net_wage;	
		}
  		else if (hours > 40)
  		{
  	
  			X_hours = hours - 40;
  			R_hours = hours - X_hours;
  			X_Pay = 15.90 * X_hours;
  	
  			gross_wage = M_Pay * R_hours + X_Pay;
			net_wage = gross_wage * M_Rate/100;
			net_wage = gross_wage - net_wage;
  		}	
  		else
 		{
 			cout<<"Invalid Input"<<endl;
		  }
		}
  
	
  	else if (category == 'F')
  		{
  		if (hours <= 40)
  			{
	
    			gross_wage = F_Pay * hours;
	    		net_wage = gross_wage * F_Rate/100;
	    		net_wage = gross_wage - net_wage;	
			}
  		else if(hours > 40)
  		{
  	
  			X_hours = hours - 40;
  			R_hours = hours - X_hours;
  			X_Pay = 12.45 * X_hours;
  	
  			gross_wage = F_Pay * R_hours + X_Pay;
			net_wage = gross_wage * F_Rate/100;
			net_wage = gross_wage - net_wage;
  		}		
 		else
 		{
 			cout<<"Invalid Input"<<endl;
		}
		}
	else 
		{
			cout<<"\t\t\t\tInvalid Input\n"<<endl;
		}
	

	cout<<"\t\t___________________________"<<endl;
	cout<<"\t\t|Summary:        |        |\n";
	cout<<"\t\t|----------------|        |\n";
	cout<<"\t\t|Staff Category: |"<<category<<"       |"<<endl;
	cout<<"\t\t|Hours Worked:   |"<<hours<<"      |"<<endl;
	cout<<"\t\t|NET WAGE:$      |"<<net_wage<<"    |\n";
	cout<<"\t\t|_________________________|"<<endl;

	return 0;
}
