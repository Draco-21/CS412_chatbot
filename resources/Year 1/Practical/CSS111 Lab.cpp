#include <iostream>
using namespace std;

int main()
{
	string fname;
    int pno;
    string email;
    cout << "Please enter your first name:";
    cin >> fname;
    cout<<"Please enter your phone number:";
    cin>>pno;
    cout<<"Please enter your email address:";
    cin>>email;
    cout << "Dear " << fname <<endl;
    cout<<"Mobile:"<<pno<<endl;
    cout<<"Email:"<<email<<endl;
    cout << "You are invited to my public lecture next Friday.\n";
    return 0;
}
