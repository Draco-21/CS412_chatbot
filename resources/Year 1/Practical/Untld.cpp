#include <iostream>
using namespace std;

int main()
{
	int x = 0;
cout << "Please provide a value for x:";
cin >> x;

if ( x <= 1){
   if (x > 0) {
      cout << "Answer A\n";
   }
   else{
      cout << "Answer B\n";
   }
}
else{
  cout << "Answer C\n";
}				
	return 0;
}
