#include <iostream>
#include <cstdlib>
#include <string>
#include <ctime>

using namespace std;

  int main(){
    int j;
    for(int i = 10; i>=0; i--)
       for(j = 0; j != 3; j++){
         cout << i+j << ",\t";
       }
       cout << endl;
    
  
  return 0;
 }

