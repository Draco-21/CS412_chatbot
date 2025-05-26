#include <iostream>
using namespace std;

int main()
 {
   int animals = 0;
    int dogs, cats;
  
    cout << "How many dogs have you seen? ";
    cin >> dogs;
  
    if( dogs > 0 ){
      cout << "How many animals have you seen altogether? ";
      cin >> animals;
     cats = animals - dogs;
    }

   cout << "There were " << cats << " cats\n";
   cout << "There were " << dogs << " dogs\n";
   cout << "The percentage dogs is " << 100 * dogs/animals << "%\n";
	return 0;
}
