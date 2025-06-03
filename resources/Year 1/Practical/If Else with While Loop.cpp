#include <iostream>					//Name: Yash Pratik Prasad	Student ID: S11208719
#include <string>
#include <iomanip>
#include <limits>

using namespace std;

int main()
{
    const string Hotel1 = "Marriot";	//Declarations for hotel name
    const string Hotel2 = "Hilton";	
    const string Hotel3 = "Uprising";
    const string Hotel4 = "Grand Pacific Hotel";
    const string Hotel5 = "Sheraton";
    const string Quit = "Exit";
    double pc1 = 0, pc2 = 0, pc3 = 0, pc4 = 0, pc5 = 0;	//percentages
    double MV = 0, HV = 0, UV = 0, GV = 0, SV = 0;	//individual votes per hotel
    double vv = 0;		//valid votes
    double t_inv = 0;	//total invalid votes
    double inv = 0;		//invalid votes
    double tvv = 0;		//total valid votes
    double tv = 0;		//total votes
    string winner;		//highest voted hotel
    string draw1, draw2, draw3, draw4, draw5;	//incase of multiple hotels getting equal vote
    double percent = 100;	//total percentage(avoid magic numbers)
    int selection = 0;		//input

    cout << "*********** HOTELS IN FIJI *********** \n";

    cout << "\t 1. Marriot \n";
    cout << "\t 2. Hilton \n";
    cout << "\t 3. Uprising \n";
    cout << "\t 4. Grand Pacific Hotel \n";
    cout << "\t 5. Sheraton \n";
    cout << "\t 6. Quit voting \n";			//initial display

    while (selection != 6)	//vote calculations

    {
        for (int i = 0; selection > i; i++)
        {
           if (selection == 1)	
            {
                MV = MV + 1;	//marriot votes
            }

            else if (selection == 2)
            {
                HV = HV + 1;	//hilton votes
            }

            else if (selection == 3)
            {
                UV = UV + 1;	//uprising votes
            }

            else if (selection == 4)
            {
                GV = GV + 1;	//GPH votes
            }

            else if (selection == 5)
            {
                SV = SV + 1;	//Sheraton votes
            }

            else
            {
                cout << "\n\t\t\t INVALID INPUT \n\n"; 	//invalid votes 
                inv = inv + 1;
            }
            break;		
        }

        cout << "Please choose your favorite hotels in Fiji from the list above by number. ";
        cin >> selection;

        while (1)
        {
            if (cin.fail())
            {
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');

                cout << " \t\t\t \n \t\t\t Numbers from 1 - 6 only please!!! \n"	//avoid alphabets
                     << endl;
                cin >> selection;
            }
            if (!cin.fail())
                break;
        }
    }

    double a = max(MV, HV);
    double b = max(UV, GV);
    double c = max(SV, a);
    double d = max(b, c);	//calculation for most voted hotel

    

    if (MV == d)
    {
        winner = Hotel1;

        draw1 = winner;
    }

    if (HV == d)
    {
        winner = Hotel2;

        draw2 = winner;
    }
    if (UV == d)
    {
        winner = Hotel3;

        draw3 = winner;
    }
    if (GV == d)
    {
        winner = Hotel4;

        draw4 = winner;
    }
    if (SV == d)
    {
        winner = Hotel5;

        draw5 = winner;
    }
    else
    {
        cout << "INVALID";
    }
    
    tvv = (MV + HV + UV + GV + SV);
    vv = tvv;
    t_inv = inv;

    tv = tvv + t_inv; //final vote calculation

    pc1 = (MV * percent) / tvv;
    pc2 = (HV * percent) / tvv;
    pc3 = (UV * percent) / tvv; //percentage calculation
    pc4 = (GV * percent) / tvv;
    pc5 = (SV * percent) / tvv;

    if (selection == 6)
    {
        cout << "\n\n\n";

        cout << "\n\n\n\n\n\n\n *********** FAVORITE FIJIAN HOTEL OPINION POLL *********** \n";
        cout << " \t ITEM \t\t\t VOTES \t\t\t % \n";
        cout << " \t ---- \t\t\t ----- \t\t\t - \n";
        cout << " \t Marriot"
             << " \t\t" << fixed << setprecision(0) << MV << "\t\t\t" << fixed << setprecision(1) << pc1 << endl;
        cout << " \t Hilton"
             << " \t\t" << fixed << setprecision(0) << HV << "\t\t\t" << fixed << setprecision(1) << pc2 << endl;
        cout << " \t Uprising"
             << " \t\t" << fixed << setprecision(0) << UV << "\t\t\t" << fixed << setprecision(1) << pc3 << endl;
        cout << " \t Grand Pacific Hotel"
             << " \t" << fixed << setprecision(0) << GV << "\t\t\t" << fixed << setprecision(1) << pc4 << endl;
        cout << " \t Sheraton"
             << " \t\t" << fixed << setprecision(0) << SV << "\t\t\t" << fixed << setprecision(1) << pc5 << endl
             << endl;

        cout << " \t According to this Poll, the hotel with majority of votes is " << fixed << setprecision(0) << endl <<endl <<"\t\t" <<draw1 << endl;
        cout <<" \t\t" <<draw2 << endl;
        cout <<" \t\t" <<draw3 << endl;
        cout <<" \t\t" <<draw4 << endl;
        cout <<" \t\t" <<draw5 << "\n\n";

        cout << " \t Total Valid Votes: \t\t " << fixed << setprecision(0) << tvv << endl;
        cout << " \t Total Invalid Votes:  \t\t"
             << " " << fixed << setprecision(0) << t_inv;
        cout << " \t Total votes received: \t\t" << fixed << setprecision(0) << tv;  //final display table
    }

    return 0;
}
