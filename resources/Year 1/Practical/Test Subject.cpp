#include <iostream>
#include <string>
#include <iomanip>
#include <limits>

using namespace std;

int main()
{
    const string ITEM_1 = "Marriot";
    const string ITEM_2 = "Hilton";
    const string ITEM_3 = "Uprising";
    const string ITEM_4 = "Grand Pacific Hotel";
    const string ITEM_5 = "Sheraton";
    const string ITEM_6 = "Exit";
    double p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0;
    double VOTE_1 = 0, VOTE_2 = 0, VOTE_3 = 0, VOTE_4 = 0, VOTE_5 = 0;
    double total_valid_vote = 0;
    double total_invalid_vote = 0;
    double invalid = 0;
    double total_votes = 0;
    double total_all = 0;
    string winner;
    string draw1, draw2, draw3, draw4, draw5;
    double percent = 100;
    int options = 0;

    cout << "*********** FAVORITE FIJIAN HOTELS OPINION POLL *********** \n";

    cout << "\t 1. Marriot \n";
    cout << "\t 2. Hilton \n";
    cout << "\t 3. Uprising \n";
    cout << "\t 4. Grand Pacific Hotel \n";
    cout << "\t 5. Sheraton \n";
    cout << "\t 6. Quit voting \n";

    while (options != 6)

    {
        for (int i = 0; options > i; i++)
        {
            if (options == 1)
            {
                VOTE_1 = VOTE_1 + 1;
            }

            else if (options == 2)
            {
                VOTE_2 = VOTE_2 + 1;
            }

            else if (options == 3)
            {
                VOTE_3 = VOTE_3 + 1;
            }

            else if (options == 4)
            {
                VOTE_4 = VOTE_4 + 1;
            }

            else if (options == 5)
            {
                VOTE_5 = VOTE_5 + 1;
            }

            else
            {
                cout << "\n\t\t\t INVALID INPUT \n\n";
                invalid = invalid + 1;
            }
            break;
        }

        cout << "Please choose your favorite hotels in Fiji from the list above by number. ";
        cin >> options;

        while (1)
        {
            if (cin.fail())
            {
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(
                ), '\n');

                cout << " \t\t\t \n ONLY NUMERICAL VALUES!!! \n"
             
                             << endl;
                cin >> options;
            }
            if (!cin.fail())
                break;
        }
    }                

    double a = max(VOTE_1, VOTE_2);
    double b = max(VOTE_3, VOTE_4);
    double c = max(VOTE_5, a);
    double d = max(b, c);

    if (VOTE_1 == d)
    {
        winner = ITEM_1;
        draw1 = winner;
    }

    if (VOTE_2 == d)
    {
        winner = ITEM_2;
        draw2 = winner;
    }
    if (VOTE_3 == d)
    {
        winner = ITEM_3;
        draw3 = winner;
    }
    if (VOTE_4 == d)
    {
        winner = ITEM_4;
        draw4 = winner;
    }
    if (VOTE_5 == d)
    {
        winner = ITEM_5;
        draw5 = winner;
    }
    else
    {
        cout << " ";
    }

    total_votes = (VOTE_1 + VOTE_2 + VOTE_3 + VOTE_4 + VOTE_5);
    total_valid_vote = total_votes;
    total_invalid_vote = invalid;

    total_all = total_valid_vote + total_invalid_vote;

    p1 = (VOTE_1 * percent) / total_valid_vote;
    p2 = (VOTE_2 * percent) / total_valid_vote;
    p3 = (VOTE_3 * percent) / total_valid_vote;
    p4 = (VOTE_4 * percent) / total_valid_vote;
    p5 = (VOTE_5 * percent) / total_valid_vote;

    if (options == 6)

    {
        cout << "\n\n\n";

        cout << "\n\n\n\n\n\n\n *********** FAVORITE FIJIAN HOTEL OPINION POLL *********** \n";
        cout << " \t ITEM \t\t\t VOTES \t\t\t % \n";
        cout << " \t ---- \t\t\t ----- \t\t\t - \n";
        cout << " \t Marriot"
             << " \t\t" << fixed << setprecision(0) << VOTE_1 << "\t\t\t" << fixed << setprecision(1) << p1 << endl;
        cout << " \t Hilton"
             << " \t\t" << fixed << setprecision(0) << VOTE_2 << "\t\t\t" << fixed << setprecision(1) << p2 << endl;
        cout << " \t Uprising"
             << " \t\t" << fixed << setprecision(0) << VOTE_3 << "\t\t\t" << fixed << setprecision(1) << p3 << endl;
        cout << " \t Grand Pacific Hotel"
             << " \t" << fixed << setprecision(0) << VOTE_4 << "\t\t\t" << fixed << setprecision(1) << p4 << endl;
        cout << " \t Sheraton"
             << " \t\t" << fixed << setprecision(0) << VOTE_5 << "\t\t\t" << fixed << setprecision(1) << p5 << endl
             << endl;

        cout << " \t According to this Poll, the hotel with majority of votes is " << fixed << setprecision(0) << draw1 << endl;
        cout << draw2 << endl;
        cout << draw3 << endl;
        cout << draw4 << endl;
        cout << draw5 << "\n\n";

        cout << " \t Total Valid Votes: \t\t " << fixed << setprecision(0) << total_valid_vote << endl;
        cout << " \t Total Invalid Votes:  \t\t"
             << " " << fixed << setprecision(0) << total_invalid_vote;
        cout << " \t Total votes received: \t\t" << fixed << setprecision(0) << total_all;
    }

    return 0;
}
