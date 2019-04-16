#include <iostream>
#include <string>

#include "Noeud.h"
#include "Arretes.h"

using namespace std;

int effectuer_transition(Arrete transition, int NumNoeudActu)
{
    bool dispo (transition.getDisponible());
    if (dispo == true)
    {
        Noeud noeuddepart (transition.getNoeudDepart());
        int noeudactu (noeuddepart.getNumeroDuNoeud());
        if (noeudactu == NumNoeudActu)
        {
            Noeud noeudarrivee (transition.getNoeudArrivee());
            int NewNumActu (noeudarrivee.getNumeroDuNoeud());
            std::string NomNewNoeud (noeudarrivee.getNomNoeud());
            cout << "Le nouvel etat est " + NomNewNoeud << endl;
            return NewNumActu;
        }
        else
        {
            cout << "Ce n'est pas le noeud actuel !" << endl;
            return NumNoeudActu;
        }
    }
    else
    {
        cout << "Transition non disponible" << endl;
        return NumNoeudActu;
    }
}

int main()
{
    //Construction de Noeuds

    Noeud VoitureEteinte(1, "Voiture Eteinte");
    Noeud VoitureAllumee(2, "Voiture Allumee");
    Noeud MarcheAvant(3, "Marche Avant");
    Noeud MarcheArriere(4, "Marche Arriere");

    int NumNoeudActu(1);

    //Construction des arretes

    Arrete Demarrage(VoitureEteinte, VoitureAllumee);
    Arrete Extinction(VoitureAllumee, VoitureEteinte);
    Arrete ModeMarcheAvant(VoitureAllumee, MarcheAvant);
    Arrete ModeMarcheArriere(VoitureAllumee, MarcheArriere);
    Arrete PointMortFromMAv(MarcheAvant, VoitureAllumee);
    Arrete PointMortFromMAr(MarcheArriere,VoitureAllumee);

    //Petite série de tests

    NumNoeudActu = effectuer_transition(Demarrage, NumNoeudActu);  //On allume la voiture
    NumNoeudActu = effectuer_transition(Extinction, NumNoeudActu); //On eteint la voiture
    NumNoeudActu = effectuer_transition(Extinction, NumNoeudActu); //On essaie de reeteindre la voiture (impossible)
    NumNoeudActu = effectuer_transition(Demarrage, NumNoeudActu); //On allume la voiture
    NumNoeudActu = effectuer_transition(ModeMarcheAvant, NumNoeudActu); //On met la marche avant
    NumNoeudActu = effectuer_transition(ModeMarcheArriere, NumNoeudActu); //On veut mettre la marche arrière sans repasser au point mort
    NumNoeudActu = effectuer_transition(PointMortFromMAv, NumNoeudActu); // On revient au point mort

    return 0;
}
