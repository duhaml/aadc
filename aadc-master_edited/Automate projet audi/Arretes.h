#ifndef ARRETES_H_INCLUDED
#define ARRETES_H_INCLUDED

#include <string>
#include "Noeud.h"

class Arrete

{
    private :
        // attributs

        Noeud m_noeud_de_depart;
        Noeud m_noeud_darrivee;
        bool m_disponible;

    public :
        //Méthodes

        Arrete(Noeud noeud_de_depart, Noeud noeud_darrivee);
        bool getDisponible() const;
        Noeud getNoeudDepart() const;
        Noeud getNoeudArrivee() const;

};

#endif // ARRETES_H_INCLUDED
