#ifndef NOEUD_H_INCLUDED
#define NOEUD_H_INCLUDED

#include <string>

class Noeud

{
    private:

        //Attributs

        int m_NumeroDuNoeud;
        std::string m_NomDuNoeud;

    public:
        //Méthodes

        Noeud(int NumeroDuNoeud, std::string NomDuNoeud);
        int getNumeroDuNoeud() const;
        std::string getNomNoeud();
};

#endif // NOEUD_H_INCLUDED
