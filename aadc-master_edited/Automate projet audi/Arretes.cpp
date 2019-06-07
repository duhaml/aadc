#include "Arretes.h"
#include "Noeud.h"

using namespace std;

Arrete::Arrete(Noeud noeud_de_depart, Noeud noeud_darrivee) : m_noeud_de_depart(noeud_de_depart),m_noeud_darrivee(noeud_darrivee),m_disponible(true)
{

}

bool Arrete::getDisponible() const
{
    return m_disponible;
}

Noeud Arrete::getNoeudDepart() const
{
    return m_noeud_de_depart;
}

Noeud Arrete::getNoeudArrivee() const
{
    return m_noeud_darrivee;
}
