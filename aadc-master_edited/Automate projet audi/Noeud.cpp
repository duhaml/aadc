#include <string>
#include "Noeud.h"

using namespace std;

Noeud::Noeud(int NumeroDuNoeud, std::string NomDuNoeud) : m_NumeroDuNoeud(NumeroDuNoeud),m_NomDuNoeud(NomDuNoeud)
{

}

std::string Noeud::getNomNoeud()
{
    return m_NomDuNoeud;
}

int Noeud::getNumeroDuNoeud() const
{
    return m_NumeroDuNoeud;
}

