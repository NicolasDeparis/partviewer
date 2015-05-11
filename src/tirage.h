#ifndef __partViewerMac__tirage__
#define __partViewerMac__tirage__

#include <stdio.h>

void tirage_pos_1D( const float* X, int NPart, int offPos, const float* P, const float* L, size_t NBins, int NTirage, float* Xout, unsigned long int seed );

void tirage_SFR( const float* pos, int NPart, const float* P, const float* L, int NTirage, float rhoMin, float* posOut, unsigned long int seed );

#endif /* defined(__partViewerMac__tirage__) */
