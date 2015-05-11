#include <GSL/gsl_histogram.h>
#include <GSL/gsl_randist.h>
#include <GSL/gsl_cdf.h>

#include <GSL/gsl_rng.h>
#include <math.h>
#include "tirage.h"
#include "Part.h"

void tirage_pos_1D( const float* X, int NPart, int offPos, const float* P, const float* L, size_t NBins, int NTirage, float* Xout, unsigned long int seed ) {
    
    // nombre de bins
    // size_t  NBins = 5;
    // vecteur de valeurs
    // float X[5]  = {0.05, 0.1, 0.3, 0.4, 1};
    // vecteur de poids
    // float P[5]  = {0.1, 0.1, 0.1, 0.1, 0.1};
    
    // creation de l'histogramme
    gsl_histogram * histX = gsl_histogram_alloc (NBins);
    // initialisation des limites des bins, cas uniforme
    gsl_histogram_set_ranges_uniform( histX, 0., 1.);
    
    // parcours des valeurs de X, et replisage de l'histogramme avec la pondération
    for(int i=0;i<NPart;i++){
        if(L[i]>9) { gsl_histogram_accumulate( histX, X[i*3+offPos], P[i] ); }
        //gsl_histogram_increment( histX, X[i*3+offPos] );
    }
    
    // gsl_histogram_fprintf(stdout, histX, "%g", "%g");
    
    // creation de la pdf
    gsl_histogram_pdf * pdfX = gsl_histogram_pdf_alloc(NBins);
    // initialise la pdf a partitr de l'histogramme
    gsl_histogram_pdf_init( pdfX, histX );
    
    // ecriture des histogrammes
    char filename[256];
    if(offPos==0)     { sprintf(filename, "/Users/gillet/partViewerMac/partViewerMac/histgramX.dat"); }
    else if(offPos==1){ sprintf(filename, "/Users/gillet/partViewerMac/partViewerMac/histgramY.dat"); }
    else if(offPos==2){ sprintf(filename, "/Users/gillet/partViewerMac/partViewerMac/histgramZ.dat"); }
    
    FILE* f = NULL;
    f = fopen(filename, "w+");
    gsl_histogram_fprintf(f, histX, "%g", "%g");
    fclose(f);
    
    // nombre de point du tirage
    // int NTirage = 5;
    
    // initialisation du random
    gsl_rng * foncRand = gsl_rng_alloc(gsl_rng_taus);
    //unsigned long int seed1 = 210289;
    //srand(seed);
    gsl_rng_set(foncRand, seed);
    
    // tirage du sample
    float monRand = 0.;
    for(int i=0;i<NTirage;i++){
        monRand = gsl_rng_uniform(foncRand);
        //monRand = rand() / RAND_MAX;
        Xout[i*3+offPos] = gsl_histogram_pdf_sample(pdfX, monRand);
        // printf("%f %f\n",monRand, Xout[i*3+offPos]);
    }
    
    // liberation de l'histogramme, la pdf et le random
    gsl_histogram_free(histX);
    gsl_histogram_pdf_free(pdfX);
    gsl_rng_free(foncRand);
    // free(sample);
    
}

void tirage_SFR( const float* pos, int NPart, const float* P, const float* L, int NTirage, float rhoSeuilMin, float* posOut, unsigned long int seed ) {
    
    // calcul du rho total
    float rhoTot = 0.;
    float rhoMax = P[0];
    float rhoMin = P[0];
    for(int i=0;i<NPart;i++){
        rhoTot+=P[i];
        if(P[i]>rhoMax) {rhoMax=P[i];}
        if(P[i]<rhoMin) {rhoMin=P[i];}
    }
    
    printf("rhoMin=%f, rhoMax=%f, rhoTot=%f\n",rhoMin,rhoMax,rhoTot);
    
    // initialisation du random
    gsl_rng * foncRand = gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(foncRand, seed);
    
    int lambda = 0;
    int nPoint = 0;
    int cptTirage = 0;
    float dx = 0.;
    float TH = 0.;
    float sinPH = 0.;
    float R = 0.;
    
    /*
    float Rt[100];
    for(int i=0;i<100;i++){
        Rt[i]= pow( (float)i/100, 3 );
    }
    static gsl_ran_discrete_t *pdfR = NULL;
    pdfR = gsl_ran_discrete_preproc (100, Rt);
    unsigned long monRand = gsl_ran_discrete ( foncRand, pdfR );s
    */
    
    for(int i=0;i<NPart;i++){
        
        // Determine le nombre de part a tirer dans la cell
        if(P[i]>rhoSeuilMin) {
            lambda = ceil( (float)NTirage *pow(0.5,L[i]*3) *log(P[i]) / log(rhoMax) );
            nPoint = lambda;
            // tirage de poisson de moyenne lambda
            // nPoint = gsl_ran_poisson (foncRand, lambda);
        }
        else {
            lambda=0;
            nPoint=0;
        }
        
        // calcul de la position des particules tirees
        if(nPoint) {
            // si on depasse le nomnre de particules
            if( (cptTirage+nPoint)>NTirage ){
                nPoint=(cptTirage+nPoint) - NTirage;
                printf("Tirage: trop de particulre tirees! Revoire la loi!\n");
                i=NPart;
            }
            for(int p=cptTirage;p<(cptTirage+nPoint);p++){
                dx = pow(0.5,L[i]);
                
                // tirage NON UNIFORM dans la sphere de rayon 2dx
                TH = gsl_rng_uniform(foncRand)*M_PI*2;
                sinPH = (gsl_rng_uniform(foncRand)*2.-1);
                R = gsl_rng_uniform(foncRand)*dx*4;
                
                posOut[p*3+0] = pos[i*3+0] + R*sinPH*cos(TH);
                posOut[p*3+1] = pos[i*3+1] + R*sinPH*sin(TH);
                posOut[p*3+2] = pos[i*3+2] + R*cos( asin(sinPH) );
            }
            cptTirage+=nPoint;
        }
    }
    if(cptTirage<NTirage) {
        printf("Tirage: pas assez de tirage! Revoire la loi!\n");
        printf("%d / %d\n",cptTirage,NTirage);
    }
    
    // gsl_ran_discrete_free (pdfR);
}











