#ifndef PARAM
#define PARAM
#include <math.h>

#define FPS 60
//#define FULLSCREEN
#define LARGEUR_FENETRE 1280
#define HAUTEUR_FENETRE 720

const int ANIME_TIME= 90;
const float SCALE = 256;
const int NPARTMAX=pow(256,3);

const int STAR = 0;

char FOLDER[128] = "/home/deparis/data/FF_cond150/data/"; const int  NPROC = 64;
//char FOLDER[128] = "/home/deparis/data/L3_cond25/data/"; const int  NPROC = 128;
//char FOLDER[128] = "/home/deparis/Quartz/data/test256/data/"; const int  NPROC = 6; const int NSTEP = 1; int STEP_NUMBER[NSTEP]={0};

//const int NSTEP = 1; int STEP_NUMBER[NSTEP]={10};
const int NSTEP = 2; int STEP_NUMBER[NSTEP]={0,10};
//const int NSTEP = 6; int STEP_NUMBER[NSTEP]={0,2,4,6,8,10};
//const int NSTEP = 11; int STEP_NUMBER[NSTEP]={0,1,2,3,4,5,6,7,8,9,10};
//const int NSTEP = 6; int STEP_NUMBER[NSTEP]={5,6,7,8,9,10};


#endif
