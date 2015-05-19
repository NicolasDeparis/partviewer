#include <math.h>
#include "Part.h"

float a2t(float a){

	float 	az = 1.0/(1.0+1.0* (1./a-1.) );

	float 	age = 0.;
	float 	adot;
	int    	n  = 1000 ;

	float 	H0 = 67;
	float 	h  = H0/100;

	float 	WM = 0.3175;
	float 	WR = 4.165e-5 /h/h;

	float	WV = 0.6825 ;
	float 	WK = 1.0 - WM - WR - WV;
	float 	Tyr = 977.8 ;

	for (int i=0; i< n; i++){
		a = az*(i+0.5)/n;
		adot = sqrt( WK + WM/a + WR/(a*a) + WV*a*a );
		age = age + 1./adot;
	}
	float zage = az*age/n;
	float zage_Gyr = (Tyr/ H0)*zage;

	return zage_Gyr*1e9;
}

float getdt(Part* p1, Part* p2){
  float a1 = p1->getA();
  float a2 = p2->getA();

  float t1 = a2t(a1);
  float t2 = a2t(a2);

  return t2-t1;
}
