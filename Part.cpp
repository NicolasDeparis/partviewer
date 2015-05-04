#include "Part.h"

#include <cstdlib>
#include <cstdio>

#include <math.h>
#include <omp.h>


//#define STARS

Part::Part(int n){
	alloc(n);
}

Part::Part( char* folder, int  fileNumber, int  nproc){
  int npartmax = 128*128*128;// getNpart(folder,fileNumber,nproc);
	alloc(npartmax);
  read(folder, fileNumber, nproc);
 // setAge();
}

void Part::alloc(int npartmax){
	m_x   =  (float*)calloc(npartmax,sizeof(float));
	m_y   =  (float*)calloc(npartmax,sizeof(float));
	m_z   =  (float*)calloc(npartmax,sizeof(float));
	m_vx  =  (float*)calloc(npartmax,sizeof(float));
	m_vy  =  (float*)calloc(npartmax,sizeof(float));
	m_vz  =  (float*)calloc(npartmax,sizeof(float));
	m_idx =  (float*)calloc(npartmax,sizeof(float));
	m_age =  (float*)calloc(npartmax,sizeof(float));
}

void Part::read(char* folder, int  fileNumber, int  nproc){

	m_folder=folder;
	m_fileNumber=fileNumber;
	m_nproc=nproc;
	m_N=0;

	int i=0, nloc, dump;
	float mass, epot, ekin;
	char filename[256];
	FILE* f = NULL;

  //sprintf(filename, "%s%05d/part/part.%05d", m_folder,m_fileNumber, m_fileNumber);
	//printf("Reading %s\n",filename);

	for (int np=0; np<m_nproc; np++){
#ifdef STARS
    sprintf(filename, "%s%05d/star/star.%05d.p%05d", m_folder,m_fileNumber, m_fileNumber, np);
#else
    sprintf(filename, "%s%05d/part/part.%05d.p%05d", m_folder,m_fileNumber, m_fileNumber, np);
#endif // STARS

  //  printf("Reading %s\n",filename);
		f = fopen(filename, "rb");

		dump = fread (&nloc, sizeof(int)  ,1,f);
		dump = fread (&m_a,  sizeof(float),1,f);
		m_N += nloc;

		for(int ii=0; ii<nloc; ii++){
			dump = fread (&(m_x[i]),  sizeof(float), 1, f);
			dump = fread (&(m_y[i]),  sizeof(float), 1, f);
			dump = fread (&(m_z[i]),  sizeof(float), 1, f);
			dump = fread (&(m_vx[i]), sizeof(float), 1, f);
			dump = fread (&(m_vy[i]), sizeof(float), 1, f);
			dump = fread (&(m_vz[i]), sizeof(float), 1, f);
			dump = fread (&(m_idx[i]),sizeof(float), 1, f);
			dump = fread (&mass,      sizeof(float), 1, f);
			dump = fread (&epot,      sizeof(float), 1, f);
			dump = fread (&ekin,      sizeof(float), 1, f);
#ifdef STARS
			dump = fread (&(m_age[i]),sizeof(float), 1, f);
#endif // STARS
			i++;
		}
		fclose(f);
	}
}

int Part::getNpart(char* folder, int  fileNumber, int  nproc){

  int N=0;

  int np;
	for (np=0; np<nproc; np++){

    char filename[128];

#ifdef STARS
    sprintf(filename, "%s%05d/star/star.%05d.p%05d", folder,fileNumber, fileNumber, np);
#else
    sprintf(filename, "%s%05d/part/part.%05d.p%05d", folder,fileNumber, fileNumber, np);
#endif // STARS

    FILE* f = fopen(filename, "rb");
    int nloc;
		int dump = fread (&nloc, sizeof(int)  ,1,f);
		m_N += nloc;
		N += nloc;
    fclose(f);
  }
  return N;
}

void Part::setAge(){
	m_t = a2t(m_a);

	printf("\tt = %e yr \n\tZ = %f \n",m_t, 1./m_a -1.);
  #pragma omp parallel for
	for( int i=0; i< m_N; i++){
		m_age[i] = m_t - m_age[i];
		m_agemax = fmax(m_age[i], m_agemax );
//		printf("i %d\t idx %d \n", i, m_idx[i]);
	}
}

int   Part::getN()       {	return m_N;     }
float Part::getA()       {	return m_a;     }
float Part::getT()       {	return m_t;     }
float Part::getX  (int i){	return m_x[i];  }
float Part::getY  (int i){	return m_y[i];  }
float Part::getZ  (int i){	return m_z[i];  }
float Part::getVX  (int i){	return m_vx[i];  }
float Part::getVY  (int i){	return m_vy[i];  }
float Part::getVZ  (int i){	return m_vz[i];  }
float Part::getAge(int i){	return m_age[i];}
int   Part::getIdx(int i){	return (int)m_idx[i];}
float Part::getAgeMax(){	return m_agemax;}




void Part::move(float dt){
  #pragma omp parallel for
	for(int i=0; i< m_N; i++){
    m_x[i] += m_vx[i]*dt;
    if (m_x[i]>1) m_x[i]--;
    if (m_x[i]<0) m_x[i]++;

    m_y[i] += m_vy[i]*dt;
    if (m_y[i]>1) m_y[i]--;
    if (m_y[i]<0) m_y[i]++;

    m_z[i] += m_vz[i]*dt;
    if (m_z[i]>1) m_z[i]--;
    if (m_z[i]<0) m_z[i]++;
  }
}

void Part::setV( Part* stop, float dt){
  #pragma omp parallel for
	for(int i=0; i< m_N; i++){

    int id_current = getIdx(i);
    int id_stop = id_current; //stop.findIdx(id_current);

    float dx = stop->getX(id_stop) - m_x[i];
    float dy = stop->getY(id_stop) - m_y[i];
    float dz = stop->getZ(id_stop) - m_z[i];

    float lim = 0.5;

    if (dx> lim) dx-=1.;    if (dx<-lim) dx+=1.;
    if (dy> lim) dy-=1.;    if (dy<-lim) dy+=1.;
    if (dz> lim) dz-=1.;    if (dz<-lim) dz+=1.;

    m_vx[i] = dx/dt;
    m_vy[i] = dy/dt;
    m_vz[i] = dz/dt;

    //   ||.|||||||||||.|
    //      |||||||||||.|||.
    // .|||.|||||||||||
  }
}

void Part::interpPos(Part* start, float *t, int step, int time_max){

  float dt = t[step+1]-t[step];
  float t0 = t[step];

  #pragma omp parallel for
  for (int i=0; i<m_N; i++){

    float dt;
    float cur_t = m_age[i];

    if (cur_t>start->getAge(i)){

      float dx = start->getX(i) - m_x[i];
      float dy = start->getY(i) - m_y[i];
      float dz = start->getZ(i) - m_z[i];

      float lim = 0.5;

      if (dx> lim) dx-=1.;  if (dx<-lim) dx+=1.;
      if (dy> lim) dy-=1.;  if (dy<-lim) dy+=1.;
      if (dz> lim) dz-=1.;  if (dz<-lim) dz+=1.;

      dt = time_max;
      m_vx[i] = dx/dt;
      m_vy[i] = dy/dt;
      m_vz[i] = dz/dt;

      dt = (t[step+1]-cur_t )/ (t[step+1]-t[step]) * time_max;
      m_x[i] += dx/dt;
      m_y[i] += dy/dt;
      m_z[i] += dz/dt;


    }
  }


}


void Part::sort(Part* init){
  m_N = init->getN();
  #pragma omp parallel for
  for (int i=0;i<init->getN();i++){

      int id2sort = init->getIdx(i);

      m_idx[id2sort]=init->getIdx(i);
      m_x[id2sort]=init->getX(i);
      m_y[id2sort]=init->getY(i);
      m_z[id2sort]=init->getZ(i);

      m_age[id2sort]=init->getAge(i);
	}
}

void Part::copy(Part* init){
  m_N = init->getN();
  #pragma omp parallel for
  for (int i=0;i<m_N;i++){
      m_idx[i]  = (float)init->getIdx(i);
      m_x[i]    = init->getX(i);
      m_y[i]    = init->getY(i);
      m_z[i]    = init->getZ(i);
      //m_age[i]  = init.getAge(i);
	}
}

int Part::append(Part* next, int cur_part, float t){

  int npart_append=0;
  for(int i=cur_part;i<next->getN();i++){

      if(next->getAge(i)>t) continue;
      m_idx[i]  = (float)next->getIdx(i);
      m_x[i]    = next->getX(i);
      m_y[i]    = next->getY(i);
      m_z[i]    = next->getZ(i);

      m_vx[i]    = next->getVX(i);
      m_vy[i]    = next->getVY(i);
      m_vz[i]    = next->getVZ(i);

      m_age[i]  = next->getAge(i);

      npart_append++;
	}

	m_N+=npart_append;
	return npart_append;
}

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////



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
