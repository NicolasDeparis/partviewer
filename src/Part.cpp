#include "Part.h"

#include <cstdlib>
#include <cstdio>
#include <math.h>
#include <omp.h>
#include <SDL/SDL_opengl.h>

//#include <GL/gl.h>
//#include <GL/glu.h>



Part::Part(int n){
	alloc(n);
}

Part::Part( char* folder, int  fileNumber, int  nproc, int s, int star){
  m_star = star;
  int npartmax = 128*128*128;// getNpart(folder,fileNumber,nproc);
	alloc(npartmax);
  read(folder, fileNumber, nproc);
  //read_amr(folder, fileNumber);

 // setAge();
}

float *Part::getPos()    {	return m_pos;     }
float *Part::getVel()    {	return m_vel;     }
int   Part::getN()       {	return m_N;     }
float Part::getA()       {	return m_a;     }
float Part::getT()       {	return m_t;     }
float Part::getX  (int i){	return m_pos[3*i+0];  }
float Part::getY  (int i){	return m_pos[3*i+1];  }
float Part::getZ  (int i){	return m_pos[3*i+2];  }
float Part::getVX (int i){	return m_vel[3*i+0];  }
float Part::getVY (int i){	return m_vel[3*i+1];  }
float Part::getVZ (int i){	return m_vel[3*i+2];  }
float Part::getAge(int i){	return m_age[i];}
int   Part::getIdx(int i){	return (int)m_idx[i];}
float Part::getAgeMax(){	return m_agemax;}

void Part::alloc(int npartmax){
	m_pos =  (float*)calloc(3*npartmax,sizeof(float));
	m_vel =  (float*)calloc(3*npartmax,sizeof(float));
	m_idx =  (float*)calloc(npartmax,sizeof(float));
	m_age =  (float*)calloc(npartmax,sizeof(float));
	m_mass=  (float*)calloc(npartmax,sizeof(float));
	m_level=  (float*)calloc(npartmax,sizeof(float));
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

  sprintf(filename, "%s%05d/part/part.%05d", m_folder,m_fileNumber, m_fileNumber);
	printf("Reading %s\n",filename);

	for (int np=0; np<m_nproc; np++){
    if(m_star)
      sprintf(filename, "%s%05d/star/star.%05d.p%05d", m_folder,m_fileNumber, m_fileNumber, np);
    else
      sprintf(filename, "%s%05d/part/part.%05d.p%05d", m_folder,m_fileNumber, m_fileNumber, np);


  //  printf("Reading %s\n",filename);
		f = fopen(filename, "rb");

		dump = fread (&nloc, sizeof(int)  ,1,f);
		dump = fread (&m_a,  sizeof(float),1,f);
		m_N += nloc;

		for(int ii=0; ii<nloc; ii++){
			dump = fread(&(m_pos[3*i]), sizeof(float), 3, f);
			dump = fread(&(m_vel[3*i]), sizeof(float), 3, f);

			dump = fread (&(m_idx[i]),sizeof(float), 1, f);

			dump = fread(&mass,         sizeof(float), 1, f);
			dump = fread(&epot,         sizeof(float), 1, f);
			dump = fread(&ekin,         sizeof(float), 1, f);
      if(m_star)
			dump = fread(&(m_age[i]),   sizeof(float), 1, f);

			i++;
		}
		fclose(f);
	}

	sort();
}

void Part::read_amr(char* folder, int  fileNumber){

	m_folder=folder;
	m_fileNumber=fileNumber;
	m_nproc=0;

	int dump;
	char filename[256];
	FILE* f = NULL;

  sprintf(filename, "%s%05d/grid/alloct.%05d.field.d", m_folder,m_fileNumber, m_fileNumber);
	printf("Reading %s\n",filename);

  f = fopen(filename, "rb");

  dump = fread (&m_N,sizeof(int)  ,1,f);
  printf("Npart=%d\n",m_N);

  dump = fread (&m_a,sizeof(float),1,f);

  for(int i=0; i<m_N; i++){
    dump = fread(&(m_pos[3*i]),sizeof(float), 3, f);
    dump = fread(&(m_level[i]),sizeof(float), 1, f);
    dump = fread(&(m_mass[i] ),sizeof(float), 1, f);
  }

  fclose(f);
  printf("Read OK\n");

}

int Part::getNpart(char* folder, int  fileNumber, int  nproc){

  int N=0;

  int np;
	for (np=0; np<nproc; np++){

    char filename[128];

if(m_star)  sprintf(filename, "%s%05d/star/star.%05d.p%05d", folder,fileNumber, fileNumber, np);
else        sprintf(filename, "%s%05d/part/part.%05d.p%05d", folder,fileNumber, fileNumber, np);

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


void Part::move(float dt){
  #pragma omp parallel for
	for(int i=0; i< 3*m_N; i++){
    m_pos[i] += m_vel[i]*dt;
    if (m_pos[i]>1) m_pos[i]--;
    if (m_pos[i]<0) m_pos[i]++;
  }
}

void Part::setV( Part* stop, float dt){
  #pragma omp parallel for
	for(int i=0; i< m_N; i++){

    int id_current = getIdx(i);
    int id_stop = id_current; //stop.findIdx(id_current);

    float dx = stop->getX(id_stop) - m_pos[3*i+0];
    float dy = stop->getY(id_stop) - m_pos[3*i+1];
    float dz = stop->getZ(id_stop) - m_pos[3*i+2];

    float lim = 0.5;

    if (dx> lim) dx-=1.;    if (dx<-lim) dx+=1.;
    if (dy> lim) dy-=1.;    if (dy<-lim) dy+=1.;
    if (dz> lim) dz-=1.;    if (dz<-lim) dz+=1.;

    m_vel[3*i+0] = dx/dt;
    m_vel[3*i+1] = dy/dt;
    m_vel[3*i+2] = dz/dt;

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

      float dx = start->getX(i) - m_pos[3*i+0];
      float dy = start->getY(i) - m_pos[3*i+1];
      float dz = start->getZ(i) - m_pos[3*i+2];

      float lim = 0.5;

      if (dx> lim) dx-=1.;  if (dx<-lim) dx+=1.;
      if (dy> lim) dy-=1.;  if (dy<-lim) dy+=1.;
      if (dz> lim) dz-=1.;  if (dz<-lim) dz+=1.;

      dt = time_max;
      m_vel[3*i+0] = dx/dt;
      m_vel[3*i+1] = dy/dt;
      m_vel[3*i+2] = dz/dt;

      dt = (t[step+1]-cur_t )/ (t[step+1]-t[step]) * time_max;
      m_pos[3*i+0] += dx/dt;
      m_pos[3*i+1] += dy/dt;
      m_pos[3*i+2] += dz/dt;

    }
  }


}


void Part::sort(){

  static float *idx_tmp =  (float*)calloc(m_N,sizeof(float));
  static float *pos_tmp =  (float*)calloc(3*m_N,sizeof(float));
  static float *age_tmp =  (float*)calloc(m_N,sizeof(float));

  #pragma omp parallel for
  for (int i=0;i<m_N;i++){

      int id2sort = m_idx[i];

      idx_tmp[id2sort]=m_idx[i];

      pos_tmp[3*id2sort+0]=m_pos[3*i+0];
      pos_tmp[3*id2sort+1]=m_pos[3*i+1];
      pos_tmp[3*id2sort+2]=m_pos[3*i+2];

      age_tmp[id2sort]=m_age[i];
	}


// copy

  #pragma omp parallel for
  for (int i=0;i<m_N;i++){
    m_idx[i]  = idx_tmp[i];
  }
	#pragma omp parallel for
  for (int i=0;i<3*m_N;i++){
    m_pos[i]  = pos_tmp[i];
  }
}

int Part::append(Part* next, int cur_part, float t){

  int npart_append=0;
  for(int i=cur_part;i<next->getN();i++){

      if(next->getAge(i)>t) continue;
      m_idx[i]  = (float)next->getIdx(i);
      m_pos[3*i+0]    = next->getX(i);
      m_pos[3*i+1]    = next->getY(i);
      m_pos[3*i+2]    = next->getZ(i);

      m_vel[3*i+0]    = next->getVX(i);
      m_vel[3*i+1]    = next->getVY(i);
      m_vel[3*i+2]    = next->getVZ(i);

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
