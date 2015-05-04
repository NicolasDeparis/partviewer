#ifndef PART
#define PART

class Part{
private:
	char* 	m_folder;
	int  	m_fileNumber;
	int  	m_nproc;
	int  	m_npartmax;

	int 	m_N;
	float 	m_a;
	float	m_t;
	float m_agemax;

	float* 	m_x;
	float* 	m_y;
	float* 	m_z;

	float* 	m_vx;
	float* 	m_vy;
	float* 	m_vz;

	float* 	m_idx;
	float* 	m_age;

	void alloc(int npartmax);
	void setAge();
  int getNpart(char* folder, int  fileNumber, int  nproc);

  int findIdx(int idx);

public:
	Part(int N);
	Part(char* folder, int  fileNumber, int  nproc);

	void read(char* folder, int  fileNumber, int  nproc);

	int   getN();
	float getA();
	float getT();
	float getX(int i);
	float getY(int i);
	float getZ(int i);
	float getVX(int i);
	float getVY(int i);
	float getVZ(int i);
	int   getIdx(int i);
	float getAge(int i);
	float getAgeMax();
	void move(float t);
	void setV(Part* stop, float dt);
	void sort(Part* init);
	void copy(Part* init);
  int append(Part* next, int cur_part, float t);
  void interpPos(Part* start, float *t, int step, int time_max);
};

float a2t(float az);
float getdt(Part* p1, Part* p2);

#endif
