#ifndef PART
#define PART

#include <SDL/SDL_opengl.h>


class Part{
private:
	char* 	m_folder;
	int  	m_fileNumber;
	int  	m_nproc;
	int  	m_npartmax;
	int   m_type;

	int 	m_N;
	float m_a;
	float	m_t;
	float	m_tyr;
	float m_agemax;
	int m_star;

	float *m_pos;
	float *m_vel;

	float *m_idx;
	float *m_age;
	float *m_mass;
	float *m_level;
	float *m_color;

	void alloc(int n);
	void setAge();
  int getNpart(char* folder, int  fileNumber, int  nproc);

public:
	Part(int n);
	Part(char* folder, int  fileNumber, int  nproc, int star, int n, int type);
  ~Part();

	void EMMA_read_part(char* folder, int  fileNumber, int  nproc);
	void EMMA_read_amr(char* folder, int  fileNumber);

	int   getN();
	float getA();
	float getT();
	float getTyr();
	float getX(int i);
	float getY(int i);
	float getZ(int i);
	float getVX(int i);
	float getVY(int i);
	float getVZ(int i);
	int   getIdx(int i);
	float getAge(int i);
	GLuint getVbo();

	float *getPos();
	float *getVel();
  float *getColor();
  void setColors();

	float getAgeMax();

  unsigned int m_vbo;

	void move(float t);
	void sort();
	void copy(Part* init);
	void setV(Part* stop, float dt);
  int append(Part* next, int cur_part, float t);
  void interpPos(Part* start, float *t, int step, int time_max);
};


#endif
