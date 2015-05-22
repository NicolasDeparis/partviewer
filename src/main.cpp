 #include <cstdlib>
#include <SDL/SDL_opengl.h>

#ifdef CUDA
  #include <cuda_runtime.h>
  #include <cuda_gl_interop.h>
#endif // CUDA

#include "freeflycamera.h"
#include "scene.h"
#include "Part.h"
#include "physic.h"
#include "param.h"

void DrawGL(GLuint *vbo, int *N);

FreeFlyCamera * camera;

void stop()
{
    delete camera;
    SDL_Quit();}

void init_gl(void)
{
    unsigned int width = LARGEUR_FENETRE;
    unsigned int height = HAUTEUR_FENETRE;

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    gluPerspective(70,(double)width/height,0.001,1000);


    glEnable(GL_DEPTH_TEST);
    //glEnable(GL_POINT_SMOOTH);
}

int main(int argc, char *argv[]){

////////////////////////////////////////////////////////////////////////////////////////
    printf("Initializing\n");
////////////////////////////////////////////////////////////////////////////////////////

    SDL_Event event;
    const Uint32 time_per_frame = 1000/FPS;

    unsigned int width = LARGEUR_FENETRE;
    unsigned int height = HAUTEUR_FENETRE;

    SDL_Init(SDL_INIT_VIDEO);
    atexit(stop);

    Uint32 flags = SDL_SWSURFACE;

    SDL_Surface* surface = SDL_SetVideoMode(width, height, 32, SDL_OPENGL );

    init_gl();

    camera = new FreeFlyCamera(Vector3D(0.5,0.5,0.5), SCALE);

#ifdef CUDA
    cudaGLSetGLDevice(0);
    printf("CUDA enable\n");
#endif // CUDA


///////////////////////////////////////////////////////////////////////////////


    int time_max = ANIME_TIME * 1000/ NSTEP;


    const int NFIELD= 1;
    int Npart[NSTEP*NFIELD];
    GLuint all_vbo[2*NFIELD*NSTEP];

    Part **all_part =  (Part**)calloc(NSTEP*NFIELD,sizeof(Part*));
    float *dt =  (float*)calloc(NSTEP,sizeof(float));

    for(int i=0; i<NSTEP; i++){

if(1)
{
      all_part[i] = new Part(FOLDER, STEP_NUMBER[i], NPROC,  NPARTMAX, 0);
      all_part[i]->alloc_GPU(&all_vbo[NFIELD*i+0],NPARTMAX);
      all_part[i]->setColors();
      Npart[i] = all_part[i]->getN();
}

if(0){
      Part* part_amr = new Part(FOLDER, STEP_NUMBER[i], NPROC,  NPARTMAX, 1);
      all_part[i+1] = new Part( part_amr, NPARTMAX );
      delete part_amr;
      //all_part[i+1] = new Part(FOLDER, STEP_NUMBER[i], NPROC, STAR, NPARTMAX, 1);
      all_part[i+1]->alloc_GPU(&all_vbo[NFIELD*i+2 ],NPARTMAX);
      all_part[i+1]->setColors();
      Npart[i+1] = all_part[i+1]->getN();
}

if(0){
      all_part[i+2] = new Part(FOLDER, STEP_NUMBER[i], NPROC, NPARTMAX, 2);
      all_part[i+2]->alloc_GPU(&all_vbo[NFIELD*i+4],NPARTMAX);
      all_part[i+2]->setColors();
      Npart[i+2] = all_part[i+2]->getN();
}
    }

    for(int i=0; i<NSTEP-1; i++){
      dt[i] = getdt(all_part[i], all_part[i+1])/time_max;
      all_part[i]->setV(all_part[i+1], time_max);
    //    all_part[i+1]->interpPos(all_part[i], t_yr, i, time_max);
    }

////////////////////////////////////////////////////////////////////////////////////////////

    for(int i=0; i<NFIELD; i++)
    {
      int size = (int)Npart[0] * sizeof(float);
      //printf("Npart=%d\n",Npart[i]);
      glBindBuffer(GL_ARRAY_BUFFER, all_vbo[0]);
        glBufferSubData(GL_ARRAY_BUFFER, 0, 3*size,  all_part[0]->getPos());
        glBufferSubData(GL_ARRAY_BUFFER, 3*size, size,  all_part[0]->getColor());
      glBindBuffer(GL_ARRAY_BUFFER, 0);
    }


    all_part[0]->init_GPU_mem();

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
    printf("OK let's Go!!\n");
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

    int current_step=0;

    Uint32 stop_time, current_time,elapsed_time, total_time;
    Uint32 start_time = SDL_GetTicks();
    Uint32 last_time = start_time;

    for (;;){

      current_time = SDL_GetTicks();
      elapsed_time = current_time - last_time;
      last_time= current_time;
      total_time=current_time-start_time;
      float cur_t = all_part[current_step]->getTyr()+total_time*dt[current_step];
      float z = 1./all_part[current_step]->getA() -1.;
      int fps = (int)(1000./elapsed_time);
      char caption[256];
      sprintf(caption,"FPS %d z=%.2f, t=%.2fMyr" ,  fps ,z, cur_t/1e6);
      SDL_WM_SetCaption(caption, NULL);

////////////////////////////////////////////////////////////////////////////////////////

      while(SDL_PollEvent(&event)){
            switch(event.type){
                case SDL_QUIT:
                exit(0);
                break;

                case SDL_KEYDOWN:
                  switch (event.key.keysym.sym){
                      case SDLK_p:
                      break;

                      case SDLK_f:
                      SDL_WM_ToggleFullScreen(surface);
                      break;

                      case SDLK_a:
                      start_time = current_time;
                      last_time = start_time;
                      all_part[0]->sendVel();
                      break;

                      case SDLK_ESCAPE:
                      exit(0);
                      break;
                      default :
                      camera->OnKeyboard(event.key);
                  }
                break;

                case SDL_KEYUP:
                camera->OnKeyboard(event.key);
                break;

                case SDL_MOUSEMOTION:
                camera->OnMouseMotion(event.motion);
                break;

                case SDL_MOUSEBUTTONUP:
                case SDL_MOUSEBUTTONDOWN:
                camera->OnMouseButton(event.button);
                break;
            }
        }

        camera->animate(elapsed_time);

        if (current_time< (NSTEP-1)*time_max){
          all_part[current_step]->move(elapsed_time);
        }

////////////////////////////////////////////////////////////////////////////////////////


        if (current_step<NSTEP-1){
          if( total_time >= time_max* (current_step+1)){
       //   all_part[current_step]->sendVel();
          current_step++;
          }
        }


/*
        if((int)all_part[current_step]->getN() != (int)all_part[current_step+1]->getN()){
          //Npart[current_step]+=part_current->append(all_part[current_step+1], Npart[current_step], cur_t);
          printf("t=%f npart=%d \n",all_part[current_step]->getTyr(), (int)all_part[current_step]->getN());
        }
*/

        DrawGL(all_vbo,Npart);

        stop_time = SDL_GetTicks();
        if ((stop_time - last_time) < time_per_frame){
            SDL_Delay(time_per_frame - (stop_time - last_time));
        }
      }


    for(int i=0; i<NSTEP; i++){
      delete all_part[i];
    }
    free(dt);
    free(all_part);
    return 0;
}

void DrawGL(GLuint *vbo, int* N)
{
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity( );

    camera->look();

    render(vbo, N);

    glFlush();
    SDL_GL_SwapBuffers();
}


