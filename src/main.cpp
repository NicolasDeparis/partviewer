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
    SDL_Quit();
}

void init_gl(void)
{
    unsigned int width = LARGEUR_FENETRE;
    unsigned int height = HAUTEUR_FENETRE;

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    gluPerspective(70,(double)width/height,0.001,1000);

//    glEnable(GL_DEPTH_TEST);
    //glEnable(GL_POINT_SMOOTH);

}

int main(int argc, char *argv[]){

// init GL

    SDL_Event event;
    const Uint32 time_per_frame = 1000/FPS;

    unsigned int width = LARGEUR_FENETRE;
    unsigned int height = HAUTEUR_FENETRE;

    Uint32 last_time,current_time,elapsed_time, total_time; //for time animation
    Uint32 start_time,stop_time; //for frame limit

    SDL_Init(SDL_INIT_VIDEO);
    atexit(stop);

    Uint32 flags = SDL_SWSURFACE;

    SDL_Surface* surface = SDL_SetVideoMode(width, height, 32, SDL_OPENGL );

//    initFullScreen(&width,&height);

    init_gl();


#ifdef CUDA
    printf("CUDA enable\n");
#endif // CUDA

    int  fileNumber;
    int time_max = ANIME_TIME * 1000/ NSTEP;
////////////////////////////////////////////////////////////////////////////////////////
    printf("Initializing\n");
////////////////////////////////////////////////////////////////////////////////////////

// allocating memory on CPU

    int Npart[NSTEP*2];

    GLuint all_vbo[4];

    Part **all_part =  (Part**)calloc(NSTEP*2,sizeof(Part*));
    float *dt =  (float*)calloc(NSTEP,sizeof(float));

    for(int i=0; i<NSTEP; i++){
      all_part[i] = new Part(FOLDER, STEP_NUMBER[i], NPROC, STAR, NPARTMAX, 0);
      all_part[i]->alloc_GPU(&all_vbo[2*i+0],NPARTMAX);
      all_part[i]->setColors();
      Npart[i] = all_part[i]->getN();

      all_part[i+1] = new Part(FOLDER, STEP_NUMBER[i], NPROC, STAR, NPARTMAX, 1);
      all_part[i+1]->alloc_GPU(&all_vbo[2*i+2],NPARTMAX);
      all_part[i+1]->setColors();
      Npart[i+1] = all_part[i+1]->getN();
    }

// Preliminar computations

    for(int i=0; i<NSTEP-1; i++){
      dt[i] = getdt(all_part[i], all_part[i+1])/time_max;
      all_part[i]->setV(all_part[i+1], time_max);
    //    all_part[i+1]->interpPos(all_part[i], t_yr, i, time_max);
    }


////////////////////////////////////////////////////////////////////////////////////////
    printf("OK let's Go!!\n");
////////////////////////////////////////////////////////////////////////////////////////



// allocating memory on GPU
    int size;
/*
  for(int i=0; i<2; i++){
    glBindBuffer(GL_ARRAY_BUFFER, all_vbo[2*i+0]);
      size = 3* NPARTMAX * sizeof(float);
      glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, all_vbo[2*i+1]);
      size = 4* NPARTMAX * sizeof(float);
      glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

#ifdef CUDA
    cudaGLRegisterBufferObject( vbo[0] );
    cudaMalloc((void **)&vbo,size);
#endif // CUDA
*/

    camera = new FreeFlyCamera(Vector3D(0.5,0.5,0.5), SCALE);

    start_time = SDL_GetTicks();
    last_time = start_time;



    for(int i=0; i<2*NSTEP; i++){
      size = (int)Npart[i] * sizeof(float);

      printf("Npart=%d\n",Npart[i]);

      glBindBuffer(GL_ARRAY_BUFFER, all_vbo[2*i+0]);
        glBufferData(GL_ARRAY_BUFFER, 3*size,  all_part[i]->getPos(), GL_STATIC_DRAW);
      glBindBuffer(GL_ARRAY_BUFFER, all_vbo[2*i+1]);
        glBufferData(GL_ARRAY_BUFFER, 4*size,  all_part[i]->getColor(), GL_STATIC_DRAW);
      glBindBuffer(GL_ARRAY_BUFFER, 0);
    }




    int current_step=0;

    for (;;){

      current_time = SDL_GetTicks();
      elapsed_time = current_time - last_time;
      last_time= current_time;
      total_time=current_time-start_time;
      float cur_t = all_part[current_step]->getTyr()+total_time*dt[current_step];
      //printf("%f\n",cur_t);

      float z = 1./all_part[current_step]->getA() -1.;
      int fps = (int)(1000./elapsed_time);
      char caption[256];
      sprintf(caption,"FPS %d z=%.2f, t=%.2fMyr" ,  fps ,z, cur_t/1e6);
      SDL_WM_SetCaption(caption, NULL);

////////////////////////////////////////////////////////////////////////////////////////

/*
        if((int)all_part[current_step]->getN() != (int)all_part[current_step+1]->getN()){
          //Npart[current_step]+=part_current->append(all_part[current_step+1], Npart[current_step], cur_t);
          printf("t=%f npart=%d \n",all_part[current_step]->getTyr(), (int)all_part[current_step]->getN());
        }
*/

////////////////////////////////////////////////////////////////////////////////////////

      //  start_time = SDL_GetTicks();

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

/*
        if (current_time< (NSTEP-1)*time_max){
          all_part[current_step]->move(elapsed_time);
          int size = 3* (int)all_part[current_step]->getN() * sizeof(float);
          glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
            glBufferData(GL_ARRAY_BUFFER, size,  all_part[current_step]->getPos(), GL_DYNAMIC_DRAW);
          glBindBuffer(GL_ARRAY_BUFFER, 0);
        }
*/
        DrawGL(all_vbo,  Npart );

        stop_time = SDL_GetTicks();
        if ((stop_time - last_time) < time_per_frame){
            SDL_Delay(time_per_frame - (stop_time - last_time));
        }

        if (current_step<NSTEP-1){
          if( total_time >= time_max* (current_step+1)){
          //printf("time=%d step=%d Npart=%d\n", total_time, current_step, (int)all_part[current_step]->getN());
          current_step++;
          //   printf("nstep=%d\n",current_step);
          }
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


