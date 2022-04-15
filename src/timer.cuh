
class GpuTimer {
  public:
   GpuTimer() {}
   void timerStart() {
     cudaEventCreate(&start);
     cudaEventCreate(&stop);
     cudaEventRecord(start, NULL);
   }
   void timerStop() {
     cudaEventRecord(stop, NULL);
     cudaEventSynchronize(stop);
     cudaEventElapsedTime(&temp_time, start, stop);

     cudaEventDestroy(start);
     cudaEventDestroy(stop);
   }

   float getNsElapsed() { return temp_time * 1000; }

   float getSElapsed() { return temp_time * 0.001f; }
   ~GpuTimer(){};

  private:
   float temp_time = 0.0f;
   cudaEvent_t start, stop;
 };
